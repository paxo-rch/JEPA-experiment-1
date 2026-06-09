import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from config import Config as config

class RoPE2D(nn.Module):
    def __init__(self, dim, height, width, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.height = height
        self.width = width
        
        d_h = dim // 2
        d_w = dim - d_h

        inv_freq_h = 1.0 / (theta ** (torch.arange(0, d_h, 2).float() / d_h))
        inv_freq_w = 1.0 / (theta ** (torch.arange(0, d_w, 2).float() / d_w))
        
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)

        freqs_h = torch.einsum('i, j -> ij', h_coords, inv_freq_h)
        freqs_w = torch.einsum('i, j -> ij', w_coords, inv_freq_w)

        emb_h = torch.cat((freqs_h, freqs_h), dim=-1) 
        emb_w = torch.cat((freqs_w, freqs_w), dim=-1) 

        full_grid = torch.zeros(height, width, dim)
        full_grid[:, :, :d_h] = emb_h.unsqueeze(1).expand(-1, width, -1)
        full_grid[:, :, d_h:] = emb_w.unsqueeze(0).expand(height, -1, -1)

        self.register_buffer("cos", full_grid.cos()) 
        self.register_buffer("sin", full_grid.sin()) 

    def get_masked_freqs_variable(self, mask):
        B = mask.shape[0]
        
        cos_list = []
        sin_list = []
        valid_lens = []
        
        for i in range(B):
            m = mask[i] 
            c = self.cos[m]
            cos_list.append(c)
            sin_list.append(self.sin[m])
            valid_lens.append(c.shape[0])

        cos_padded = nn.utils.rnn.pad_sequence(cos_list, batch_first=True) 
        sin_padded = nn.utils.rnn.pad_sequence(sin_list, batch_first=True) 
        
        # Generate boolean attention mask for SDPA (True = attend, False = ignore padding)
        max_len = cos_padded.shape[1]
        attn_mask = torch.arange(max_len, device=mask.device).unsqueeze(0) < torch.tensor(valid_lens, device=mask.device).unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) # Broadcastable to (B, 1, 1, max_len)
        
        return cos_padded, sin_padded, attn_mask

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rope(x, cos, sin):
        if x.ndim == 4:
            cos = cos.unsqueeze(1) # Align with (B, 1, T, D) to match (B, H, T, D)
            sin = sin.unsqueeze(1)
        return (x * cos) + (RoPE2D.rotate_half(x) * sin)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, cos, sin, attn_mask):
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=self.heads), qkv)
        
        q = RoPE2D.apply_rope(q, cos, sin)
        k = RoPE2D.apply_rope(k, cos, sin)
        
        # Apply scaled dot product attention with explicit tracking padding mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, heads, depth, img_size=(11, 20)):
        super().__init__()
        self.rope_cache = RoPE2D(dim=dim // heads, height=img_size[0], width=img_size[1])
        self.layers = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm_f = nn.LayerNorm(dim) # <--- ADD THIS

    def forward(self, x, mask): 
        cos, sin, attn_mask = self.rope_cache.get_masked_freqs_variable(mask)
        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)
        return self.norm_f(x) # <--- APPLY THIS


def reconstruct_grid(encoder_outputs, mask, img_hp, img_wp, dim, mask_token=None):
    """
    encoder_outputs: (B, max_N_context, D) - Output from the encoder (padded)
    mask:            (B, Hp, Wp)           - Boolean mask used for pruning (True = kept)
    img_hp, img_wp:  int, int              - Grid height and width
    dim:             int                   - Embedding dimension (D)
    mask_token:      Parameter or Tensor   - (Optional) Learnable mask token to fill empty slots
    """
    B = mask.shape[0]

    if mask_token is not None:
        # Cast to match bfloat16/float16 if autocast is active
        grid = mask_token.to(dtype=encoder_outputs.dtype).expand(B, img_hp, img_wp, dim).clone()
    else:
        # Pass the dtype from encoder_outputs
        grid = torch.zeros(B, img_hp, img_wp, dim, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        
    for i in range(B):
        num_valid_tokens = mask[i].sum().item()
        valid_tokens = encoder_outputs[i, :num_valid_tokens] # Shape: (num_valid_tokens, D)
        grid[i][mask[i]] = valid_tokens

    return grid # Shape: (B, Hp, Wp, D)

class Encoder(nn.Module):
    def __init__(self, dim=config.EMBEDDING_SIZE, heads=8, depth=6, img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.img_w, self.img_h = img_size
        self.img_wp, self.img_hp = self.img_w // patch_size, self.img_h // patch_size

        self.encoder = Transformer(dim, heads, depth, (self.img_hp, self.img_wp))
        self.conv_in = nn.Conv2d(3, dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0)

    def encode_masked(self, img, mask):
        B, C, H, W = img.size()
        x = self.conv_in(img) # (B, D, Hp, Wp)
        
        x = rearrange(x, 'b d h w -> b h w d')
        
        x_list = [x[i][mask[i]] for i in range(B)]
        x_padded = nn.utils.rnn.pad_sequence(x_list, batch_first=True) # (B, max_N, D)

        x_out = self.encoder(x_padded, mask)

        x_grid = reconstruct_grid(x_out, mask, self.img_hp, self.img_wp, self.dim)

        return x_grid
    
    def encode(self, img):
            B, C, H, W = img.size()
            x = self.conv_in(img) # (B, D, Hp, Wp)
            
            x = rearrange(x, 'b d h w -> b (h w) d')
            
            mask = torch.ones((B, self.img_hp, self.img_wp), dtype=torch.bool, device=img.device)
            
            x_out = self.encoder(x, mask)

            x_out = rearrange(x_out, 'b (h w) d -> b h w d', h=self.img_hp, w=self.img_wp)

            return x_out


class Predictor(nn.Module):
    def __init__(self, dim=config.EMBEDDING_SIZE, heads=8, depth=6, img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.img_w, self.img_h = img_size
        self.img_wp, self.img_hp = self.img_w // patch_size, self.img_h // patch_size

        self.rope_cache = RoPE2D(dim=dim // heads, height=self.img_hp, width=self.img_wp)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, dim))
        
        self.layers = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])

    def predict(self, x, mask_encoder, mask_predictor):
        B, H, W, D = x.size()

        mask_tokens = self.mask_token.to(dtype=x.dtype).expand(B, H, W, D)
        x = torch.where(mask_encoder.unsqueeze(-1), x, mask_tokens)

        x = rearrange(x, 'b h w d -> b (h w) d')

        cos = rearrange(self.rope_cache.cos.unsqueeze(0).expand(B, -1, -1, -1), 'b h w d -> b (h w) d')
        sin = rearrange(self.rope_cache.sin.unsqueeze(0).expand(B, -1, -1, -1), 'b h w d -> b (h w) d')

        mask_used = mask_encoder | mask_predictor
        attn_mask = rearrange(mask_used, 'b h w -> b 1 1 (h w)')

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask=attn_mask)

        x = rearrange(x, 'b (h w) d -> b h w d', h=H, w=W)
        
        return x