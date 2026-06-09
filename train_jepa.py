import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config as config
from dataset import FrameDataset
from model import *
import copy

from utils import update_ema

DEVICE = config.DEVICE
BATCH_SIZE = 64
LR = 5e-5
N_TARGETS = 4  # Number of target blocks to predict per image (Standard JEPA)

def generate_jepa_masks(b, n, h, w, device="cpu"):
    """
    Generates N target blocks per image.
    The context mask is defined as everything that is NOT a target block.
    """
    grid_y = torch.arange(h, device=device).view(1, 1, h, 1)
    grid_x = torch.arange(w, device=device).view(1, 1, 1, w)

    # Limit maximum size of target blocks to ensure we don't accidentally mask the whole image
    max_h = max(2, h // 2)
    max_w = max(2, w // 2)

    # 1. Generate top-left coordinates for N targets
    y1 = torch.randint(0, h, (b, n, 1, 1), device=device)
    x1 = torch.randint(0, w, (b, n, 1, 1), device=device)

    # 2. Generate random heights and widths
    dh = torch.randint(1, max_h + 1, (b, n, 1, 1), device=device)
    dw = torch.randint(1, max_w + 1, (b, n, 1, 1), device=device)

    # 3. Calculate bottom-right coordinates (clipped to grid boundaries)
    y2 = torch.clamp(y1 + dh, max=h)
    x2 = torch.clamp(x1 + dw, max=w)

    # Shape: (B, N, H, W) -> True for patches to predict
    target_masks = (grid_y >= y1) & (grid_y < y2) & (grid_x >= x1) & (grid_x < x2)

    # 4. Context mask is strictly the inverse of the union of all targets
    union_targets = target_masks.any(dim=1) # (B, H, W)
    context_mask = ~union_targets           # (B, H, W)
    
    return context_mask, target_masks


def train():
    encoder = Encoder(dim=config.EMBEDDING_SIZE, heads=8, depth=6, img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE).to(DEVICE)
    predictor = Predictor(dim=config.EMBEDDING_SIZE, heads=8, depth=6, img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE).to(DEVICE)
    
    try:
        encoder.load_state_dict(torch.load('encoder_latest.pt', map_location=DEVICE))
        predictor.load_state_dict(torch.load('predictor_latest.pt', map_location=DEVICE))

        print("Loaded existing encoder weights.")
    except FileNotFoundError:
        print("No existing encoder weights found, starting from scratch.")

    encoder_ema = copy.deepcopy(encoder).to(DEVICE)
    encoder_ema.eval()
    encoder_ema.requires_grad_(False)

    dataset = FrameDataset("../dataset/frames")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=LR)
    optimizer_predictor = torch.optim.Adam(predictor.parameters(), lr=LR)

    encoder.train()
    predictor.train()

    autocast_dev = 'cuda' if 'cuda' in DEVICE else 'cpu'

    for epoch in range(10):
        pbar = tqdm(loader)
        for batch_idx, images in enumerate(pbar):
            images = images.to(DEVICE)
            curr_batch_size = images.size(0)
            
            optimizer_encoder.zero_grad()
            optimizer_predictor.zero_grad()
            
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                
                # 1. Generate JEPA multi-masks
                context_mask, target_masks = generate_jepa_masks(
                    curr_batch_size, 
                    N_TARGETS, 
                    encoder.img_hp, 
                    encoder.img_wp, 
                    device=DEVICE
                )

                # 2. Get Ground Truth Embeddings from EMA
                with torch.no_grad():
                    ground_truth = encoder_ema.encode(images) # (B, Hp, Wp, D)
                
                # 3. Encode ONLY the context patches
                encoded = encoder.encode_masked(images, context_mask) # (B, Hp, Wp, D)
                
                # 4. Expand Batch to run predictor N times per image (B -> B * N)
                # Expand Context Embeddings: (B, Hp, Wp, D) -> (B*N, Hp, Wp, D)
                encoded_expanded = encoded.repeat_interleave(N_TARGETS, dim=0)
                
                # Expand Ground Truth: (B, Hp, Wp, D) -> (B*N, Hp, Wp, D)
                ground_truth_expanded = ground_truth.repeat_interleave(N_TARGETS, dim=0)
                
                # Expand Context Mask: (B, Hp, Wp) -> (B*N, Hp, Wp)
                context_mask_expanded = context_mask.repeat_interleave(N_TARGETS, dim=0)
                
                # Flatten Target Masks: (B, N, Hp, Wp) -> (B*N, Hp, Wp)
                target_masks_flat = target_masks.view(-1, encoder.img_hp, encoder.img_wp)

                # 5. Predict the target blocks
                predicted = predictor.predict(
                    encoded_expanded, 
                    mask_encoder=context_mask_expanded, 
                    mask_predictor=target_masks_flat
                )

                pred_norm = F.layer_norm(predicted, normalized_shape=(config.EMBEDDING_SIZE,))
                gt_norm = F.layer_norm(ground_truth_expanded, normalized_shape=(config.EMBEDDING_SIZE,))

                sq_error = torch.pow(pred_norm - gt_norm, 2).mean(dim=-1)
                loss = sq_error[target_masks_flat].mean()

            loss.backward()
            optimizer_encoder.step()
            optimizer_predictor.step()

            update_ema(encoder, encoder_ema, tau=0.99)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            with open("loss_history.txt", "a") as f:
                f.write(f"{loss.item():.6f}\n")
            
            if (batch_idx + 1) % 200 == 0:
                torch.save(encoder.state_dict(), "encoder_latest.pt")
                torch.save(predictor.state_dict(), "predictor_latest.pt")


if __name__ == "__main__":
    train()