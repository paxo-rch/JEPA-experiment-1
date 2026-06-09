import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import Config as config
from model import Encoder

DEVICE = config.DEVICE
VIDEO_PATH = "video.mp4"
AE_PATH = "embedding_ae_latest.pt"
ENCODER_PATH = "encoder_latest.pt"
FPS_TARGET = 20

# ==========================================
# 1. Embedding Autoencoder (768 -> 3 -> 768)
# ==========================================
class EmbeddingAutoencoder(nn.Module):
    def __init__(self, embed_dim=config.EMBEDDING_SIZE, hidden_dim=256):
        super().__init__()
        
        # Squeezes 768D down to 3D
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()  # Forces the 3 dimensions into [0, 1] so they act exactly like RGB colors!
        )
        
        # Attempts to rebuild the 768D vector from just 3 numbers
        self.decoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        # x shape: (B, Hp, Wp, 768)
        rgb_bottleneck = self.encoder(x)        # Shape: (B, Hp, Wp, 3)
        reconstructed_embeds = self.decoder(rgb_bottleneck) # Shape: (B, Hp, Wp, 768)
        return rgb_bottleneck, reconstructed_embeds

    def get_rgb_map(self, x):
        return self.encoder(x)

# ==========================================
# 2. Image Processing
# ==========================================
def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (config.WIDTH, config.HEIGHT))
    
    img_array = frame_resized.astype(np.float32) / 255.0
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    img_array = (img_array - mean) / std
    
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).to(DEVICE)
    return img_tensor, frame_resized

# ==========================================
# 3. Training the Bottleneck AE
# ==========================================
def train_ae_on_video(encoder, video_path, sample_frames=1000, epochs=10, batch_size=64):
    print(f"\n--- Training 3-Channel Bottleneck AE ---")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // sample_frames)
    
    tensors = []
    print("Extracting embeddings from video...")
    for i in tqdm(range(min(sample_frames, total_frames))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret: break
        
        img_tensor, _ = preprocess_frame(frame)
        tensors.append(img_tensor.cpu())
    cap.release()
    
    dataset = TensorDataset(torch.stack(tensors))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    ae = EmbeddingAutoencoder().to(DEVICE)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print(f"Training for {epochs} epochs to map 768D -> 3D -> 768D...")
    ae.train()
    
    autocast_dev = 'cuda' if 'cuda' in DEVICE else 'cpu'
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch_images,) in loader:
            batch_images = batch_images.to(DEVICE)
            
            # Get original JEPA embeddings
            with torch.no_grad():
                with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                    target_embeddings = encoder.encode(batch_images)
            
            # Forward pass through AE
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                _, reconstructed = ae(target_embeddings)
                
                # We normalize before MSE loss to prevent magnitude explosion (just like JEPA!)
                target_norm = F.layer_norm(target_embeddings, (config.EMBEDDING_SIZE,))
                recon_norm = F.layer_norm(reconstructed, (config.EMBEDDING_SIZE,))
                loss = F.mse_loss(recon_norm, target_norm)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Reconstruction Loss: {epoch_loss/len(loader):.4f}")
        
    torch.save(ae.state_dict(), AE_PATH)
    print(f"Saved AE bottleneck model to {AE_PATH}!\n")
    return ae

# ==========================================
# 4. Live Playback
# ==========================================
def play_video():
    print(f"Using device: {DEVICE}")

    # 1. Load JEPA
    encoder = Encoder(
        dim=config.EMBEDDING_SIZE, heads=8, depth=6, 
        img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE
    ).to(DEVICE)
    
    if not os.path.exists(ENCODER_PATH):
        print(f"Error: {ENCODER_PATH} not found!")
        return
        
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    encoder.requires_grad_(False)

    # 2. Load or Train AE
    if os.path.exists(AE_PATH):
        print(f"Loading existing AE from {AE_PATH}...")
        ae = EmbeddingAutoencoder().to(DEVICE)
        ae.load_state_dict(torch.load(AE_PATH, map_location=DEVICE))
    else:
        ae = train_ae_on_video(encoder, VIDEO_PATH)
        
    ae.eval()
    ae.requires_grad_(False)

    # 3. Setup Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    window_name = "JEPA Semantic Bottleneck (768D -> 3D RGB)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.WIDTH * 2, config.HEIGHT)
    
    frame_delay_ms = int(1000 / FPS_TARGET)
    autocast_dev = 'cuda' if 'cuda' in DEVICE else 'cpu'

    print("Playing video... Press 'q' to quit.")
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret: break
            
        img_tensor, frame_resized = preprocess_frame(frame)
        
        # Inference
        with torch.no_grad():
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                # 1. Extract 768D Embeddings
                features = encoder.encode(img_tensor.unsqueeze(0))
                # 2. Extract 3D RGB Bottleneck
                rgb_bottleneck = ae.get_rgb_map(features) # Shape: (1, Hp, Wp, 3), Values: [0.0, 1.0]
                
        # Format Bottleneck for OpenCV (H, W, C)

        rgb_map = rgb_bottleneck[0].to(dtype=torch.float32).cpu().numpy()
        rgb_map_bgr = cv2.cvtColor((rgb_map * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Scale back up to original size using NEAREST interpolation to see the patches clearly
        semantic_map_large = cv2.resize(rgb_map_bgr, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        # Original Image
        orig_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        
        # Stack side-by-side
        combined_frame = np.hstack((orig_bgr, semantic_map_large))
        
        cv2.imshow(window_name, combined_frame)
        
        process_time_ms = (time.time() - start_time) * 1000
        wait_time = max(1, frame_delay_ms - int(process_time_ms))
        
        if cv2.waitKey(wait_time) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video()