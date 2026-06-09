import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange
from tqdm import tqdm

from config import Config as config
from model import Encoder

DEVICE = config.DEVICE
VIDEO_PATH = "video.mp4"
ADVANCED_DECODER_PATH = "advanced_decoder_latest.pt"
ENCODER_PATH = "encoder_latest.pt"
FPS_TARGET = 20

# ==========================================
# 1. Advanced Non-Linear Pixel Decoder
# ==========================================
class AdvancedPixelDecoder(nn.Module):
    def __init__(self, embed_dim=config.EMBEDDING_SIZE, patch_size=config.PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        
        # A 3-layer MLP that can learn to "un-compress" highly abstract semantic concepts
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            
            # Output the raw pixels
            nn.Linear(embed_dim * 2, 3 * patch_size * patch_size)
        )

    def forward(self, x):
        # x is from encoder: (B, Hp, Wp, 768)
        pixels = self.net(x) 
        
        # Rearrange the patch grids back into a standard image format (B, 3, H, W)
        img = rearrange(pixels, 'b hp wp (c p1 p2) -> b c (hp p1) (wp p2)', 
                        p1=self.patch_size, p2=self.patch_size, c=3)
        return img

# ==========================================
# 2. Image Processing Utilities
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

def tensor_to_bgr(tensor):
    img_array = tensor.to(dtype=torch.float32).permute(1, 2, 0).cpu().numpy()
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    
    img_array = (img_array * std) + mean
    img_array = np.clip(img_array, 0, 1) * 255.0
    return cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)

# ==========================================
# 3. Training the Advanced Decoder
# ==========================================
def train_advanced_decoder(encoder, video_path, sample_frames=1200, epochs=10, batch_size=32):
    print(f"\n--- Training Advanced Non-Linear Decoder ---")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Could not open video {video_path}")
    
    step = max(1, total_frames // sample_frames)
    tensors = []
    
    print(f"Extracting {sample_frames} frames to train the MLP...")
    for i in tqdm(range(min(sample_frames, total_frames))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret: break
        img_tensor, _ = preprocess_frame(frame)
        tensors.append(img_tensor.cpu())
        
    cap.release()
    
    dataset = TensorDataset(torch.stack(tensors))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    decoder = AdvancedPixelDecoder().to(DEVICE)
    # Using AdamW with a bit of weight decay to keep the MLP healthy
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print(f"Training MLP Decoder for {epochs} epochs...")
    decoder.train()
    autocast_dev = 'cuda' if 'cuda' in DEVICE else 'cpu'
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch_images,) in loader:
            batch_images = batch_images.to(DEVICE)
            
            # 1. Get highly abstract, frozen embeddings from JEPA
            with torch.no_grad():
                with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                    embeddings = encoder.encode(batch_images)
            
            # 2. Reconstruct pixels via the 3-layer MLP
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                reconstructed = decoder(embeddings)
                loss = F.mse_loss(reconstructed, batch_images)
            
            # 3. Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Reconstruction Loss: {epoch_loss/len(loader):.4f}")
        
    torch.save(decoder.state_dict(), ADVANCED_DECODER_PATH)
    print(f"Saved trained advanced decoder to {ADVANCED_DECODER_PATH}!\n")
    return decoder

# ==========================================
# 4. Live Playback
# ==========================================
def play_video():
    print(f"Using device: {DEVICE}")

    # 1. Load JEPA Encoder (Frozen)
    encoder = Encoder(
        dim=config.EMBEDDING_SIZE, heads=8, depth=6, 
        img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE
    ).to(DEVICE)
    
    if not os.path.exists(ENCODER_PATH):
        print(f"Error: {ENCODER_PATH} not found!")
        return
        
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval().requires_grad_(False)
    print("Successfully loaded Fully-Trained JEPA Encoder.")

    # 2. Load or Train Advanced Pixel Decoder
    if os.path.exists(ADVANCED_DECODER_PATH):
        print(f"Found existing advanced decoder at {ADVANCED_DECODER_PATH}. Loading...")
        decoder = AdvancedPixelDecoder().to(DEVICE)
        decoder.load_state_dict(torch.load(ADVANCED_DECODER_PATH, map_location=DEVICE))
    else:
        print(f"No advanced decoder found. Training a new MLP on {VIDEO_PATH}...")
        decoder = train_advanced_decoder(encoder, VIDEO_PATH)
        
    decoder.eval().requires_grad_(False)

    # 3. Start Live Playback
    cap = cv2.VideoCapture(VIDEO_PATH)
    window_name = "JEPA Advanced MLP Reconstruction"
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
                # Encode (Original -> 768D Highly Abstract Embeddings)
                features = encoder.encode(img_tensor.unsqueeze(0))
                # Decode (768D Embeddings -> MLP -> RGB Pixels)
                reconstructed = decoder(features)
                
        # Format images for OpenCV
        orig_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        recon_bgr = tensor_to_bgr(reconstructed[0])
        
        # Stack side-by-side
        combined_frame = np.hstack((orig_bgr, recon_bgr))
        
        # Add a label to make the video clear
        cv2.putText(combined_frame, "Original Video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_frame, "MLP Semantic Decoding", (config.WIDTH + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show Image
        cv2.imshow(window_name, combined_frame)
        
        # Dynamic FPS locking
        process_time_ms = (time.time() - start_time) * 1000
        wait_time = max(1, frame_delay_ms - int(process_time_ms))
        
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video()