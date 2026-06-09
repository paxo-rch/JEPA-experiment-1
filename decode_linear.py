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
DECODER_PATH = "decoder_latest.pt"
ENCODER_PATH = "encoder_latest.pt"
FPS_TARGET = 20

# ==========================================
# 1. Pixel Decoder Architecture
# ==========================================
class PixelDecoder(nn.Module):
    def __init__(self, embed_dim=config.EMBEDDING_SIZE, patch_size=config.PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        # Maps the 768-D vector back to 3 * 16 * 16 = 768 pixel values
        self.head = nn.Linear(embed_dim, 3 * patch_size * patch_size)

    def forward(self, x):
        # x is from encoder: (B, Hp, Wp, 768)
        pixels = self.head(x) 
        
        # Rearrange the patch grids back into a standard image format (B, 3, H, W)
        img = rearrange(pixels, 'b hp wp (c p1 p2) -> b c (hp p1) (wp p2)', 
                        p1=self.patch_size, p2=self.patch_size, c=3)
        return img

# ==========================================
# 2. Image Processing Utilities
# ==========================================
def preprocess_frame(frame_bgr):
    """Converts a raw OpenCV BGR frame to the normalized PyTorch tensor."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (config.WIDTH, config.HEIGHT))
    
    img_array = frame_resized.astype(np.float32) / 255.0
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    img_array = (img_array - mean) / std
    
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).to(DEVICE)
    return img_tensor, frame_resized

def tensor_to_bgr(tensor):
    """Converts a normalized PyTorch tensor (3, H, W) back to an OpenCV BGR image."""
    tensor = tensor.to(dtype=torch.float32) # Ensure it's in float32 for the math
    img_array = tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    
    # Denormalize
    img_array = (img_array * std) + mean
    img_array = np.clip(img_array, 0, 1) * 255.0
    
    # Convert RGB back to BGR for OpenCV
    return cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)

# ==========================================
# 3. Training the Decoder
# ==========================================
def train_decoder_on_video(encoder, video_path, sample_frames=800, epochs=5, batch_size=32):
    print(f"\n--- Training Decoder ---")
    print(f"Extracting up to {sample_frames} frames from '{video_path}'...")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Could not open video {video_path} or it is empty.")
    
    step = max(1, total_frames // sample_frames)
    tensors = []
    
    for i in tqdm(range(min(sample_frames, total_frames))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        img_tensor, _ = preprocess_frame(frame)
        tensors.append(img_tensor.cpu()) # Keep on CPU for now to save VRAM
        
    cap.release()
    
    dataset = TensorDataset(torch.stack(tensors))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    decoder = PixelDecoder().to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    
    print(f"Training Linear Decoder for {epochs} epochs...")
    decoder.train()
    
    autocast_dev = 'cuda' if 'cuda' in DEVICE else 'cpu'
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch_images,) in loader:
            batch_images = batch_images.to(DEVICE)
            
            # 1. Get frozen embeddings from JEPA
            with torch.no_grad():
                with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                    embeddings = encoder.encode(batch_images)
            
            # 2. Reconstruct pixels
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                reconstructed = decoder(embeddings)
                loss = F.mse_loss(reconstructed, batch_images)
            
            # 3. Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")
        
    torch.save(decoder.state_dict(), DECODER_PATH)
    print(f"Saved trained decoder to {DECODER_PATH}!\n")
    return decoder

# ==========================================
# 4. Live Playback
# ==========================================
def play_video():
    print(f"Using device: {DEVICE}")

    # 1. Load JEPA Encoder (Frozen)
    encoder = Encoder(
        dim=config.EMBEDDING_SIZE, 
        heads=8, 
        depth=6, 
        img_size=(config.WIDTH, config.HEIGHT), 
        patch_size=config.PATCH_SIZE
    ).to(DEVICE)
    
    if not os.path.exists(ENCODER_PATH):
        print(f"Error: {ENCODER_PATH} not found!")
        return
        
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    encoder.requires_grad_(False)
    print("Successfully loaded JEPA Encoder.")

    # 2. Load or Train Pixel Decoder
    if os.path.exists(DECODER_PATH):
        print(f"Found existing decoder at {DECODER_PATH}. Loading...")
        decoder = PixelDecoder().to(DEVICE)
        decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    else:
        print(f"No decoder found. Training a new one on {VIDEO_PATH}...")
        decoder = train_decoder_on_video(encoder, VIDEO_PATH)
        
    decoder.eval()
    decoder.requires_grad_(False)

    # 3. Start Live Playback
    cap = cv2.VideoCapture(VIDEO_PATH)
    window_name = "JEPA AE Reconstruction"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.WIDTH * 2, config.HEIGHT)
    
    frame_delay_ms = int(1000 / FPS_TARGET)
    autocast_dev = 'cuda' if 'cuda' in DEVICE else 'cpu'

    print("Playing video... Press 'q' to quit.")
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
            
        img_tensor, frame_resized = preprocess_frame(frame)
        
        # Inference
        with torch.no_grad():
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                # Encode (Original -> 768D Embeddings)
                features = encoder.encode(img_tensor.unsqueeze(0))
                # Decode (768D Embeddings -> RGB Pixels)
                reconstructed = decoder(features)
                
        # Format images for OpenCV
        orig_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        recon_bgr = tensor_to_bgr(reconstructed[0])
        
        # Stack side-by-side
        combined_frame = np.hstack((orig_bgr, recon_bgr))
        
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