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
from model import Encoder, Predictor

DEVICE = config.DEVICE
VIDEO_PATH = "video.mp4"
EXPORT_PATH = "export.mp4"
TRUE_DECODER_PATH = "true_semantic_decoder.pt"
ENCODER_PATH = "encoder_latest.pt"
PREDICTOR_PATH = "predictor_latest.pt"
FPS_TARGET = 20

# ==========================================
# 1. Pixel Decoder Architecture
# ==========================================
class PixelDecoder(nn.Module):
    def __init__(self, embed_dim=config.EMBEDDING_SIZE, patch_size=config.PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.head = nn.Linear(embed_dim, 3 * patch_size * patch_size)
        self.head2 = nn.Linear(3 * patch_size * patch_size, 3 * patch_size * patch_size)
        self.head3 = nn.Linear(3 * patch_size * patch_size, 3 * patch_size * patch_size)

    def forward(self, x):
        pixels = self.head(x)
        pixels = self.head2(pixels)
        pixels = self.head3(pixels)
        img = rearrange(pixels, 'b hp wp (c p1 p2) -> b c (hp p1) (wp p2)', 
                        p1=self.patch_size, p2=self.patch_size, c=3)
        return img

# ==========================================
# 2. Image Processing & Masking
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

def generate_random_masks(b, hp, wp, mask_ratio=0.6):
    noise = torch.rand(b, hp, wp, device=DEVICE)
    target_mask = noise < mask_ratio
    context_mask = ~target_mask
    return context_mask, target_mask

# ==========================================
# 3. Training the TRUE Semantic Decoder
# ==========================================
def train_true_decoder(encoder, predictor, video_path, sample_frames=1000, epochs=8, batch_size=32):
    print(f"\n--- Training True Semantic Decoder ---")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // sample_frames)
    
    tensors = []
    print("Extracting frames...")
    for i in tqdm(range(min(sample_frames, total_frames))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret: break
        img_tensor, _ = preprocess_frame(frame)
        tensors.append(img_tensor.cpu())
    cap.release()
    
    dataset = TensorDataset(torch.stack(tensors))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    decoder = PixelDecoder().to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    
    print(f"Training for {epochs} epochs strictly on Predictor Hallucinations...")
    decoder.train()
    autocast_dev = 'cuda' if 'cuda' in DEVICE else 'cpu'
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch_images,) in loader:
            batch_images = batch_images.to(DEVICE)
            b = batch_images.size(0)
            
            context_mask, target_mask = generate_random_masks(b, encoder.img_hp, encoder.img_wp, mask_ratio=0.6)
            
            with torch.no_grad():
                with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                    encoded_context = encoder.encode_masked(batch_images, context_mask)
                    predicted_embeddings = predictor.predict(encoded_context, context_mask, target_mask)
            
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                reconstructed_img = decoder(predicted_embeddings)
                recon_patches = rearrange(reconstructed_img, 'b c (h p1) (w p2) -> b h w (c p1 p2)', p1=config.PATCH_SIZE, p2=config.PATCH_SIZE)
                target_patches = rearrange(batch_images, 'b c (h p1) (w p2) -> b h w (c p1 p2)', p1=config.PATCH_SIZE, p2=config.PATCH_SIZE)
                loss = F.mse_loss(recon_patches[target_mask], target_patches[target_mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | True Semantic Loss: {epoch_loss/len(loader):.4f}")
        
    torch.save(decoder.state_dict(), TRUE_DECODER_PATH)
    print("Saved true semantic decoder!\n")
    return decoder

# ==========================================
# 4. Processing & Exporting Video
# ==========================================
def play_video():
    print(f"Using device: {DEVICE}")

    encoder = Encoder(dim=config.EMBEDDING_SIZE, heads=8, depth=6, img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE).to(DEVICE)
    predictor = Predictor(dim=config.EMBEDDING_SIZE, heads=8, depth=6, img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE).to(DEVICE)
    
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
    encoder.eval().requires_grad_(False)
    predictor.eval().requires_grad_(False)

    if os.path.exists(TRUE_DECODER_PATH):
        print(f"Found {TRUE_DECODER_PATH}. Loading...")
        decoder = PixelDecoder().to(DEVICE)
        decoder.load_state_dict(torch.load(TRUE_DECODER_PATH, map_location=DEVICE))
    else:
        decoder = train_true_decoder(encoder, predictor, VIDEO_PATH)
        
    decoder.eval().requires_grad_(False)

    Hp, Wp = encoder.img_hp, encoder.img_wp
    y1, y2 = Hp // 4, 3 * Hp // 4
    x1, x2 = Wp // 4, 3 * Wp // 4
    
    target_mask = torch.zeros((1, Hp, Wp), dtype=torch.bool, device=DEVICE)
    target_mask[:, y1:y2, x1:x2] = True
    context_mask = ~target_mask 

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Video Writer Initialization
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Vertical stack configuration: height is multiplied by 2
    out = cv2.VideoWriter(EXPORT_PATH, fourcc, FPS_TARGET, (config.WIDTH, config.HEIGHT * 2))
    
    window_name = "True Semantic Hallucination"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.WIDTH, config.HEIGHT * 2)
    
    frame_delay_ms = int(1000 / FPS_TARGET)
    autocast_dev = 'cuda' if 'cuda' in DEVICE else 'cpu'

    print("Processing video and writing to export.mp4... Press 'q' to stop.")
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret: break
            
        img_tensor, frame_resized = preprocess_frame(frame)
        batch_images = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                encoded_context = encoder.encode_masked(batch_images, context_mask)
                predicted_grid = predictor.predict(encoded_context, context_mask, target_mask)
                final_embeddings = torch.where(target_mask.unsqueeze(-1), predicted_grid, encoded_context)
                reconstructed = decoder(final_embeddings)
                
        recon_bgr = tensor_to_bgr(reconstructed[0])
        orig_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        
        pt1 = (x1 * config.PATCH_SIZE, y1 * config.PATCH_SIZE)
        pt2 = (x2 * config.PATCH_SIZE, y2 * config.PATCH_SIZE)
        cv2.rectangle(orig_bgr, pt1, pt2, (0, 0, 255), 2)
        cv2.rectangle(recon_bgr, pt1, pt2, (0, 0, 255), 2)
        
        # Stack vertically: original on top, reconstruction on bottom
        combined_frame = np.vstack((orig_bgr, recon_bgr))
        
        # Save frame to export file
        out.write(combined_frame)
        
        cv2.imshow(window_name, combined_frame)
        
        process_time_ms = (time.time() - start_time) * 1000
        wait_time = max(1, frame_delay_ms - int(process_time_ms))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished exporting video to {EXPORT_PATH}")

if __name__ == "__main__":
    play_video()