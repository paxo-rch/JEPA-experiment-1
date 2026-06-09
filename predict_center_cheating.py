import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from config import Config as config
from model import Encoder, Predictor

DEVICE = config.DEVICE
VIDEO_PATH = "video.mp4"
ENCODER_PATH = "encoder_latest.pt"
PREDICTOR_PATH = "predictor_latest.pt"
DECODER_PATH = "decoder_latest.pt"
FPS_TARGET = 20

# ==========================================
# 1. Pixel Decoder Architecture
# ==========================================
class PixelDecoder(nn.Module):
    def __init__(self, embed_dim=config.EMBEDDING_SIZE, patch_size=config.PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.head = nn.Linear(embed_dim, 3 * patch_size * patch_size)

    def forward(self, x):
        pixels = self.head(x) 
        img = rearrange(pixels, 'b hp wp (c p1 p2) -> b c (hp p1) (wp p2)', 
                        p1=self.patch_size, p2=self.patch_size, c=3)
        return img

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

def tensor_to_bgr(tensor):
    img_array = tensor.to(dtype=torch.float32).permute(1, 2, 0).cpu().numpy()
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    
    img_array = (img_array * std) + mean
    img_array = np.clip(img_array, 0, 1) * 255.0
    return cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)

# ==========================================
# 3. Live Playback & Hallucination
# ==========================================
def play_video():
    print(f"Using device: {DEVICE}")

    # 1. Load Models
    encoder = Encoder(dim=config.EMBEDDING_SIZE, heads=8, depth=6, 
                      img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE).to(DEVICE)
    predictor = Predictor(dim=config.EMBEDDING_SIZE, heads=8, depth=6, 
                          img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE).to(DEVICE)
    decoder = PixelDecoder().to(DEVICE)
    
    # Ensure checkpoints exist
    for path in [ENCODER_PATH, PREDICTOR_PATH, DECODER_PATH]:
        if not os.path.exists(path):
            print(f"Error: Required checkpoint '{path}' not found!")
            return

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    
    encoder.eval().requires_grad_(False)
    predictor.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)

    # 2. Define the Center Target Mask
    Hp, Wp = encoder.img_hp, encoder.img_wp
    
    # Let's mask the middle 50% of the screen
    y1, y2 = Hp // 4, 3 * Hp // 4
    x1, x2 = Wp // 4, 3 * Wp // 4
    
    target_mask = torch.zeros((1, Hp, Wp), dtype=torch.bool, device=DEVICE)
    target_mask[:, y1:y2, x1:x2] = True
    context_mask = ~target_mask # Everything outside the center box

    # 3. Setup Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    window_name = "JEPA Center Hallucination"
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
        batch_images = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                
                # A. Encode ONLY the outer context border
                encoded_context = encoder.encode_masked(batch_images, context_mask)
                
                # B. Predict the missing center box
                predicted_grid = predictor.predict(encoded_context, context_mask, target_mask)
                
                # C. Stitch them together! 
                # Use Encoder outputs for the outside (sharp/cheating), Predictor outputs for the inside (hallucinated)
                final_embeddings = torch.where(target_mask.unsqueeze(-1), predicted_grid, encoded_context)
                
                # D. Decode to pixels
                reconstructed = decoder(final_embeddings)
                
        # Format for OpenCV
        recon_bgr = tensor_to_bgr(reconstructed[0])
        orig_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        
        # Draw a red rectangle on both images to show the hallucination zone boundary
        pt1 = (x1 * config.PATCH_SIZE, y1 * config.PATCH_SIZE)
        pt2 = (x2 * config.PATCH_SIZE, y2 * config.PATCH_SIZE)
        
        # Draw on Original
        cv2.rectangle(orig_bgr, pt1, pt2, (0, 0, 255), 2)
        # Draw on Reconstruction
        cv2.rectangle(recon_bgr, pt1, pt2, (0, 0, 255), 2)
        
        # Stack side-by-side
        combined_frame = np.hstack((orig_bgr, recon_bgr))
        cv2.imshow(window_name, combined_frame)
        
        process_time_ms = (time.time() - start_time) * 1000
        wait_time = max(1, frame_delay_ms - int(process_time_ms))
        
        if cv2.waitKey(wait_time) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video()