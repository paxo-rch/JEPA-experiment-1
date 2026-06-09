import torch
import cv2
import numpy as np
import time
from config import Config as config
from model import Encoder

DEVICE = config.DEVICE
VIDEO_PATH = "video.mp4" 
FPS_TARGET = 20

def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (config.WIDTH, config.HEIGHT))
    
    img_array = frame_resized.astype(np.float32) / 255.0
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    img_array = (img_array - mean) / std
    
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img_tensor, frame_resized

def compute_dynamic_pca(encoder, video_path, sample_size=64):
    print(f"Sampling {sample_size} frames to compute Per-Patch Mean...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    step = max(1, total_frames // sample_size)
    
    sample_tensors = []
    for i in range(sample_size):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret: break
        img_tensor, _ = preprocess_frame(frame)
        sample_tensors.append(img_tensor)
        
    cap.release()
    
    batch = torch.cat(sample_tensors, dim=0) # (B, 3, H, W)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda' if 'cuda' in DEVICE else 'cpu', dtype=torch.bfloat16):
            features = encoder.encode(batch) # (B, Hp, Wp, 768)
            
    features = features.float()
    
    # === YOUR BRILLIANT IDEA ===
    # 1. Calculate the mean of EACH specific patch coordinate across time
    # This isolates the static positional encoding of the network
    spatial_mean = features.mean(dim=0, keepdim=True) # Shape: (1, Hp, Wp, 768)
    
    # 2. Subtract the positional encoding from the features to leave ONLY the dynamic semantics
    semantic_features = features - spatial_mean
    
    # 3. Now run PCA on these centered features!
    features_flat = semantic_features.view(-1, config.EMBEDDING_SIZE)
    U, S, V = torch.pca_lowrank(features_flat, q=3)
    
    # Compute bounds for consistent coloring
    global_proj = torch.matmul(features_flat, V[:, :3])
    p_min = global_proj.min(dim=0).values
    p_max = global_proj.max(dim=0).values
    
    print("Dynamic PCA computed!")
    # Return the PCA projection matrix, AND the spatial mean grid to subtract during playback!
    return V[:, :3], spatial_mean, p_min, p_max

def play_video():
    encoder = Encoder(dim=config.EMBEDDING_SIZE, heads=8, depth=6, 
                      img_size=(config.WIDTH, config.HEIGHT), patch_size=config.PATCH_SIZE).to(DEVICE)
    encoder.load_state_dict(torch.load('encoder_latest.pt', map_location=DEVICE))
    encoder.eval().requires_grad_(False)

    # Get the Projection Matrix AND the Spatial Mean
    V_matrix, spatial_mean, p_min, p_max = compute_dynamic_pca(encoder, VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    cv2.namedWindow("Dynamic Semantic PCA", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dynamic Semantic PCA", config.WIDTH * 2, config.HEIGHT)
    
    frame_delay_ms = int(1000 / FPS_TARGET)
    autocast_dev = 'cuda' if 'cuda' in DEVICE else 'cpu'

    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret: break
            
        img_tensor, frame_resized = preprocess_frame(frame)
        
        with torch.no_grad():
            with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
                features = encoder.encode(img_tensor).float() # (1, Hp, Wp, 768)
                
        Hp, Wp = features.shape[1], features.shape[2]
        
        # === APPLY YOUR MATH TO THE LIVE FRAME ===
        # Subtract the static positional encoding grid
        centered_features = features - spatial_mean
        
        # Apply PCA
        features_flat = centered_features.view(-1, config.EMBEDDING_SIZE)
        pca_3d = torch.matmul(features_flat, V_matrix)
        
        # Normalize to RGB [0, 1]
        pca_3d = (pca_3d - p_min) / (p_max - p_min)
        pca_3d = torch.clamp(pca_3d, 0, 1)
        
        pca_img = pca_3d.view(Hp, Wp, 3).cpu().numpy()
        pca_bgr_small = cv2.cvtColor((pca_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        pca_bgr_large = cv2.resize(pca_bgr_small, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        orig_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        combined_frame = np.hstack((orig_bgr, pca_bgr_large))
        
        cv2.imshow("Dynamic Semantic PCA", combined_frame)
        
        process_time_ms = (time.time() - start_time) * 1000
        wait_time = max(1, frame_delay_ms - int(process_time_ms))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video()