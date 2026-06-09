import torch
import cv2
import numpy as np
import time
from config import Config as config
from model import Encoder

DEVICE = config.DEVICE
VIDEO_PATH = "video.mp4" # Put your video path here
FPS_TARGET = 20

def preprocess_frame(frame_bgr):
    """Converts a raw OpenCV BGR frame to the normalized PyTorch tensor."""
    # Convert BGR to RGB and resize to match model expected input
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (config.WIDTH, config.HEIGHT))
    
    # Normalize to [0, 1] then apply ImageNet stats
    img_array = frame_resized.astype(np.float32) / 255.0
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # HWC to CHW and add batch dimension
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img_tensor, frame_resized

def compute_global_pca(encoder, video_path, sample_size=32):
    """
    Samples frames from the video to compute a single, stable PCA projection matrix.
    This stops the colors from flickering wildly from frame to frame.
    """
    print(f"Sampling {sample_size} frames to compute global semantic colors...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError("Could not open video or video is empty.")
        
    step = max(1, total_frames // sample_size)
    
    sample_tensors = []
    for i in range(sample_size):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        img_tensor, _ = preprocess_frame(frame)
        sample_tensors.append(img_tensor)
        
    cap.release()
    
    # Batch the sampled frames
    batch = torch.cat(sample_tensors, dim=0) # (32, 3, H, W)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda' if 'cuda' in DEVICE else 'cpu', dtype=torch.bfloat16):
            features = encoder.encode(batch) # (32, Hp, Wp, 768)
            
    features_flat = features.view(-1, config.EMBEDDING_SIZE).float()
    
    # Run PCA on the diverse batch
    U, S, V = torch.pca_lowrank(features_flat, q=3)
    
    # Compute the global min and max so we can consistently normalize the colors
    global_proj = torch.matmul(features_flat, V[:, :3])
    p_min = global_proj.min(dim=0).values
    p_max = global_proj.max(dim=0).values
    
    print("Global PCA computed!")
    return V[:, :3], p_min, p_max

def play_video():
    print(f"Using device: {DEVICE}")

    # 1. Initialize Encoder
    encoder = Encoder(
        dim=config.EMBEDDING_SIZE, 
        heads=8, 
        depth=6, 
        img_size=(config.WIDTH, config.HEIGHT), 
        patch_size=config.PATCH_SIZE
    ).to(DEVICE)
    
    # 2. Load Checkpoint
    try:
        encoder.load_state_dict(torch.load('encoder_latest.pt', map_location=DEVICE))
        print("Successfully loaded 'encoder_latest.pt'")
    except FileNotFoundError:
        print("Error: 'encoder_latest.pt' not found.")
        return

    encoder.eval()

    # 3. Get global PCA matrix for consistent coloring
    V_matrix, p_min, p_max = compute_global_pca(encoder, VIDEO_PATH)

    # 4. Start Live Playback
    cap = cv2.VideoCapture(VIDEO_PATH)
    cv2.namedWindow("JEPA Semantic Visualization", cv2.WINDOW_NORMAL)
    # Resize window to fit side-by-side (WIDTH * 2, HEIGHT)
    cv2.resizeWindow("JEPA Semantic Visualization", config.WIDTH * 2, config.HEIGHT)
    
    frame_delay_ms = int(1000 / FPS_TARGET)

    print("Playing video... Press 'q' to quit.")
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
            
        img_tensor, frame_resized = preprocess_frame(frame)
        
        # Encode Frame
        with torch.no_grad():
            with torch.autocast(device_type='cuda' if 'cuda' in DEVICE else 'cpu', dtype=torch.bfloat16):
                features = encoder.encode(img_tensor) # (1, Hp, Wp, 768)
                
        Hp, Wp = features.shape[1], features.shape[2]
        
        # Apply global PCA projection
        features_flat = features.view(-1, config.EMBEDDING_SIZE).float()
        pca_3d = torch.matmul(features_flat, V_matrix)
        
        # Normalize strictly between [0, 1] using global bounds
        pca_3d = (pca_3d - p_min) / (p_max - p_min)
        pca_3d = torch.clamp(pca_3d, 0, 1)
        
        # Reshape to grid
        pca_img = pca_3d.view(Hp, Wp, 3).cpu().numpy()
        
        # Format PCA output for OpenCV
        # Convert [0, 1] RGB to [0, 255] BGR
        pca_bgr_small = cv2.cvtColor((pca_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Upscale the small patch grid to the original image dimensions using nearest neighbor
        # (This keeps the distinct 16x16 patch blocks visible)
        pca_bgr_large = cv2.resize(pca_bgr_small, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        # Format Original image for OpenCV
        orig_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        
        # Stack side-by-side
        combined_frame = np.hstack((orig_bgr, pca_bgr_large))
        
        # Show Image
        cv2.imshow("JEPA Semantic Visualization", combined_frame)
        
        # Calculate dynamic delay to maintain exact FPS (accounting for inference time)
        process_time_ms = (time.time() - start_time) * 1000
        wait_time = max(1, frame_delay_ms - int(process_time_ms))
        
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video()