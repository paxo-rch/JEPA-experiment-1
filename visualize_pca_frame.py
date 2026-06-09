import torch
import matplotlib.pyplot as plt
import numpy as np
from config import Config as config
from dataset import FrameDataset
from model import Encoder

DEVICE = config.DEVICE

def visualize():
    print(f"Using device: {DEVICE}")

    # 1. Initialize Encoder
    encoder = Encoder(
        dim=config.EMBEDDING_SIZE, 
        heads=8, 
        depth=6, 
        img_size=(config.WIDTH, config.HEIGHT), 
        patch_size=config.PATCH_SIZE
    ).to(DEVICE)
    
    # 2. Load the trained checkpoint
    try:
        encoder.load_state_dict(torch.load('encoder_latest.pt', map_location=DEVICE))
        print("Successfully loaded 'encoder_latest.pt'")
    except FileNotFoundError:
        print("Error: 'encoder_latest.pt' not found. Please ensure the model has saved a checkpoint.")
        return

    encoder.eval()

    # 3. Load Dataset and get the first image
    dataset = FrameDataset("../dataset/frames")
    print(f"Dataset loaded with {len(dataset)} images.")
    
    # Grab the very first image
    image_tensor = dataset[120000] # Shape: (3, H, W)
    image_batch = image_tensor.unsqueeze(0).to(DEVICE) # Shape: (1, 3, H, W)

    # 4. Get the Embeddings
    with torch.no_grad():
        with torch.autocast(device_type='cuda' if 'cuda' in DEVICE else 'cpu', dtype=torch.bfloat16):
            features = encoder.encode(image_batch) # Shape: (1, Hp, Wp, 768)

    Hp, Wp = features.shape[1], features.shape[2]
    
    # 5. PCA Dimensionality Reduction (768 -> 3)
    # Flatten grid to list of patches: (Hp * Wp, 768)
    features_flat = features.view(-1, config.EMBEDDING_SIZE).float() # Cast to float32 for PCA math stability
    
    # Run PCA to extract the 3 most dominant semantic features
    """U, S, V = torch.pca_lowrank(features_flat, q=3)
    pca_3d = U[:, :3] # Shape: (Hp * Wp, 3)
    
    # Normalize the 3 values to [0, 1] so they can be viewed as RGB colors
    pca_3d = (pca_3d - pca_3d.min(dim=0, keepdim=True).values) / \
             (pca_3d.max(dim=0, keepdim=True).values - pca_3d.min(dim=0, keepdim=True).values)"""
    
    U, S, V = torch.pca_lowrank(features_flat, q=6)
    
    # THE TRICK: Throw away Component 0 and 1 (The X/Y Spatial Positional Encodings)
    # Grab components 2, 3, and 4 for the RGB channels!
    pca_3d = U[:, 2:5] # Shape: (Hp * Wp, 3)
    
    # Normalize the 3 values to [0, 1] so they can be viewed as RGB colors
    pca_3d = (pca_3d - pca_3d.min(dim=0, keepdim=True).values) / \
             (pca_3d.max(dim=0, keepdim=True).values - pca_3d.min(dim=0, keepdim=True).values)
             
    # Reshape back to the image patch grid
    pca_img = pca_3d.view(Hp, Wp, 3).cpu().numpy()

    # 6. Denormalize the original image for plotting
    orig_img = image_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array(config.MEAN)
    std = np.array(config.STD)
    orig_img = np.clip(orig_img * std + mean, 0, 1)

    # 7. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    ax[0].imshow(orig_img)
    ax[0].set_title("Original Minecraft Frame")
    ax[0].axis("off")
    
    # We use interpolation='nearest' to clearly see the distinct 16x16 blocks of the patch grid
    # You can change it to 'bilinear' if you want it to look smoothed out
    ax[1].imshow(pca_img, interpolation='nearest')
    ax[1].set_title("JEPA Semantic PCA Segmentation (768D -> 3D)")
    ax[1].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize()