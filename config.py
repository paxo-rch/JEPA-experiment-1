import torch

class Config:
    # Infrastructure
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRECISION = torch.bfloat16 # Native to Blackwell

    # Video Specs
    WIDTH, HEIGHT = 640, 352
    
    PATCH_SIZE = 16
    FPS = 20

    EMBEDDING_SIZE = 768

    # Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]