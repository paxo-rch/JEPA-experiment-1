import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
from config import Config as config

vit_transform = transforms.Compose([
    transforms.Resize((config.HEIGHT, config.WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

class FrameDataset(Dataset):
    def __init__(self, root_dir, cache_file="cache.csv"):
        self.root = Path(root_dir)
        if(Path(cache_file).exists()):
            self.images_paths = pd.read_csv(cache_file)["path"].tolist()
        else:
            self.images_paths = [str(p) for p in self.root.rglob("*.jpg")]
            pd.DataFrame({"path": self.images_paths}).to_csv(cache_file, index=False)

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image = Image.open(image_path).convert('RGB')
        return vit_transform(image)

class BatchedDiffusionDataset(Dataset):
    def __init__(self, data_dir: str, batch_size: int = 256, timesteps: int = 50):
        self.data_dir = Path(data_dir)
        self.files = sorted([str(p) for p in self.data_dir.glob("*.pt")])
        self.batch_size = batch_size
        self.num_steps = timesteps
        
        # Pre-compute schedule
        steps = timesteps
        s = 0.008
        t = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((t / steps) + s) / (1 + s) * torch.pi / 2)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        beta = torch.clip(betas, 0, 0.999)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

    def __len__(self):
        return len(self.files) * self.batch_size
    
    def __getitem__(self, index: int):
        file_idx = index // self.batch_size
        local_idx = index % self.batch_size
        file_path = self.files[file_idx]
        
        data = torch.load(file_path, map_location='cpu', weights_only=True, mmap=True)
        x0 = data[local_idx % len(data)].to(torch.float32)
        
        t = torch.randint(0, self.num_steps, (1,)).item()
        noise = torch.randn_like(x0)
        
        xt = self.sqrt_alpha_cumprod[t] * x0 + self.sqrt_one_minus_alpha_cumprod[t] * noise
        return xt, noise, t
