
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path



class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling block with maxpool followed by double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling block with transpose conv and skip connections"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNetWithHeight(nn.Module):
    """Enhanced U-Net with height estimation capabilities"""
    def __init__(self, n_channels, n_classes):
        super(UNetWithHeight, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        # Decoder for segmentation
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        # Height estimation branch
        self.height_up1 = Up(1024, 512)
        self.height_up2 = Up(512, 256)
        self.height_up3 = Up(256, 128)
        self.height_up4 = Up(128, 64)
        self.height_out = OutConv(64, 1)
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Segmentation decoder path
        seg_x = self.up1(x5, x4)
        seg_x = self.up2(seg_x, x3)
        seg_x = self.up3(seg_x, x2)
        seg_x = self.up4(seg_x, x1)
        segmentation = self.outc(seg_x)
        # Height estimation decoder path
        height_x = self.height_up1(x5, x4)
        height_x = self.height_up2(height_x, x3)
        height_x = self.height_up3(height_x, x2)
        height_x = self.height_up4(height_x, x1)
        height_map = self.height_out(height_x)
        return segmentation, height_map

def load_model_for_worker(model_path, device):
    """Load model in Dask worker - clean and efficient"""
    try:
        # Create model architecture
        model = UNetWithHeight(n_channels=3, n_classes=1).to(device)

        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model in worker: {e}")
        return None

def process_tile_with_model(model_path, image_tile, tile_info, device_str='cuda'):
    """Process a single tile - clean worker function"""
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model_for_worker(model_path, device)
    if model is None:
        return {
            'prediction': np.zeros((512, 512), dtype=np.float32),
            'tile_info': tile_info,
            'status': 'model_load_failed'
        }

    try:
        with torch.no_grad():
            # Prepare input
            if isinstance(image_tile, np.ndarray):
                tile_tensor = torch.FloatTensor(image_tile).permute(2, 0, 1).unsqueeze(0)
                tile_tensor = tile_tensor / 255.0
            else:
                tile_tensor = image_tile.unsqueeze(0)

            tile_tensor = tile_tensor.to(device)

            # Run inference
            outputs = model(tile_tensor)

            # Handle dual output from UNetWithHeight
            if isinstance(outputs, tuple):
                segmentation, height_map = outputs
                prediction = torch.sigmoid(segmentation).cpu().numpy()[0, 0]
            else:
                prediction = torch.sigmoid(outputs).cpu().numpy()[0, 0]

            return {
                'prediction': prediction,
                'tile_info': tile_info,
                'status': 'success'
            }

    except Exception as e:
        print(f"Error during inference in worker: {e}")
        return {
            'prediction': np.zeros((512, 512), dtype=np.float32),
            'tile_info': tile_info,
            'status': 'inference_failed'
        }
