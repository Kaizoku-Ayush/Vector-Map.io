"""

Key Functions:
1. load_inference_model() - Load trained model for inference
2. process_large_image() - Process large satellite images with tiling
3. convert_to_polygons() - Convert segmentation masks to vector polygons
4. assess_quality() - Quality assessment for vector outputs
5. TilingProcessor - Handle large-scale image processing
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Tuple, Optional

# Model Architecture Classes
class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNetWithHeight(nn.Module):
    """Enhanced U-Net with height estimation capabilities"""
    def __init__(self, n_channels=3, n_classes=1):
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
        # Segmentation decoder
        seg_x = self.up1(x5, x4)
        seg_x = self.up2(seg_x, x3)
        seg_x = self.up3(seg_x, x2)
        seg_x = self.up4(seg_x, x1)
        segmentation = self.outc(seg_x)
        # Height decoder
        height_x = self.height_up1(x5, x4)
        height_x = self.height_up2(height_x, x3)
        height_x = self.height_up3(height_x, x2)
        height_x = self.height_up4(height_x, x1)
        height_map = self.height_out(height_x)
        return segmentation, height_map


def load_inference_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model for inference"""
    try:
        # Try to load as full model first
        model = torch.load(model_path, map_location=device, weights_only=False)
        if hasattr(model, 'eval'):
            model.eval()
            return model

        # If it's a state dict, create architecture and load weights
        model = UNetWithHeight(n_channels=3, n_classes=1).to(device)
        if isinstance(model, dict):
            if 'model_state_dict' in model:
                model.load_state_dict(model['model_state_dict'])
            else:
                model.load_state_dict(model)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

class TilingProcessor:
    """Handle large-scale image processing with tiling"""

    def __init__(self, patch_size: int = 512, overlap: float = 0.1):
        self.patch_size = patch_size
        self.overlap = overlap
        self.step = int(patch_size * (1 - overlap))

    def create_tiles(self, image_shape: Tuple[int, int]) -> List[Dict]:
        """Create tiling plan for large images"""
        h, w = image_shape[:2]
        tiles = []
        for y in range(0, h - self.patch_size + 1, self.step):
            for x in range(0, w - self.patch_size + 1, self.step):
                tiles.append({
                    'x': x, 'y': y,
                    'x_end': min(x + self.patch_size, w),
                    'y_end': min(y + self.patch_size, h)
                })
        return tiles

    def reconstruct_from_tiles(self, predictions: List[np.ndarray], tiles: List[Dict], 
                             original_shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct full image from tile predictions"""
        h, w = original_shape[:2]
        output = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for pred, tile in zip(predictions, tiles):
            pred_tile = pred[0] if len(pred.shape) == 3 else pred
            x, y = tile['x'], tile['y']
            x_end, y_end = tile['x_end'], tile['y_end']

            tile_h, tile_w = pred_tile.shape
            actual_h = min(tile_h, y_end - y)
            actual_w = min(tile_w, x_end - x)

            output[y:y+actual_h, x:x+actual_w] += pred_tile[:actual_h, :actual_w]
            weight_map[y:y+actual_h, x:x+actual_w] += 1

        output = np.divide(output, weight_map, out=np.zeros_like(output), where=weight_map!=0)
        return output

def process_large_image(model: torch.nn.Module, image_path: str, device: torch.device,
                       patch_size: int = 512, overlap: float = 0.1) -> np.ndarray:
    """Process large satellite image with tiling"""
    from torchvision import transforms

    # Load image
    image = np.array(Image.open(image_path))
    original_shape = image.shape

    # Create tiling processor
    processor = TilingProcessor(patch_size, overlap)
    tiles = processor.create_tiles(original_shape)

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process tiles
    predictions = []
    with torch.no_grad():
        for tile in tqdm(tiles, desc="Processing image tiles"):
            x, y = tile['x'], tile['y']
            patch = image[y:y+patch_size, x:x+patch_size]

            # Handle edge cases
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded_patch

            # Inference
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            outputs = model(patch_tensor)

            # Handle model outputs
            if isinstance(outputs, tuple):
                segmentation, height_map = outputs
                prediction = torch.sigmoid(segmentation).cpu().numpy()[0, 0]
            else:
                prediction = torch.sigmoid(outputs).cpu().numpy()[0, 0]

            predictions.append(prediction)

    # Reconstruct full prediction
    full_prediction = processor.reconstruct_from_tiles(predictions, tiles, original_shape)
    return full_prediction

def convert_to_polygons(prediction: np.ndarray, threshold: float = 0.5, 
                       min_area: int = 50) -> List[np.ndarray]:
    """Convert segmentation mask to polygons"""
    # Convert to binary mask
    binary_mask = (prediction > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and convert to polygons
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            # Simplify polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified) >= 3:  # Valid polygon
                polygons.append(simplified.squeeze())

    return polygons

def assess_quality(prediction: np.ndarray, polygons: List[np.ndarray]) -> Dict:
    """Quality assessment for vector outputs"""
    metrics = {
        'image_shape': prediction.shape,
        'mean_confidence': float(np.mean(prediction)),
        'coverage_percentage': float((prediction > 0.5).sum() / prediction.size * 100),
        'num_polygons': len(polygons),
        'polygon_areas': [float(cv2.contourArea(poly.reshape(-1, 1, 2))) for poly in polygons] if polygons else []
    }

    if metrics['polygon_areas']:
        metrics['avg_polygon_area'] = float(np.mean(metrics['polygon_areas']))
        metrics['total_area'] = float(np.sum(metrics['polygon_areas']))
    else:
        metrics['avg_polygon_area'] = 0.0
        metrics['total_area'] = 0.0

    return metrics

def save_results(prediction: np.ndarray, polygons: List[np.ndarray], 
                quality_metrics: Dict, output_dir: str) -> None:
    """Save processing results for pipeline"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Save prediction mask
    prediction_uint8 = (prediction * 255).astype(np.uint8)
    cv2.imwrite(str(output_path / "prediction_mask.png"), prediction_uint8)

    # Save quality metrics
    with open(output_path / "quality_metrics.json", 'w') as f:
        json.dump(quality_metrics, f, indent=2)

    # Save polygon count info
    polygon_info = {
        'num_polygons': len(polygons),
        'polygon_points': [poly.tolist() for poly in polygons] if polygons else []
    }

    with open(output_path / "polygon_data.json", 'w') as f:
        json.dump(polygon_info, f, indent=2)

    print(f"Results saved to: {output_path}")

