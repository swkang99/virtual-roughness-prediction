import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from src.model.feature.glcm import gray_level_co_occurrence_matrix
from src.model.feature.lbp import extract_lbp_feature

# 폴더 경로 설정
folders = [
    "data/original/texture_image",
    "data/original/normal_map",
    "data/original/height_map"
]


def extract_resnet_features(image_path):
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Hook for intermediate layer output
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    model.layer4.register_forward_hook(get_features('layer4'))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        _ = model(img_tensor)
    
    return features['layer4'].squeeze().cpu().numpy()

for folder in folders:
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"The folder does not exist: {folder}")
        continue
    
    image_files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg"))
    
    if len(image_files) == 0:
        print(f"No image available: {folder}")
        continue
    
    # Test with the first image
    # sample_image_path = image_files[0]

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        img_color = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"\nProcessing: {folder} - {image_path.name}")
        output_dir = Path("output/feature_visualizations")
        output_dir = output_dir / image_path.name[:-3]
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Visualization GLCM
        glcm_features = gray_level_co_occurrence_matrix(img_color)
        
        plt.figure(figsize=(6, 5))
        plt.imshow(glcm_features, cmap="hot", interpolation="nearest")
        plt.colorbar(label="Co-occurrence frequency")
        plt.title("GLCM 8x8 Matrix (as Image)")
        plt.xlabel("Gray level (i)")
        plt.ylabel("Gray level (j)")
        plt.xticks(range(8))
        plt.yticks(range(8))
        
        glcm_output = output_dir / f"glcm_{folder_path.name}_{image_path.stem}.png"
        plt.savefig(glcm_output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Save GLCM: {glcm_output}")
        
        # 2. Visualization LBP
        lbp_feat, lbp_maps = extract_lbp_feature(img_color)
        
        rows, cols = (7, 7)
        full_lbp = np.zeros((224, 224), dtype=np.uint8)
        cell_h = 224 // rows
        cell_w = 224 // cols
        
        for r in range(rows):
            for c in range(cols):
                lbp_cell = lbp_maps[r][c]
                # Since it is 2 pixels smaller than the original cell, add padding
                h, w = lbp_cell.shape
                y0, y1 = r * cell_h + 1, r * cell_h + 1 + h
                x0, x1 = c * cell_w + 1, c * cell_w + 1 + w
                
                if y1 <= 224 and x1 <= 224:
                    full_lbp[y0:y1, x0:x1] = lbp_cell
        
        # Visualization (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: LBP Grid
        resized_color = cv2.resize(image, (224, 224))
        # Grayscale Conversion
        if resized_color.ndim == 3:
            resized_gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
        else:
            resized_gray = resized_color

        # Convert grayscale to 3 channels (for grid line colors)
        fig_grid = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)

        # Drawing Grid Lines (Keep Green)
        for r in range(rows + 1):
            y = r * cell_h
            cv2.line(fig_grid, (0, y), (224, y), (0, 255, 0), 1)
        for c in range(cols + 1):
            x = c * cell_w
            cv2.line(fig_grid, (x, 0), (x, 224), (0, 255, 0), 1)

        axes[0].imshow(fig_grid) 
        axes[0].set_title(f'LBP Grid ({rows}x{cols} cells)', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Right: LBP Feature Map
        im = axes[1].imshow(full_lbp, cmap='hot')
        axes[1].set_title('LBP Feature Map', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        lbp_output = output_dir / f"lbp_{folder_path.name}_{image_path.stem}.png"
        plt.tight_layout()
        plt.savefig(lbp_output, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Save LBP Visualization: {lbp_output}")
        print(f"Feature vector shape: {lbp_feat.shape}")
        print(f"LBP grid: {rows}x{cols} cells, {59} bins per cell")
        
        # 3. Visualization of ResNet-50 Feature Maps
        resnet_features = extract_resnet_features(image_path)
        
        # Visualize only the first 16 channels
        n_features = min(16, resnet_features.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        for i in range(n_features):
            row, col = i // 4, i % 4
            axes[row, col].imshow(resnet_features[i], cmap='viridis')
            axes[row, col].set_title(f'Ch {i}')
            axes[row, col].axis('off')
        
        plt.suptitle(f'ResNet-50 Feature Maps - {folder_path.name}', fontsize=14)
        plt.tight_layout()
        
        resnet_output = output_dir / f"resnet_{folder_path.name}_{image_path.stem}.png"
        plt.savefig(resnet_output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ResNet 저장: {resnet_output}")

print(f"\nAll feature maps have been saved to {output_dir}.")