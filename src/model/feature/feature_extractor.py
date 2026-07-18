from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from src.model.feature.glcm import gray_level_co_occurrence_matrix
from src.model.feature.lbp import extract_lbp_feature

class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model_resnet50, self.transform_resnet50 = self.build_resnet50_extractor()
    
    def build_resnet50_extractor(self):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
        model.to(self.device)

        for p in model.parameters():
            p.requires_grad = False

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        return model, transform
    
    def extract_glcm_features(self, image_array):
        glcm_2d = gray_level_co_occurrence_matrix(image_array)
        return glcm_2d.flatten().astype(np.float32)

    def extract_lbp_features(self, image_array):
        feature_vector, _ = extract_lbp_feature(image_array, grid=(7, 7))
        return np.asarray(feature_vector, dtype=np.float32)

    def extract_resnet50_features(self, image_array):
        if isinstance(image_array, np.ndarray):
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
            image_array = Image.fromarray(image_array)

        img_tensor = self.transform_resnet50(image_array)

        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        with torch.no_grad():
            features = self.model_resnet50(img_tensor.unsqueeze(0).to(self.device))

        return features.cpu().numpy().flatten().astype(np.float32)

    def extract_single_image_features(self, img_path):
        texture_img = Image.open(img_path).convert('L')  # mode L: Grayscale
        texture_np_2d = np.asarray(texture_img, dtype=np.float32)
        
        glcm_feat = self.extract_glcm_features(texture_np_2d)
        lbp_feat = self.extract_lbp_features(texture_np_2d)
        resnet_feat = self.extract_resnet50_features(texture_np_2d)

        return np.concatenate([glcm_feat, lbp_feat, resnet_feat]).astype(np.float32)

    def precompute_features_and_targets(self, df):
        print("Precomputing features for all samples...")
        
        all_features = []
        all_targets = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Precompute features", unit="sample"):
            gt = float(row['roughness'])
            texture_feat = self.extract_single_image_features(row['texture_path'])

            all_features.append(texture_feat)
            all_targets.append(gt)

        return (
            np.stack(all_features),
            np.array(all_targets, dtype=np.float32).reshape(len(all_targets), -1),
        )