from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from src.model.feature.glcm import gray_level_co_occurrence_matrix
from src.model.feature.lbp import extract_lbp_feature
from src.model.feature.geometry_statistic import load_height_map, load_normal_map, extract_height_features, extract_normal_features, HEIGHT_KEYS, NORMAL_KEYS, dict_to_ordered_vector
class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model_resnet50, self.transform_resnet50 = self.build_resnet50_extractor()
        self.transform_spatial = transforms.Compose([
            transforms.Resize(
                (1568, 1568), 
                interpolation=InterpolationMode.BICUBIC,
                antialias=True),
            transforms.ToTensor(),
        ])
    
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

    def extract_resnet50_features(self, img_tensor):
        with torch.no_grad():
            features = self.model_resnet50(img_tensor.unsqueeze(0).to(self.device))
        return features.cpu().numpy().flatten().astype(np.float32)

    def extract_single_image_features(self, img_path):
        texture_img = Image.open(img_path).convert('L') # mode L: Grayscale
        texture_img_resized = self.transform_spatial(texture_img)
        texture_np_2d = texture_img_resized.squeeze(0).numpy()
        
        glcm_feat = self.extract_glcm_features(texture_np_2d)
        lbp_feat = self.extract_lbp_features(texture_np_2d)

        img_tensor = self.transform_resnet50(texture_img)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1) 
        resnet_feat = self.extract_resnet50_features(img_tensor)

        return np.concatenate([glcm_feat, lbp_feat, resnet_feat]).astype(np.float32)
        # return np.concatenate([glcm_feat, lbp_feat]).astype(np.float32)
    
    def extract_texture_descriptor(self, img_path):
        texture_img = Image.open(img_path).convert('L') # mode L: Grayscale
        texture_img_resized = self.transform_spatial(texture_img)
        texture_np_2d = texture_img_resized.squeeze(0).numpy()
        
        glcm_feat = self.extract_glcm_features(texture_np_2d)
        lbp_feat = self.extract_lbp_features(texture_np_2d)

        return np.concatenate([glcm_feat, lbp_feat]).astype(np.float32)
    
    def extract_geometric_statistical_features(self, height_path, normal_path):
        height_map = load_height_map(height_path, normalize=True)
        normal_map = load_normal_map(normal_path, normalize=True)

        height_features = extract_height_features(height_map)
        normal_features = extract_normal_features(normal_map)

        height_vec = dict_to_ordered_vector(height_features, HEIGHT_KEYS)
        normal_vec = dict_to_ordered_vector(normal_features, NORMAL_KEYS)

        return height_vec, normal_vec

    def precompute_features_and_targets(self, df, conf, target_col):
        print("Precomputing features for all samples...")
        
        all_features = []
        all_targets = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Precompute features", unit="sample"):
                   
            if target_col == 'roughness':
                gt = float(row['roughness'])
            elif target_col == 'haptic_attribute':
                gt = row['haptic_attribute']
            
            texture_feat = self.extract_single_image_features(row['texture_path'])
            if conf['dataset_input'] == 'texture_maps':
                normal_feat = self.extract_single_image_features(row['normal_path'])
                height_feat = self.extract_single_image_features(row['height_path'])
                combined_feat = np.concatenate([texture_feat, normal_feat, height_feat]).astype(np.float32)
            else: 
                combined_feat = texture_feat

            all_features.append(combined_feat)
            all_targets.append(gt)

        return (
            np.stack(all_features),
            np.array(all_targets, dtype=np.float32).reshape(len(all_targets), -1),
        )
    
    def precompute_features_and_targets_separated(self, df, conf, target_col):
        print("Precomputing features for all samples...")
        
        texture_feats = []
        height_feats = []
        normal_feats = []
        all_targets = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Precompute features", unit="sample"):
            gt = row['roughness']
        
            texture_feat = self.extract_single_image_features(row['texture_path'])
            height_feat, normal_feat = self.extract_geometric_statistical_features(row['height_path'], row['normal_path'])

            texture_feats.append(texture_feat)
            height_feats.append(height_feat)
            normal_feats.append(normal_feat)
            all_targets.append(gt)
        
        return texture_feats, height_feats, normal_feats, all_targets
             
