import torch
from src.model.feature.glcm import gray_level_co_occurrence_matrix
from src.model.feature.lbp import extract_lbp_feature
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize

class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
def extract_glcm_features(image_array):
    glcm_2d = gray_level_co_occurrence_matrix(image_array)
    return glcm_2d.flatten().astype(np.float32)


def extract_lbp_features(image_array):
    feature_vector, lbp_maps = extract_lbp_feature(image_array, grid=(7, 7))
    return np.asarray(feature_vector, dtype=np.float32)


def extract_resnet50_features(image_tensor, model, device):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("Loading ResNet50...")
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50.eval()
    resnet50.to(device)

    for p in resnet50.parameters():
        p.requires_grad = False
    model.eval()
    with torch.no_grad():
        features = model(image_tensor.unsqueeze(0).to(device))
    return features.cpu().numpy().flatten().astype(np.float32)

def extract_single_image_features(img_path, transform, resnet50, device):
    texture_img = Image.open(img_path).convert('L') # mode L: Grayscale
    texture_img_np = np.array(texture_img)
    texture_img_resized = resize(texture_img_np, (1568, 1568), preserve_range=False, anti_aliasing=False)
    
    glcm_feat = extract_glcm_features(texture_img_resized)

    lbp_feat = extract_lbp_features(texture_img_resized)

    img_tensor = transform(texture_img)
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1) # if image is grayscale, repeat to make 3 channel
    resnet_feat = extract_resnet50_features(img_tensor, resnet50, device)

    return np.concatenate([glcm_feat, lbp_feat, resnet_feat]).astype(np.float32)

def build_all_features(full_df, transform, resnet50, device):
    print("Precomputing features for all samples...")
    all_features = []
    all_targets = []
    image_ids = []

    glcm_times = []
    lbp_times = []
    resnet_times = []

    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Precompute features", unit="sample"):
        # texture_img = cv2.imread(row['texture_path'], cv2.IMREAD_GRAYSCALE)
        # texture_img_resized = cv2.resize(texture_img, dsize=(1568, 1568))
        
        # height_img = Image.open(row['height_path']).convert('L')
        # normal_img = Image.open(row['normal_path']).convert('RGB')
        
        # gt = float(row['roughness']) # Use roughness only
        gt = row['haptic_attribute']
 
        texture_feat, texture_glcm_time, texture_lbp_time, texture_resnet_time = extract_single_image_features(row['texture_path'], transform, resnet50, device)
        # normal_feat, normal_glcm_time, normal_lbp_time, normal_resnet_time  = extract_single_image_features(normal_img, transform, resnet50, device)
        # height_feat, height_glcm_time, height_lbp_time, height_resnet_time  = extract_single_image_features(height_img, transform, resnet50, device)

        # combined_feat = np.concatenate([texture_feat, normal_feat, height_feat]).astype(np.float32)
        combined_feat = texture_feat

        all_features.append(combined_feat)
        all_targets.append(gt)
        image_ids.append(str(int(Path(row['texture_path']).stem)))

        glcm_times.append(texture_glcm_time)
        # glcm_times.append(normal_glcm_time)
        # glcm_times.append(height_glcm_time)
        lbp_times.append(texture_lbp_time)
        # lbp_times.append(normal_lbp_time)
        # lbp_times.append(height_lbp_time)
        resnet_times.append(texture_resnet_time)
        # resnet_times.append(normal_resnet_time)
        # resnet_times.append(height_resnet_time)
    
    return (
        np.stack(all_features),
        np.array(all_targets, dtype=np.float32).reshape(len(all_targets), -1),
        image_ids,
    )