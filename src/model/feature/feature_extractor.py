"""
Image features for the feature-based baselines (LR, SVR, ANN, Hassan 1D-CNN, Scirep 1D-CNN):
GLCM (64) + LBP (49 cells x 59 = 2891) + ResNet-50 (1000) = 3955 values per 256 x 256 patch.

mode="legacy"   : the pipeline used for the results so far
                  (order GLCM|LBP|ResNet, PIL grey, ResNet without normalisation, raw logits).

mode="original" : re-implementation of Hassan et al. (Sci. Rep. 2023). The paper is followed first;
                  details it leaves open follow the authors' code (WASEEM Code/Send).
  grey        : all images converted to greyscale (paper); MATLAB rgb2gray weights (authors' code).
  LBP         : uniform 59-bin LBP on a circular neighbourhood, 7 x 7 cells, histogram / sum.
  GLCM        : 8 levels, horizontal 2-pixel offset, non-symmetric, sum-normalised, column-major.
                Per-feature min-max scaling [0, 1] is fitted on the training fold (test/cv_experiment.py).
  ResNet-50   : greyscale (paper) -> 224 x 224 (cv2 INTER_AREA, authors' code) -> 3 channels ->
                ImageNet-pretrained ResNet-50 -> softmax over the 1000 classes (authors' code: Keras
                predict). torchvision weights + torchvision normalisation are used in place of the
                Keras/Caffe weights + caffe preprocessing (decided 2026-10-13).
  order       : LBP | GLCM | ResNet (authors' FeatureConcate.m).
  Patch-based : problem setting of this study = patch-level prediction (the original image-level setting
                could not be reproduced). Every baseline therefore starts from the same 256 x 256 patch;
                all other steps follow the original method. 7 x 7 cells (36 px) keep the 2891-d LBP vector.
  Framework   : PyTorch / torchvision only (no TensorFlow dependency).
"""
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from src.model.feature.glcm import gray_level_co_occurrence_matrix, flatten_glcm
from src.model.feature.lbp import extract_lbp_feature

FEATURE_MODES = ("legacy", "original")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_LBP, N_GLCM, N_RESNET = 2891, 64, 1000
# position of the GLCM block in the concatenated vector (needed for fold-wise min-max scaling)
GLCM_SLICE = {"legacy": slice(0, N_GLCM), "original": slice(N_LBP, N_LBP + N_GLCM)}


def matlab_rgb2gray(rgb):
    """MATLAB rgb2gray for uint8 RGB input (rounded to uint8)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    g = 0.298936021293775 * rgb[..., 0] + 0.587043074451121 * rgb[..., 1] + 0.114020904255103 * rgb[..., 2]
    return np.clip(np.round(g), 0, 255).astype(np.uint8)


class FeatureExtractor:
    def __init__(self, device, mode="legacy"):
        if mode not in FEATURE_MODES:
            raise ValueError(f"feature mode must be one of {FEATURE_MODES}, got {mode}")
        self.device = device
        self.mode = mode
        self.model_resnet50, self.transform_resnet50 = self.build_resnet50_extractor()

    def build_resnet50_extractor(self):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
        model.to(self.device)

        for p in model.parameters():
            p.requires_grad = False

        if self.mode == "original":
            transform = transforms.Compose([transforms.ToTensor(),                      # 224x224 grey -> (1,224,224) in [0,1]
                                            _GrayToRGB(),
                                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        else:
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        return model, transform

    # ------------------------------------------------------------------ input
    def load_image(self, path):
        """Array the extractor expects: PIL grey (legacy) or MATLAB-grey uint8 from RGB (original)."""
        if self.mode == "original":
            return matlab_rgb2gray(np.asarray(Image.open(path).convert("RGB")))
        return np.asarray(Image.open(path).convert("L"), dtype=np.float32)

    def _gray(self, image_array):
        a = np.asarray(image_array)
        if self.mode == "original" and a.ndim == 3:
            return matlab_rgb2gray(a)
        return a

    # ------------------------------------------------------------------ features
    def extract_glcm_features(self, image_array):
        g = gray_level_co_occurrence_matrix(self._gray(image_array), mode=self.mode)
        return flatten_glcm(g, self.mode).astype(np.float32)

    def extract_lbp_features(self, image_array):
        feature_vector, _ = extract_lbp_feature(self._gray(image_array), grid=(7, 7), mode=self.mode)
        return np.asarray(feature_vector, dtype=np.float32)

    def extract_resnet50_features(self, image_array):
        if self.mode == "original":
            g = self._gray(image_array)
            if g.dtype != np.uint8:
                g = np.clip(np.round(g), 0, 255).astype(np.uint8)
            g = cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA)
            img_tensor = self.transform_resnet50(g)
        else:
            if isinstance(image_array, np.ndarray):
                if image_array.dtype != np.uint8:
                    image_array = image_array.astype(np.uint8)
                image_array = Image.fromarray(image_array)
            img_tensor = self.transform_resnet50(image_array)
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)

        with torch.no_grad():
            out = self.model_resnet50(img_tensor.unsqueeze(0).to(self.device))   # fc1000 logits (1 x 1000)
            if self.mode == "original":
                out = torch.softmax(out, dim=1)                                    # Keras predict() output

        return out.cpu().numpy().flatten().astype(np.float32)

    def concat(self, glcm_feat, lbp_feat, resnet_feat):
        parts = [lbp_feat, glcm_feat, resnet_feat] if self.mode == "original" else [glcm_feat, lbp_feat, resnet_feat]
        return np.concatenate(parts).astype(np.float32)

    def extract_array_features(self, image_array):
        return self.concat(self.extract_glcm_features(image_array),
                           self.extract_lbp_features(image_array),
                           self.extract_resnet50_features(image_array))

    def extract_single_image_features(self, img_path):
        return self.extract_array_features(self.load_image(img_path))

    def precompute_features_and_targets(self, df):
        print(f"Precomputing features for all samples (feature mode: {self.mode})...")

        all_features = []
        all_targets = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Precompute features", unit="sample"):
            gt = float(row['roughness'])
            all_features.append(self.extract_single_image_features(row['texture_path']))
            all_targets.append(gt)

        return (
            np.stack(all_features),
            np.array(all_targets, dtype=np.float32).reshape(len(all_targets), -1),
        )


class _GrayToRGB:
    """(1,H,W) -> (3,H,W) (greyscale replicated, then per-channel ImageNet statistics)."""
    def __call__(self, t):
        return t.repeat(3, 1, 1) if t.shape[0] == 1 else t
