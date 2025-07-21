import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from model import MultiBackBoneRegressor
import data

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Warning: GPU is not available")

scaler = GradScaler('cuda')

# 하이퍼파라미터 설정
batch_size = 64
num_epochs = 100
feature_extractor = 'resnet18'

model = MultiBackBoneRegressor(feature_extractor).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

df_train, df_valid, df_test = data.build_dataframe()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = data.CustomRegressionDataset(df_train, transform)
valid_dataset = data.CustomRegressionDataset(df_valid, transform)
test_dataset = data.CustomRegressionDataset(df_test, transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

def train():
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for (texture_img, normal_map, height_map), labels in train_loader:
            inputs = texture_img.to(device), normal_map.to(device), height_map.to(device)
            labels = labels.to(device)

            with autocast(device_type='cuda', dtype=torch.float16):
                predict = model(*inputs)
                loss = criterion(predict, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach().item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (texture_img, normal_map, height_map), targets in valid_loader:
                inputs = texture_img.to(device), normal_map.to(device), height_map.to(device)
                targets = targets.to(device)

                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def eval_test_set(model, test_loader, criterion, device):
    # 모델 로드 및 평가 모드 전환
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    test_loss = 0.0
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for (texture, normal, height), targets in test_loader:
            # 입력 및 타겟 GPU 이동
            texture = texture.to(device)
            normal = normal.to(device)
            height = height.to(device)
            targets = targets.to(device)

            outputs = model(texture, normal, height)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # 값 저장 (배치 전체 저장)
            predictions.append(outputs.cpu().numpy())
            ground_truths.append(targets.cpu().numpy())

    # (배치, 4) 형태를 (N, 4)로 합침
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    # 전체 4개 특성별 MAE, R2 구하기
    maes = mean_absolute_error(ground_truths, predictions, multioutput='raw_values')
    r2s = r2_score(ground_truths, predictions, multioutput='raw_values')
    avg_mae = maes.mean()
    avg_r2 = r2s.mean()
    
    print("\nTest Results (각 특성별):")
    headers = ['Roughness', 'Stickiness', 'Bumpiness', 'Hardness']
    for i, h in enumerate(headers):
        print(f"{h:>10}: MAE={maes[i]:.4f} | R2={r2s[i]:.4f}")
    print(f"\n평균 MAE: {avg_mae:.4f} | 평균 R2: {avg_r2:.4f} | 평균 Loss: {test_loss / len(test_loader):.4f}")

    # 특성별 산점도 시각화
    plt.figure(figsize=(10, 10))
    for i, h in enumerate(headers):
        plt.subplot(2, 2, i+1)
        plt.scatter(ground_truths[:, i], predictions[:, i], alpha=0.5)
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title(h)
        plt.plot([ground_truths[:, i].min(), ground_truths[:, i].max()], 
                 [ground_truths[:, i].min(), ground_truths[:, i].max()],
                 'r--', lw=1)
    plt.tight_layout()
    plt.savefig('results_multi.png')
    plt.close()

def main():
    # train()
    eval_test_set(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()