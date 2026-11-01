"""Предсказание моделью 2 (rural + urban).
Положите снимки (jpg/png) в папку input/ и запустите скрипт.
Результаты сохраняются в папку output/."""
import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_PATH  = 'best_model2.pth'
INPUT_DIR   = 'input'
OUTPUT_DIR  = 'output'
NUM_CLASSES = 7

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = np.array([
    [0, 0, 0],      # background
    [255, 0, 0],    # building
    [128, 128, 128],# road
    [0, 0, 255],    # water
    [165, 42, 42],  # barren
    [0, 255, 0],    # forest
    [255, 255, 0],  # agriculture
], dtype=np.uint8)

CLASS_NAMES = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture']
SKIP = ['background']   # классы, не учитываемые в статистике

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")

model = smp.Unet(encoder_name='efficientnet-b3', encoder_weights=None,
                 in_channels=3, classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
print("Модель загружена ✓")

transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def predict(image_path):
    image = np.array(Image.open(image_path).convert('RGB'))
    h, w = image.shape[:2]
    inp = transform(image=image)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).argmax(dim=1).squeeze().cpu().numpy()
    color = np.array(Image.fromarray(COLORS[pred]).resize((w, h), Image.NEAREST))
    stats = {}
    for i, name in enumerate(CLASS_NAMES):
        if name in SKIP:
            continue
        cnt = np.sum(pred == i)
        if cnt > 0:
            stats[name] = cnt / pred.size * 100
    return image, color, stats


def save_result(image, color, stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image); axes[0].set_title('Оригинальный снимок'); axes[0].axis('off')
    axes[1].imshow(color); axes[1].set_title('Предсказание модели'); axes[1].axis('off')
    patches = [mpatches.Patch(color=COLORS[i] / 255,
                              label=f'{CLASS_NAMES[i]} — {stats.get(CLASS_NAMES[i], 0):.1f}%')
               for i in range(1, 7)]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {path}")


images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not images:
    print("Положите снимки в папку input/ и запустите снова")
else:
    for fname in images:
        print(f"\nОбрабатываю: {fname}")
        image, color, stats = predict(os.path.join(INPUT_DIR, fname))
        for name, pct in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {name:<12} {pct:.1f}%")
        save_result(image, color, stats,
                    os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + '_result.png'))
print("\nГотово!")
