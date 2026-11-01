"""Обучение модели 1 (датасет 803 снимка, RGB-маски).
Сначала запустите cache_masks.py."""
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MulticlassJaccardIndex

# ─── Настройки ────────────────────────────────────────────────────────────────
IMG_DIR     = 'data/train/images'   # файлы вида 100694_sat.jpg
CACHE_DIR   = 'masks_cache'         # файлы вида 100694_mask.npy
CKPT_PATH   = 'best_model.pth'
NUM_CLASSES = 7
EPOCHS      = 50
BATCH_SIZE  = 16                    # на слабой видеокарте/CPU поставьте 4
NUM_WORKERS = 0 if os.name == 'nt' else 2
CLASS_NAMES = ['город', 'сельхоз', 'пастбища', 'лес', 'вода', 'бесплодные']

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


class LandDataset(Dataset):
    def __init__(self, ids, transform=None):
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        image = np.array(Image.open(
            os.path.join(IMG_DIR, name + '_sat.jpg')).convert('RGB'))
        mask = np.load(os.path.join(CACHE_DIR, name + '_mask.npy'))
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out['image'], out['mask']
        return image, mask.long()


train_transform = A.Compose([
    A.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


def main():
    all_ids = sorted(f.replace('_sat.jpg', '')
                     for f in os.listdir(IMG_DIR) if f.endswith('_sat.jpg'))
    train_ids, temp_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

    def loader(ids, tf, shuffle):
        return DataLoader(LandDataset(ids, tf), batch_size=BATCH_SIZE,
                          shuffle=shuffle, num_workers=NUM_WORKERS)

    train_loader = loader(train_ids, train_transform, True)
    val_loader   = loader(val_ids,   val_transform,   False)
    test_loader  = loader(test_ids,  val_transform,   False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    model = smp.Unet(encoder_name='efficientnet-b3', encoder_weights='imagenet',
                     in_channels=3, classes=NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    ce_loss   = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)

    def criterion(pred, target):
        return dice_loss(pred, target) + ce_loss(pred, target)

    miou_metric = MulticlassJaccardIndex(
        num_classes=NUM_CLASSES, average='none', ignore_index=6).to(device)

    def run_epoch(dl, train=True):
        model.train() if train else model.eval()
        total = 0.0
        miou_metric.reset()
        with (torch.enable_grad() if train else torch.no_grad()):
            for images, masks in dl:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total += loss.item()
                miou_metric.update(preds.argmax(dim=1), masks)
        iou = miou_metric.compute().cpu().numpy()
        return total / len(dl), iou.mean(), iou

    best = 0.0
    for epoch in range(1, EPOCHS + 1):
        tl, tm, _ = run_epoch(train_loader, True)
        vl, vm, _ = run_epoch(val_loader, False)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {tl:.4f} miou {tm:.4f} | "
              f"val loss {vl:.4f} miou {vm:.4f}")
        if vm > best:
            best = vm
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✓ сохранена лучшая модель (miou={best:.4f})")

    print("\nФинальный тест...")
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=False))
    tl, tm, iou = run_epoch(test_loader, False)
    print(f"=== TEST ===  loss {tl:.4f}  mIoU {tm:.4f}")
    for n, s in zip(CLASS_NAMES, iou):
        print(f"  {n:<12} IoU = {s:.4f}")


if __name__ == '__main__':
    main()
