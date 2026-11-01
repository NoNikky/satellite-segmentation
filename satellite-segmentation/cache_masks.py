"""Конвертация RGB-масок первого датасета в файлы с индексами классов (.npy).
Запускать один раз перед train_model1.py. Сильно ускоряет обучение."""
import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

MASK_DIR  = 'data/train/masks'
CACHE_DIR = 'masks_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

PALETTE = np.array([
    [0,   255, 255],  # 0 - город
    [255, 255,   0],  # 1 - сельхоз
    [255,   0, 255],  # 2 - пастбища
    [0,   255,   0],  # 3 - лес
    [0,     0, 255],  # 4 - вода
    [255, 255, 255],  # 5 - бесплодные
    [0,     0,   0],  # 6 - неизвестное
], dtype=np.float32)


def rgb_mask_to_index_fast(mask_rgb):
    h, w = mask_rgb.shape[:2]
    pixels = mask_rgb.reshape(-1, 3).astype(np.float32)
    dists = cdist(pixels, PALETTE, metric='sqeuclidean')
    return np.argmin(dists, axis=1).reshape(h, w).astype(np.uint8)


masks = sorted(f for f in os.listdir(MASK_DIR) if f.endswith('.png'))
already = set(os.listdir(CACHE_DIR))
todo = [f for f in masks if f.replace('.png', '.npy') not in already]
print(f"Осталось конвертировать: {len(todo)}")

for i, fname in enumerate(todo):
    mask_rgb = np.array(Image.open(os.path.join(MASK_DIR, fname)).convert('RGB'))
    np.save(os.path.join(CACHE_DIR, fname.replace('.png', '.npy')),
            rgb_mask_to_index_fast(mask_rgb))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(todo)}")

print("Готово!")
