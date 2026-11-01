# Семантическая сегментация спутниковых снимков

Курсовая работа: определение типов землепользования на спутниковых снимках
с помощью U-Net и предобученного энкодера EfficientNet-B3.

## Результаты

| | Модель 1 | Модель 2 |
|---|---|---|
| Датасет | [название, ссылка] | [название, ссылка] |
| Снимков | 803 | 2522 (rural + urban) |
| Маски | RGB | Grayscale |
| Test mIoU | 0.5459 | 0.6417 |

**Модель 1:** город 0.73, сельхоз 0.84, пастбища 0.36, лес 0.71, вода 0.69, бесплодные 0.49

**Модель 2:** building 0.00, road 0.73, water 0.70, barren 0.80, forest 0.63, agriculture 0.79

Класс building не распознаётся из-за дисбаланса классов; возможное улучшение — веса классов в функции потерь.

## Технологии

PyTorch, segmentation-models-pytorch, Albumentations, torchmetrics.
U-Net, энкодер EfficientNet-B3 (ImageNet), вход 256×256.
Loss: Dice + Soft Cross-Entropy. Оптимизатор: AdamW, 50 эпох.

## Установка

```bash
git clone https://github.com/[ЛОГИН]/satellite-segmentation.git
cd satellite-segmentation
pip install -r requirements.txt
```

## Использование

1. Скачайте веса из раздела Releases и положите рядом со скриптами.
2. Положите снимки в папку `input/`.
3. Запустите `python predict_model2.py` (или `predict_model1.py`).

Результат появится в папке `output/`.

## Обучение

- Модель 1: `python cache_masks.py`, затем `python train_model1.py`
- Модель 2: `python train_model2.py`

Пути к данным указаны в начале скриптов.

## Датасеты

- [Название 1]: [ссылка]
- [Название 2]: [ссылка]

## Автор

[Имя], [вуз], [год]
