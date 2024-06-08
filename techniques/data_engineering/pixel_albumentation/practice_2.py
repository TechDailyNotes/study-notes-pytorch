import albumentations as A
import cv2
import numpy as np
import os
from tqdm import tqdm
from utils import plot_examples


image = cv2.imread(os.path.join('data/images/elon.jpeg'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = A.Compose([
    A.Resize(width=1920, height=1080),
    A.RandomCrop(width=1280, height=720),
    A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.05),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5),
    ], p=1.0),
])

images = [image]
image = np.array(image)
for _ in tqdm(range(3)):
    augmentations = transform(image=image)
    augmented_image = augmentations['image']
    images.append(augmented_image)
plot_examples(images)
