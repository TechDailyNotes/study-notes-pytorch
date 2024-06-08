import albumentations as A
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from utils import plot_examples


image = Image.open(os.path.join('data/images/elon.jpeg')).convert('RGB')
mask1 = Image.open(os.path.join('data/images/mask.jpeg')).convert('RGB')
mask2 = Image.open(os.path.join('data/images/second_mask.jpeg')).convert('RGB')
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
    ], p=1.0)
], is_check_shapes=False)

images = [image]
image = np.array(image)
mask1 = np.array(mask1)
mask2 = np.array(mask2)
for _ in tqdm(range(3)):
    augmented_image_mask = transform(image=image, masks=[mask1, mask2])
    images.append(augmented_image_mask['image'])
    images.append(augmented_image_mask['masks'][0])
    images.append(augmented_image_mask['masks'][1])
plot_examples(images)
