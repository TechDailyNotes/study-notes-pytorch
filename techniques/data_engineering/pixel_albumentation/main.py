import albumentations as A
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from utils import plot_examples


image = Image.open(os.path.join('data/images/elon.jpeg')).convert('RGB')
transform = A.Compose([
    A.Resize(width=1920, height=1080),
    A.RandomCrop(width=1280, height=720),
    A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5),
    ], p=1.0),
])

images_list = [image]
image = np.array(image)
for i in tqdm(range(15)):
    augmented_image = transform(image=image)['image']
    images_list.append(augmented_image)
plot_examples(images_list)
