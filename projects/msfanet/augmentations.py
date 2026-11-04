import random

import numpy as np
from PIL import Image
from torchvision import transforms


class RandomPatch:
    def __init__(self, probability=0.5, min_fraction=0.08, max_fraction=0.25):
        self.probability = probability
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        image = image.copy()
        width, height = image.size
        patch_w = max(1, int(width * random.uniform(self.min_fraction, self.max_fraction)))
        patch_h = max(1, int(height * random.uniform(self.min_fraction, self.max_fraction)))

        source_x = random.randint(0, max(0, width - patch_w))
        source_y = random.randint(0, max(0, height - patch_h))
        target_x = random.randint(0, max(0, width - patch_w))
        target_y = random.randint(0, max(0, height - patch_h))

        patch = image.crop((
            source_x, source_y, source_x + patch_w, source_y + patch_h
        ))
        image.paste(patch, (target_x, target_y))
        return image


def build_train_transform(height=256, width=128):
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((height, width), padding=8, padding_mode="edge"),
        RandomPatch(probability=0.5),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        transforms.RandomErasing(
            p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random"
        )
    ])
