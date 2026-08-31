import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class OCTAFAZDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=512):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size

        self.full_dir = os.path.join(data_root, 'Projection', 'FAZ_Full')
        self.ilm_dir = os.path.join(data_root, 'Projection', 'FAZ_ILM_OPL')
        self.opl_dir = os.path.join(data_root, 'Projection', 'FAZ_OPL_BM')
        self.label_dir = os.path.join(data_root, 'Label', 'FAZ')

        self.image_names = self._scan_images()

    def _scan_images(self):
        if os.path.exists(self.full_dir):
            names = sorted([f for f in os.listdir(self.full_dir)
                           if f.endswith(('.png', '.jpg', '.tif', '.bmp'))])
        else:
            names = []
            for root, dirs, files in os.walk(os.path.dirname(self.full_dir)):
                for f in files:
                    if f.endswith(('.png', '.jpg', '.tif', '.bmp')):
                        names.append(f)
            names = sorted(set(names))

        return names

    def __len__(self):
        return len(self.image_names)

    def _load_image(self, path):
        img = Image.open(path).convert('L')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _load_mask(self, path):
        mask = Image.open(path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)
        return mask

    def __getitem__(self, idx):
        name = self.image_names[idx]

        full_path = os.path.join(self.full_dir, name)
        ilm_path = os.path.join(self.ilm_dir, name)
        opl_path = os.path.join(self.opl_dir, name)

        if not os.path.exists(ilm_path):
            ilm_path = self._find_closest(name, self.ilm_dir)
        if not os.path.exists(opl_path):
            opl_path = self._find_closest(name, self.opl_dir)

        label_path = os.path.join(self.label_dir, name)
        if not os.path.exists(label_path):
            label_path = self._find_closest(name, self.label_dir)

        full = self._load_image(full_path)
        ilm = self._load_image(ilm_path)
        opl = self._load_image(opl_path)
        mask = self._load_mask(label_path)

        image = np.stack([full, ilm, opl], axis=0)

        if self.split == 'train':
            image, mask = self._augment(image, mask)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

    def _find_closest(self, name, directory):
        base = os.path.splitext(name)[0]
        for f in os.listdir(directory):
            if os.path.splitext(f)[0] == base:
                return os.path.join(directory, f)
        files = os.listdir(directory)
        if files:
            return os.path.join(directory, files[0])
        return os.path.join(directory, name)

    def _augment(self, image, mask):
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()
        return image, mask
