import os
import re
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomSiameseDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')

        bottom_img = img.crop((0, 200, 200, 400))  # 固定的底部图片
        bottom_img = self.transform(bottom_img)

        top_imgs = self.get_top_images(img)
        matched_index = self.get_marked_index(img_path)

        if matched_index is not None:
            matched_img = top_imgs[matched_index]
            matched_img = self.transform(matched_img)
            tag = torch.tensor(1.0)  # 相同特征
        else:
            unmatched_index = random.choice([i for i in range(len(top_imgs)) if i != matched_index])
            matched_img = top_imgs[unmatched_index]
            matched_img = self.transform(matched_img)
            tag = torch.tensor(0.0)  # 不同特征

        return bottom_img, matched_img, tag

    def get_top_images(self, img):
        width, height = img.size
        num_cols = width // 200
        top_imgs = [img.crop((200 * i, 0, 200 * (i + 1), 200)) for i in range(num_cols)]
        return top_imgs

    def get_marked_index(self, img_path):
        filename = os.path.basename(img_path)
        match = re.search(r"_marked_(\d+)", filename)
        if match:
            return int(match.group(1))
        return None

    def get_image_paths(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')

        bottom_img_path = img_path

        top_imgs = self.get_top_images(img)
        matched_index = self.get_marked_index(img_path)

        if matched_index is not None:
            matched_img_path = self.images[matched_index]
        else:
            unmatched_index = random.choice([i for i in range(len(top_imgs)) if i != matched_index])
            matched_img_path = self.images[unmatched_index]

        return bottom_img_path, matched_img_path
