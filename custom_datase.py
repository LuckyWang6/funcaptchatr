import os
import re
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomSiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((52, 52)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        total_right_images = width // 200  # 动态计算右图的数量

        # 获取标记索引
        marked_index = self.get_marked_index(img_path)
        if marked_index is None or marked_index >= total_right_images:
            raise ValueError(f"Invalid marked index in file {img_path}")

        # 固定的左图位置
        left_image = img.crop((0, 200, 200, 400))
        if self.transform:
            left_image = self.transform(left_image)

        # 生成右图
        right_images = []
        labels = []
        for i in range(total_right_images):
            right_image = img.crop((200 * i, 0, 200 * (i + 1), 200))
            if self.transform:
                right_image = self.transform(right_image)
            right_images.append(right_image)
            labels.append(torch.tensor(1.0) if i == marked_index else torch.tensor(0.0))

        return left_image, right_images, labels

    def get_marked_index(self, img_path):
        match = re.search(r"_marked_(\d+)", img_path)
        if match:
            return int(match.group(1))
        return None