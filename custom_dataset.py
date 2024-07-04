import os
import re
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomSiameseDataset(Dataset):
    def __init__(self, image_folder, transform = None):
        self.image_folder = image_folder
        self.images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((52, 52)),
            transforms.Grayscale(num_output_channels = 3),  # 增加灰度处理，并保持3个通道
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        left_image = img.crop((0, 200, 200, 400))  # Assume the left image is always at this position

        # Get marked index from filename
        marked_index = self.get_marked_index(img_path)
        width, height = img.size
        total_right_images = width // 200  # Assume there are right images per file
        non_marked_indexes = [i for i in range(total_right_images) if i != marked_index]
        non_marked_index = random.choice(non_marked_indexes)  # Choose a random non-matching right image

        # Matching right image
        right_image_match = img.crop((200 * marked_index, 0, 200 * (marked_index + 1), 200))
        # Non-matching right image
        right_image_non_match = img.crop((200 * non_marked_index, 0, 200 * (non_marked_index + 1), 200))

        if self.transform:
            right_image_match = self.transform(right_image_match)
            right_image_non_match = self.transform(right_image_non_match)
            left_image = self.transform(left_image)

        # Return one matching and one non-matching pair
        match_pair = (right_image_match, left_image, torch.tensor(1.0))
        non_match_pair = (right_image_non_match, left_image, torch.tensor(0.0))

        return match_pair if random.random() < 0.5 else non_match_pair

    def get_marked_index(self, img_path):
        match = re.search(r"_marked_(\d+)", img_path)
        if match:
            return int(match.group(1))
        return None
