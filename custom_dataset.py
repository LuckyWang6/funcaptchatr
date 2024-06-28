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
        right_image = img.crop((0, 200, 200, 400))  # Assume the right image is always at this position

        # Get marked index from filename
        marked_index = self.get_marked_index(img_path)
        total_left_images = 5  # Assume there are 5 left images per file
        non_marked_indexes = [i for i in range(total_left_images) if i != marked_index]
        non_marked_index = random.choice(non_marked_indexes)  # Choose a random non-matching left image

        # Matching left image
        left_image_match = img.crop((200 * marked_index, 0, 200 * (marked_index + 1), 200))
        # Non-matching left image
        left_image_non_match = img.crop((200 * non_marked_index, 0, 200 * (non_marked_index + 1), 200))

        if self.transform:
            left_image_match = self.transform(left_image_match)
            left_image_non_match = self.transform(left_image_non_match)
            right_image = self.transform(right_image)

        # Return one matching and one non-matching pair
        match_pair = (left_image_match, right_image, torch.tensor(1.0))
        non_match_pair = (left_image_non_match, right_image, torch.tensor(0.0))

        return match_pair if random.random() < 0.5 else non_match_pair

    def get_marked_index(self, img_path):
        match = re.search(r"_marked_(\d+)", img_path)
        if match:
            return int(match.group(1))
        return None
