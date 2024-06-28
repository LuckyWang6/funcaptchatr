import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from models.siamese import Siamese
from custom_dataset import CustomSiameseDataset
from torch.utils.data import DataLoader, random_split
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from torch.optim import Adam, lr_scheduler

# Set environment variable to avoid OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Initialize model
model = Siamese()

# Configuration parameters
batch_size = 32
image_folder = './jpg'
num_epochs = 300
learning_rate = 1e-3
early_stop = 60

transform = transforms.Compose([
    transforms.Resize((52, 52)),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.01),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = CustomSiameseDataset(image_folder=image_folder, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# Check for data leakage
def check_data_leakage(train_data, val_data):
    train_set = set([data[0] for data in train_data])
    val_set = set([data[0] for data in val_data])
    intersection = train_set.intersection(val_set)
    if len(intersection) > 0:
        logger.warning(f"Data leakage detected: {len(intersection)} overlapping samples.")
    else:
        logger.info("No data leakage detected.")


check_data_leakage(train_dataset, val_dataset)


# Check label distribution
def check_label_distribution(dataset):
    labels = [label.item() for _, _, label in dataset]
    label_count = Counter(labels)
    logger.info(f"Label distribution: {label_count}")
    return label_count


train_label_count = check_label_distribution(train_dataset)
val_label_count = check_label_distribution(val_dataset)


# Plot label distribution
def plot_label_distribution(dataset, title):
    labels = [label.item() for _, _, label in dataset]
    df = pd.DataFrame(labels, columns=["label"])
    plt.figure(figsize=(8, 6))
    sns.countplot(x="label", data=df)
    plt.title(title)
    plt.show()


plot_label_distribution(train_dataset, "Training Set Label Distribution")
plot_label_distribution(val_dataset, "Validation Set Label Distribution")


# Calculate class weights
def calculate_class_weights(label_count):
    total = sum(label_count.values())
    weights = {k: total / v for k, v in label_count.items()}
    return weights


class_weights = calculate_class_weights(train_label_count)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Siamese().to(device)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for data1, data2, labels in tqdm(loader, desc="Training"):
        data1, data2, labels = data1.to(device), data2.to(device), labels.to(device).unsqueeze(1)  # Ensure label dimension
        optimizer.zero_grad()
        outputs = model(data1, data2)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data1, data2, labels in tqdm(loader, desc="Validation"):
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device).unsqueeze(1)  # Ensure label dimension
            outputs = model(data1, data2)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
    return total_loss / len(loader)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
    logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model in ONNX format
dummy_input1 = torch.randn(1, 3, 52, 52).to(device)
dummy_input2 = torch.randn(1, 3, 52, 52).to(device)
torch.onnx.export(model, (dummy_input1, dummy_input2), "trained_model.onnx", verbose=True)
