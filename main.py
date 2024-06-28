import os
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 确保你已经有 models.siamese 和 custom_dataset 的相应代码
from models.siamese import Siamese
from custom_dataset import CustomSiameseDataset

# 配置参数
batch_size = 32
image_folder = './jpg'
validation_split = 0.2  # 验证集比例
num_epochs = 300
learning_rate = 1e-3
momentum = 0.9
early_stop = 30

# 加载数据集
dataset = CustomSiameseDataset(image_folder=image_folder)
train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = Siamese()
model = model.to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 训练参数
best_accuracy = 0.0
epochs_no_improve = 0

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # 训练过程
    for batch_idx, (data1, data2, labels) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")):
        data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data1, data2)
        loss = criterion(outputs.squeeze(), labels.float())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.float()).sum().item()

    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / (batch_idx + 1)}, Accuracy: {epoch_accuracy}%")

    # 验证过程
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for val_data1, val_data2, val_labels in val_loader:
            val_data1, val_data2, val_labels = val_data1.to(device), val_data2.to(device), val_labels.to(device)

            val_outputs = model(val_data1, val_data2)
            val_loss += criterion(val_outputs.squeeze(), val_labels.float()).item()

            val_predicted = (val_outputs.squeeze() > 0.5).float()
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels.float()).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy}%")

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    torch.save(model.state_dict(), "checkpoints/last.pt")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), "checkpoints/best.pt")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == early_stop:
        print(f"No improvement in {early_stop} epochs, stopping training.")
        break
