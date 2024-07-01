import os
import random
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from custom_datase import CustomSiameseDataset
from models.siamese import Siamese


# 配置参数
batch_size = 32
image_folder = './jpg'
validation_split = 0.2
num_epochs = 300
learning_rate = 1e-3
momentum = 0.9
early_stop = 30

# 确保数据分割正确，没有数据泄露
dataset = CustomSiameseDataset(image_folder=image_folder)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(validation_split * dataset_size)

# 打乱数据集
random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

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

# 存储损失和准确率
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 创建保存数据的目录
if not os.path.exists('data'):
    os.makedirs('data')

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (left_image, right_images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")):
        left_image = left_image.to(device)

        batch_loss = 0.0
        batch_correct = 0
        batch_total = 0

        for right_image, label in zip(right_images, labels):
            right_image, label = right_image.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(left_image, right_image).view(-1)  # 使用 view(-1) 确保输出形状与目标形状一致
            loss = criterion(outputs, label.float())
            batch_loss += loss.item()

            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()
            batch_total += label.size(0)
            batch_correct += (predicted == label.float()).sum().item()

        total_loss += batch_loss / len(right_images)
        correct += batch_correct
        total += batch_total

    epoch_accuracy = 100 * correct / total
    train_losses.append(total_loss / (batch_idx + 1))
    train_accuracies.append(epoch_accuracy)



    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / (batch_idx + 1)}, Accuracy: {epoch_accuracy}%")

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for val_left_image, val_right_images, val_labels in val_loader:
            val_left_image = val_left_image.to(device)

            val_batch_loss = 0.0
            val_batch_correct = 0
            val_batch_total = 0

            for val_right_image, val_label in zip(val_right_images, val_labels):
                val_right_image, val_label = val_right_image.to(device), val_label.to(device)

                val_outputs = model(val_left_image, val_right_image).view(-1)  # 使用 view(-1) 确保输出形状与目标形状一致
                val_batch_loss += criterion(val_outputs, val_label.float()).item()

                val_predicted = (val_outputs > 0.5).float()
                val_batch_total += val_label.size(0)
                val_batch_correct += (val_predicted == val_label.float()).sum().item()

            val_loss += val_batch_loss / len(val_right_images)
            val_correct += val_batch_correct
            val_total += val_batch_total

    val_accuracy = 100 * val_correct / val_total
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)

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


# 导出为 ONNX 模型
dummy_input1 = torch.randn(1, 3, 52, 52).to(device)  # 修改 batch_size 为 1
dummy_input2 = torch.randn(1, 3, 52, 52).to(device)
torch.onnx.export(model, (dummy_input1, dummy_input2), "checkpoints/3d_rollball_objects_v2.onnx",
                  input_names=['input_left', 'input_right'],
                  output_names=['output'])
print("Model has been successfully exported to ONNX format.")
