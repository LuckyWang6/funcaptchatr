import os.path

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from models.siamese import Siamese
from custom_dataset import CustomSiameseDataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


batch_size = 10
num_epochs = 100

epochs_no_improve = 0
early_stop = 10
image_folder = './jpg'
# 加载数据集
dataset = CustomSiameseDataset(image_folder = image_folder)
train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(seed):
    best_accuracy = 0.0
    # 设置随机种子
    set_seed(seed)
    model = Siamese()
    if os.path.exists('checkpoints/best_siamese_model.pt'):
        model.load_state_dict(torch.load('checkpoints/best_siamese_model.pt'))
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr = 1e-3, momentum = 0.9)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data1, data2, labels) in enumerate(
                tqdm(train_loader, desc = f"Epoch [{epoch + 1}/{num_epochs}]")):
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

        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")

        torch.save(model.state_dict(), "checkpoints/last.pt")
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), "checkpoints/best_siamese_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stop:
            print(f"No improvement in {early_stop} epochs, stopping training.")
            break
    return model


# 评估函数
def evaluate_model(net, dataloader):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            data1, data2, labels = data
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
            outputs = net(data1, data2)
            criterion = nn.BCELoss()
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()
    return total_loss / len(dataloader)


# 训练和评估多个模型
num_models = 5
seeds = [42, 43, 44, 45, 46]
models = []
model_losses = []
for seed in seeds:
    print(f"Training model with seed {seed}")
    model = train_model(seed)
    models.append(model)
    loss = evaluate_model(model, train_loader)
    model_losses.append(loss)
    print(f"Model with seed {seed} has loss: {loss}")

# 选择最好的模型
best_model_index = np.argmin(model_losses)
best_model = models[best_model_index]
print(f"Best model index: {best_model_index} with loss: {model_losses[best_model_index]}")

# 保存最好的模型
torch.save(best_model.state_dict(), 'checkpoints/best_siamese_model.pth')

# Save the trained model in ONNX format
dummy_input1 = torch.randn(1, 3, 52, 52).to(device)
dummy_input2 = torch.randn(1, 3, 52, 52).to(device)
inputnames = ['input_left', 'input_right']
torch.onnx.export(best_model, (dummy_input1, dummy_input2), "3d_rollball_objects_v2.onnx", verbose = True,
                  input_names = inputnames, output_names = ['output'])
