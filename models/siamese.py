import torch
from torch import nn
from torchvision import models

class Siamese(nn.Module):
    def __init__(self, input_shape=(3, 52, 52)):
        super(Siamese, self).__init__()

        vgg_original = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg_original.features.children())[:-1])  # 去掉最后一个池化层

        self.flatten = nn.Flatten()

        # 计算VGG16在输入52x52时的输出特征图尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            feature_dim = dummy_output.numel()

        self.fc1 = nn.Linear(feature_dim, 512)  # 根据实际计算的尺寸调整
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层

        self.fc2 = nn.Linear(512, 256)  # 增加一个全连接层
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

    def forward(self, left, right):
        left_output = self.forward_one(left)
        right_output = self.forward_one(right)

        # L1距离
        l1_distance = torch.abs(left_output - right_output)

        output = self.fc1(l1_distance)
        output = self.relu(output)
        output = self.dropout(output)  # 添加Dropout层

        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)  # 再次添加Dropout层

        output = self.fc3(output)
        output = self.sigmoid(output)

        return output
