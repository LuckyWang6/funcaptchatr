import os

import torch
from PIL import Image
from torchvision import transforms

# 确保你已经有 models.siamese 和 custom_dataset 的相应代码
from models.siamese import Siamese

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = Siamese()
model = model.to(device)

lists = []
transform = transforms.Compose([
    transforms.Resize((52, 52)),
    transforms.Grayscale(num_output_channels = 3),  # 增加灰度处理，并保持3个通道
    transforms.ToTensor()
])

dirImg = 'jpg'

for i in range(0, 105):
    error = 0
    errorList = []
    print('加载第' + str(i) + '个模型...')
    # 加载模型权重
    model.load_state_dict(torch.load('checkpoints/last_' + str(i) + '.pt'))

    # 设置模型为评估模式
    model.eval()

    files = os.listdir(dirImg)
    print('图片数量：', len(files))
    for ii, name in enumerate(files):
        num = ii + 1
        if num > 1000:
            break
        # print(name)
        # 加载并预处理图像
        # 固定的左图位置
        file = './' + dirImg + '/' + name
        # print(file)
        img = Image.open(file).convert('RGB')
        left_image = img.crop((0, 200, 200, 400))
        left_image = transform(left_image)
        img_left = left_image.to(device).unsqueeze(0)

        width, height = img.size
        total_right_images = width // 200  # 动态计算右图的数量
        img_rights = [img.crop((200 * j, 0, 200 * (j + 1), 200)) for j in range(total_right_images)]
        img_rights = [transform(img_right) for img_right in img_rights]
        max_similarity = -1
        most_similar_index = -1
        for i4, img_right in enumerate(img_rights):
            img_right = img_right.to(device).unsqueeze(0)
            output = model(img_left, img_right)
            similarity = output.item()
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_index = i4
        print('第 ' + str(ii + 1) + ' 个图 ' + 'name: ' + name + ' 准确位置：', name.split('.')[0].split('_')[-1], '预测位置：',
              most_similar_index)
        if int(most_similar_index) != int(name.split('.')[0].split('_')[-1]):
            error += 1
            print('name: ' + name)
            # print('准确位置：', name.split('.')[0].split('_')[-1], '预测位置：', most_similar_index)
            errorList.append(name)

        print('错误数量: ' + str(error))
        print('-' * 100)
    data = {'i': i, 'error': error, 'errorList': errorList}
    lists.append(data)

    print('error:', error)
    print('errorList:', data)

print('errorList:', lists)
