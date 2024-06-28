import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw

from models.siamese import Siamese
from custom_dataset import CustomSiameseDataset

# 配置参数
batch_size = 32
image_folder = './k'  # 测试数据的文件夹
checkpoint_path = "checkpoints/best.pt"  # 训练时保存的最佳模型路径
output_txt = "output_results.txt"  # 输出结果保存的文件

def mark_similar_images(img_path, index, matched_img_index):
    img = Image.open(img_path)
    width, height = img.size

    if width == 1000 and height == 400:
        num_right_imgs = 5
    elif width == 1200 and height == 400:
        num_right_imgs = 6
    else:
        raise ValueError("Unsupported image dimensions: {}x{}".format(width, height))

    img_left = img.crop((0, 200, 200, 400))  # 固定的左侧图片
    img_rights = [img.crop((200 * i, 0, 200 * (i + 1), 200)) for i in range(num_right_imgs)]

    img_right = img_rights[matched_img_index]

    # 创建一个新的图像，将两张图片并排放置
    combined_image = Image.new('RGB', (img_left.width + img_right.width, img_left.height))
    combined_image.paste(img_left, (0, 0))
    combined_image.paste(img_right, (img_left.width, 0))

    draw = ImageDraw.Draw(combined_image)
    draw.rectangle([(0, 0), (img_left.width - 1, img_left.height - 1)], outline="red", width=5)
    draw.rectangle([(img_left.width, 0), (img_left.width + img_right.width - 1, img_right.height - 1)], outline="red", width=5)

    # 保存标记的图片
    output_image_path = f"output_images/similar_{index}.jpg"
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    combined_image.save(output_image_path)
    print(f"Marked image saved to {output_image_path}")

def find_most_similar_image(model, img_left, img_rights, device):
    img_left = img_left.to(device).unsqueeze(0)
    max_similarity = -1
    most_similar_index = -1

    for i, img_right in enumerate(img_rights):
        img_right = img_right.to(device).unsqueeze(0)
        output = model(img_left, img_right)
        similarity = output.item()
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = i

    return most_similar_index

# 加载数据集
test_dataset = CustomSiameseDataset(image_folder=image_folder)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = Siamese()
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)
model.eval()  # 设置模型为评估模式

# 测试过程
results = []

with torch.no_grad():
    for batch_idx, (data1, data2, labels) in enumerate(tqdm(test_loader, desc="Testing")):
        for i in range(data1.size(0)):
            sample_index = i + batch_idx * batch_size

            img_path = test_dataset.images[sample_index]
            img = Image.open(img_path).convert('RGB')
            width, height = img.size

            if width == 1000 and height == 400:
                num_right_imgs = 5
            elif width == 1200 and height == 400:
                num_right_imgs = 6
            else:
                continue

            img_left = img.crop((0, 200, 200, 400))  # 固定的左侧图片
            img_rights = [img.crop((200 * j, 0, 200 * (j + 1), 200)) for j in range(num_right_imgs)]

            transform = test_dataset.transform
            img_left = transform(img_left)
            img_rights = [transform(img_right) for img_right in img_rights]

            most_similar_index = find_most_similar_image(model, img_left, img_rights, device)
            mark_similar_images(img_path, sample_index, most_similar_index)

print(f"Results saved to {output_txt}")
