import os
import csv
import sys

from tkinter import Tk, Canvas, Button, font
from PIL import Image, ImageTk, ImageDraw

sys.path.append(r'../')
import base


class ImageLabeler:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Labeler")

        self.image_folder = base.rootPath() + "img"  # 图片文件夹路径
        self.output_folder = "data"  # 输出文件夹路径
        self.csv_file_path = os.path.join(self.output_folder, "labels.csv")  # CSV文件路径

        print(f"图片文件夹路径: {self.image_folder}")
        self.images = [f for f in os.listdir(self.image_folder) if f.endswith(".jpg")]

        if not self.images:
            raise ValueError("图片文件夹中没有找到任何 .jpg 文件")

        self.index = 0
        self.similar_index = None

        self.skip_labeled_images()

        self.setup_ui()
        self.load_image()

    def setup_ui(self):
        self.canvas = Canvas(self.master, width = 1200, height = 600)
        self.canvas.pack()

        button_font = font.Font(size = 20)

        self.prev_button = Button(self.master, text = "上一个", command = self.prev_image, font = button_font,
                                  width = 10,
                                  height = 2)
        self.prev_button.pack(side = "left", padx = 10, pady = (10, 100))

        self.next_button = Button(self.master, text = "下一个", command = self.next_image, font = button_font,
                                  width = 10,
                                  height = 2)
        self.next_button.pack(side = "right", padx = 10, pady = (10, 100))

        self.canvas.bind("<Button-1>", self.on_click)

    def skip_labeled_images(self):
        if os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                labeled_images = {row["input_left"] for row in reader}
                while self.index < len(self.images):
                    image_path = os.path.join(self.output_folder,
                                              os.path.basename(self.images[self.index]).replace(".jpg", "_left.jpg"))
                    if image_path not in labeled_images:
                        break
                    self.index += 1

    def load_image(self):
        if self.index >= len(self.images):
            print("所有图片均已标记完成。")
            self.master.quit()
            return

        image_path = os.path.join(self.image_folder, self.images[self.index])
        self.image = Image.open(image_path)

        self.num_boxes = self.image.width // 200  # 动态计算图片数量

        self.similar_index = self.check_existing_label(image_path)
        self.display_image(self.image, "原始图片")

    def check_existing_label(self, image_path):
        if os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row["input_left"] == os.path.join(self.output_folder,
                                                         os.path.basename(image_path).replace(".jpg", "_left.jpg")):
                        for i in range(self.num_boxes):
                            if row["input_right"].endswith(f"_right_{i}.jpg") and row["label"] == "1":
                                return i
        return None

    def display_image(self, image, title):
        self.master.title(title)
        display_image = image.copy()
        if self.similar_index is not None:
            draw = ImageDraw.Draw(display_image)
            image_width, image_height = display_image.size
            box_height = image_height // 2
            box_width = image_width // self.num_boxes
            left = self.similar_index * box_width
            right = (self.similar_index + 1) * box_width
            draw.rectangle([left, 0, right, box_height], outline = "blue", width = 5)

        self.tk_image = ImageTk.PhotoImage(display_image)
        self.canvas.create_image(0, 0, anchor = "nw", image = self.tk_image)

    def on_click(self, event):
        x, y = event.x, event.y
        image_width, image_height = self.image.size
        box_height = image_height // 2
        box_width = image_width // self.num_boxes

        if y < box_height:
            for i in range(self.num_boxes):
                if i * box_width <= x < (i + 1) * box_width:
                    self.mark_image(i)
                    break

    def mark_image(self, index):
        self.similar_index = index
        self.save_label()

    def save_label(self):
        image_path = os.path.join(self.image_folder, self.images[self.index])
        data = process_image(image_path, self.output_folder, self.similar_index, self.num_boxes)

        if os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                existing_data = [row for row in reader if row["input_left"] != os.path.join(self.output_folder,
                                                                                            os.path.basename(
                                                                                                image_path).replace(
                                                                                                ".jpg", "_left.jpg"))]
        else:
            existing_data = []

        existing_data.extend(data)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        with open(self.csv_file_path, "w", newline = "") as csvfile:
            fieldnames = ["input_left", "input_right", "label"]
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            writer.writeheader()
            for row in existing_data:
                writer.writerow(row)

        self.similar_index = None
        self.next_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_image()

    def next_image(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.load_image()


def process_image(image_path, output_folder, similar_index, num_boxes):
    image = Image.open(image_path)

    image_width, image_height = image.size
    box_height = image_height // 2
    box_width = image_width // num_boxes

    input_left = image.crop((0, box_height, box_width, image_height)).resize((52, 52))
    input_right_images = [image.crop((i * box_width, 0, (i + 1) * box_width, box_height)).resize((52, 52)) for i in
                          range(num_boxes)]

    input_left_path = os.path.join(output_folder, os.path.basename(image_path).replace(".jpg", "_left.jpg"))
    input_right_paths = [os.path.join(output_folder, os.path.basename(image_path).replace(".jpg", f"_right_{i}.jpg"))
                         for i in range(num_boxes)]

    input_left.save(input_left_path)
    for i, img in enumerate(input_right_images):
        img.save(input_right_paths[i])

    return [
        {"input_left": input_left_path, "input_right": input_right_paths[i], "label": 1 if i == similar_index else 0}
        for i in range(num_boxes)]


if __name__ == "__main__":
    root = Tk()
    app = ImageLabeler(root)
    root.mainloop()
