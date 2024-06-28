import re
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os, sys
sys.path.append(r'../')
import base


class ImageMarkerApp:
    def __init__(self, root, image_folder):
        self.root = root
        self.image_folder = image_folder

        self.images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

        marked_images = [f for f in self.images if 'marked' in f]
        other_images = [f for f in self.images if 'marked' not in f]

        self.images = other_images + marked_images
        self.current_image_index = 0
        self.marked_index = None
        self.load_image(self.current_image_index)
        self.create_navigation_buttons()

    def mark_image(self, i, j):
        self.marked_index = i
        self.rename_file_for_mark(self.current_image_index, i)
        print(f"Marked segment ({i}, {j}) of the current image.")
        self.display_image()

        self.next_image()

    def rename_file_for_mark(self, img_index, index):
        old_file_path = self.images[img_index]
        directory, filename = os.path.split(old_file_path)
        name, ext = os.path.splitext(filename)

        if '_marked' not in name:
            new_name = f"{name}_marked"
            new_file_path = os.path.join(directory, f"{new_name}_{index}{ext}")
            os.rename(old_file_path, new_file_path)
            self.images[img_index] = new_file_path
        else:

            marked_name, current_index = name.rsplit('_marked', 1)

            new_index = f"{int(current_index) + 1}" if current_index.isdigit() else f"{index}"
            new_name = f"{marked_name}_marked_{new_index}"
            new_file_path = os.path.join(directory, f"{new_name}{ext}")
            os.rename(old_file_path, new_file_path)
            self.images[img_index] = new_file_path

    def check_marked_segments(self, img_index):
        self.marked_index = None
        file_path = self.images[img_index]
        filename = os.path.basename(file_path)
        if '_marked_' in filename:
            number = re.search(r"marked_(\d+)", filename).group(1)
            self.marked_index = number

    def load_image(self, index):
        if index < 0 or index >= len(self.images):
            return
        self.current_image = Image.open(self.images[index])
        self.check_marked_segments(index)
        self.display_image()

    def display_image(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_navigation_buttons()

        for i in range(6):
            for j in range(2):
                cropped_image = self.current_image.crop((200 * i, 200 * j, 200 * (i + 1), 200 * (j + 1)))
                if j == 0 and self.marked_index and i == int(self.marked_index):
                    draw = ImageDraw.Draw(cropped_image)
                    draw.rectangle([(0, 0), (199, 199)], outline = "red", width = 10)

                tk_image = ImageTk.PhotoImage(cropped_image)
                button = tk.Button(self.root, image = tk_image, command = lambda i = i, j = j: self.mark_image(i, j))
                button.image = tk_image
                button.grid(row = j + 1, column = i)

    def create_navigation_buttons(self):
        prev_button = tk.Button(self.root, text = "Previous", command = self.prev_image)
        prev_button.grid(row = 0, column = 0)

        next_button = tk.Button(self.root, text = "Next", command = self.next_image)
        next_button.grid(row = 0, column = 5)

    def next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.load_image(self.current_image_index)

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.current_image_index)


def main():
    root = tk.Tk()
    root.geometry("1500x650")
    app = ImageMarkerApp(root, "./data")
    root.mainloop()


if __name__ == "__main__":
    main()
