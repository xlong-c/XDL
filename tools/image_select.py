import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

class ImageProcessorApp: 
    def __init__(self, root, image_folder='./', output_folder='./'):
        self.root = root
        self.root.title("Image Processor (PNG/JPG/JPEG)")
        # 设置窗口初始大小, 适合1080p分辨率 (1920x1080)
        # 减去任务栏高度, 给窗口留出合适空间
        self.root.geometry("1500x950")  # 宽度1500, 高度950, 充分利用1080p空间

        # 预设文件夹路径
        self.image_folder = image_folder or os.path.join(os.path.expanduser("~"), "Pictures")
        self.output_folder = output_folder or os.path.join(os.path.expanduser("~"), "Pictures", "processed")
        self.current_image_path = ""
        self.original_image = None
        self.display_image = None
        self.photo_image = None
        self.rows_to_display = []
        self.png_files = []
        self.current_image_index = -1
        self.processed_files = set() # Track processed images
        self.scale_factor = 1.0  # 缩放比例, 默认为1.0(原始大小)

        self.create_widgets()
        
        # 初始化时填充预设路径
        self.image_folder_entry.insert(0, self.image_folder)
        self.output_folder_entry.insert(0, self.output_folder)
        self.update_processed_files_list() # Populate processed files list on startup
        
        # 绑定空格键到下一张图片
        self.root.bind('<space>', lambda event: self.load_next_image())

    def create_widgets(self):
        # Main frame split into left (controls) and right (image display)
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Controls panel with scrollbar
        left_panel = tk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Create a canvas with scrollbar for left panel
        self.left_canvas = tk.Canvas(left_panel, width=350, highlightthickness=0)
        self.left_scrollbar = tk.Scrollbar(left_panel, orient="vertical", command=self.left_canvas.yview)
        self.left_scrollable_frame = tk.Frame(self.left_canvas)

        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)
        self.left_canvas.pack(side="left", fill="both", expand=True)
        self.left_scrollbar.pack(side="right", fill="y")

        # Create a window inside the canvas to hold the scrollable content
        self.canvas_frame = self.left_canvas.create_window((0, 0), window=self.left_scrollable_frame, anchor="nw")

        # Configure canvas to expand with window size
        def configure_scroll_region(event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

        def configure_canvas_width(event):
            # Ensure the scrollable frame width matches the canvas width
            self.left_canvas.itemconfig(self.canvas_frame, width=self.left_canvas.winfo_width())

        self.left_scrollable_frame.bind("<Configure>", configure_scroll_region)
        self.left_canvas.bind("<Configure>", configure_canvas_width)

        # Mouse wheel scrolling
        def on_mousewheel(event):
            self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def bind_mousewheel(event):
            self.left_canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            self.left_canvas.unbind_all("<MouseWheel>")

        self.left_canvas.bind("<Enter>", bind_mousewheel)
        self.left_canvas.bind("<Leave>", unbind_mousewheel)

        # Right side - Image display
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls in left panel - Folder selection
        folder_label_frame = tk.LabelFrame(self.left_scrollable_frame, text="文件夹设置", padx=10, pady=10)
        folder_label_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(folder_label_frame, text="图片文件夹:").pack(anchor=tk.W)
        self.image_folder_entry = tk.Entry(folder_label_frame, width=40)
        self.image_folder_entry.pack(fill=tk.X, pady=(0, 5))
        tk.Button(folder_label_frame, text="浏览", command=self.browse_image_folder).pack(anchor=tk.W)

        tk.Label(folder_label_frame, text="输出文件夹:").pack(anchor=tk.W, pady=(10, 0))
        self.output_folder_entry = tk.Entry(folder_label_frame, width=40)
        self.output_folder_entry.pack(fill=tk.X, pady=(0, 5))
        tk.Button(folder_label_frame, text="浏览", command=self.browse_output_folder).pack(anchor=tk.W)

        # Start index selection
        start_label_frame = tk.LabelFrame(self.left_scrollable_frame, text="起始位置设置", padx=10, pady=10)
        start_label_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(start_label_frame, text="起始图片:").pack(anchor=tk.W)
        self.start_index_var = tk.StringVar(value="1")
        start_inner_frame = tk.Frame(start_label_frame)
        start_inner_frame.pack(fill=tk.X, pady=(0, 5))
        self.start_index_entry = tk.Entry(start_inner_frame, textvariable=self.start_index_var, width=10)
        self.start_index_entry.pack(side=tk.LEFT)
        tk.Button(start_inner_frame, text="设置", command=self.set_start_index).pack(side=tk.LEFT, padx=(10, 0))

        # Control buttons
        button_label_frame = tk.LabelFrame(self.left_scrollable_frame, text="图像操作", padx=10, pady=10)
        button_label_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Button(button_label_frame, text="加载图片", command=self.load_image, height=2).pack(fill=tk.X, pady=2)
        tk.Button(button_label_frame, text="下一张", command=self.load_next_image, height=2).pack(fill=tk.X, pady=2)
        tk.Button(button_label_frame, text="跳过图片", command=self.skip_image, height=2).pack(fill=tk.X, pady=2)
        tk.Button(button_label_frame, text="保存并下一张", command=self.save_image, height=2).pack(fill=tk.X, pady=2)

        # Progress bar
        progress_label_frame = tk.LabelFrame(self.left_scrollable_frame, text="进度信息", padx=10, pady=10)
        progress_label_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(progress_label_frame, text="进度:").pack(anchor=tk.W)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(progress_label_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.progress_var, length=200, state='disabled')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.progress_label = tk.Label(progress_label_frame, text="0/0")
        self.progress_label.pack(anchor=tk.W)

        # Scale control
        scale_label_frame = tk.LabelFrame(self.left_scrollable_frame, text="缩放控制", padx=10, pady=10)
        scale_label_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(scale_label_frame, text="缩放比例:").pack(anchor=tk.W)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = tk.Scale(scale_label_frame, from_=0.1, to=2.0, resolution=0.05,
                                   orient=tk.HORIZONTAL, variable=self.scale_var,
                                   command=self.on_scale_change, length=200)
        self.scale_slider.pack(fill=tk.X, pady=(0, 5))
        tk.Button(scale_label_frame, text="重置缩放", command=self.reset_scale).pack(anchor=tk.W)

        # Row deletion buttons frame (placed in left panel)
        self.row_buttons_label_frame = tk.LabelFrame(self.left_scrollable_frame, text="行操作", padx=10, pady=10)
        self.row_buttons_label_frame.pack(fill=tk.X, pady=(0, 10))

        # Inner frame to hold row deletion buttons
        self.row_buttons_frame = tk.Frame(self.row_buttons_label_frame)
        self.row_buttons_frame.pack(fill=tk.X)

        # Canvas for image display in right panel
        self.canvas = tk.Canvas(right_panel, bg="white", width=600, height=800)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def browse_image_folder(self):
        folder_selected = filedialog.askdirectory(initialdir=self.image_folder)
        if folder_selected:
            self.image_folder = folder_selected
            self.image_folder_entry.delete(0, tk.END)
            self.image_folder_entry.insert(0, folder_selected)

    def browse_output_folder(self):
        folder_selected = filedialog.askdirectory(initialdir=self.output_folder)
        if folder_selected:
            self.output_folder = folder_selected
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder_selected)

    def on_scale_change(self, value):
        """当缩放滑块值改变时调用"""
        try:
            self.scale_factor = float(value)
            if self.display_image is not None:
                self.update_image_display()
        except ValueError:
            pass

    def reset_scale(self):
        """重置缩放比例为1.0"""
        self.scale_var.set(1.0)
        self.scale_factor = 1.0
        if self.display_image is not None:
            self.update_image_display()

    def set_start_index(self):
        """设置起始图片索引"""
        try:
            start_index = int(self.start_index_var.get()) - 1  # Convert to 0-based index
            if start_index < 0:
                start_index = 0
            self.current_image_index = start_index
            messagebox.showinfo("成功", f"已设置从第 {start_index + 1} 张图片开始")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    def update_progress(self):
        """更新进度条显示"""
        if not self.png_files:
            self.progress_var.set(0)
            self.progress_label.config(text="0/0")
            return
        
        total = len(self.png_files)
        current = self.current_image_index + 1
        progress_percent = (current / total) * 100
        
        self.progress_var.set(progress_percent)
        self.progress_label.config(text=f"{current}/{total}")

    def load_image(self):
        if not self.image_folder:
            messagebox.showerror("Error", "Please select an image folder first.")
            return

        if not self.png_files:
            # If png_files is empty, populate it
            all_files = sorted([f for f in os.listdir(self.image_folder)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if not all_files:
                messagebox.showerror("Error", "No PNG files found in the selected folder.")
                return
            self.png_files = sorted([os.path.join(self.image_folder, f) for f in all_files])
            self.current_image_index = 0
            self.update_processed_files_list()

        # Skip already processed images
        while self.current_image_index < len(self.png_files):
            current_filename = os.path.basename(self.png_files[self.current_image_index])
            name, ext = os.path.splitext(current_filename)
            # For JPG files, ensure we use .jpg extension for processed files
            if ext.lower() in ['.jpeg', '.jpg']:
                processed_filename = f"{name}_processed.jpg"
            else:
                processed_filename = f"{name}_processed{ext}"
            if processed_filename in self.processed_files:
                self.current_image_index += 1
            else:
                break

        if self.current_image_index >= len(self.png_files):
            messagebox.showinfo("Info", "All images processed!")
            self.canvas.delete("all")
            self.original_image = None
            self.display_image = None
            self.photo_image = None
            self.rows_to_display = []
            return

        self.current_image_path = self.png_files[self.current_image_index]
        try:
            self.original_image = cv2.imread(self.current_image_path)
            if self.original_image is None:
                messagebox.showerror("Error", f"Failed to load image: {self.current_image_path}")
                return
            self.display_image = self.original_image.copy()
            self.rows_to_display = list(range(4)) # Initially display all 4 rows
            self.update_image_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def update_image_display(self):
        if self.display_image is None or self.original_image is None:
            return

        # Create a new blank image to draw the selected rows onto
        height, width = self.original_image.shape[:2]
        row_height = height // 4
        
        new_height = len(self.rows_to_display) * row_height
        if new_height == 0:
            self.canvas.delete("all")
            self.photo_image = None
            return

        combined_image = np.zeros((new_height, width, 3), dtype=np.uint8)
        
        for i, row_idx in enumerate(self.rows_to_display):
            # Calculate the box for the current row in the original image
            top = row_idx * row_height
            bottom = (row_idx + 1) * row_height
            
            # Crop the row from the original image
            row_image = self.original_image[top:bottom, :]
            
            # Paste it into the combined image at the correct new position
            combined_image[i*row_height:(i+1)*row_height, :] = row_image

        self.display_image = combined_image
        # Apply scale factor
        img_height, img_width = self.display_image.shape[:2]
        
        # 应用缩放比例
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        # 确保最小尺寸为1像素
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        self.display_image = cv2.resize(self.display_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        img_width, img_height = new_width, new_height
        
        # Convert OpenCV image (BGR) to PIL image (RGB)
        rgb_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.canvas.config(width=img_width, height=img_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        # Update row deletion buttons
        self.update_row_buttons()

    def on_canvas_click(self, event):
        if self.display_image is None or not self.rows_to_display: # Check if there are rows to display
            return

        img_height = self.display_image.shape[0]
        row_height_in_display = img_height // len(self.rows_to_display) if len(self.rows_to_display) > 0 else 0
        
        if row_height_in_display == 0:
            return

        clicked_row_in_display = event.y // row_height_in_display

        if 0 <= clicked_row_in_display < len(self.rows_to_display):
            self.delete_row(clicked_row_in_display)

    def delete_row(self, clicked_row_in_display):
        if self.display_image is None or not self.rows_to_display:
            return

        if 0 <= clicked_row_in_display < len(self.rows_to_display):
            del self.rows_to_display[clicked_row_in_display]
            self.update_image_display()

    def update_row_buttons(self):
        # Clear existing buttons
        for widget in self.row_buttons_frame.winfo_children():
            widget.destroy()

        if self.display_image is None or not self.rows_to_display:
            return

        # Create buttons for each row to display
        for i in range(len(self.rows_to_display)):
            original_row_index = self.rows_to_display[i]
            btn = tk.Button(self.row_buttons_frame, text=f"删除行 {original_row_index + 1}",
                            command=lambda idx=i: self.delete_row(idx))
            btn.pack(fill=tk.X, pady=2)

    def update_processed_files_list(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            self.processed_files.clear()
            return
        self.processed_files = set([f for f in os.listdir(self.output_folder) if f.lower().endswith('.png')])

    def _save_current_image(self, show_message=True):
        if self.original_image is None:
            if show_message:
                messagebox.showerror("Error", "No image to save.")
            return False

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        original_filename = os.path.basename(self.current_image_path)
        name, ext = os.path.splitext(original_filename)
        save_path = os.path.join(self.output_folder, f"{name}_processed{ext}")

        try:
            # Determine which image to save - create the processed image without scaling
            height, width = self.original_image.shape[:2]
            row_height = height // 4
            
            new_height = len(self.rows_to_display) * row_height
            if new_height == 0:
                # If no rows to display, create a minimal image or handle appropriately
                combined_image = np.zeros((row_height, width, 3), dtype=np.uint8)
            else:
                combined_image = np.zeros((new_height, width, 3), dtype=np.uint8)
                
            for i, row_idx in enumerate(self.rows_to_display):
                # Calculate the box for the current row in the original image
                top = row_idx * row_height
                bottom = (row_idx + 1) * row_height
                
                # Crop the row from the original image
                row_image = self.original_image[top:bottom, :]
                
                # Paste it into the combined image at the correct new position
                combined_image[i*row_height:(i+1)*row_height, :] = row_image

            cv2.imwrite(save_path, combined_image)
            self.update_processed_files_list() # Update the list after saving
            if show_message:
                messagebox.showinfo("Success", f"Image saved to {save_path}")
            return True
        except Exception as e:
            if show_message:
                messagebox.showerror("Error", f"Failed to save image: {e}")
            return False

    def load_next_image(self):
        # Save current image before loading next
        if self.original_image is not None: # Only save if an image was loaded
            self._save_current_image(show_message=False) # Save silently

        if self.current_image_index < len(self.png_files) - 1:
            self.current_image_index += 1
            self.load_image()
        else:
            messagebox.showinfo("Info", "No more images to load.")
            self.update_progress()  # 更新最终进度

    def skip_image(self):
        # No need to save, just move to next
        if self.current_image_path: # Ensure an image was loaded
            current_filename = os.path.basename(self.current_image_path)
            name, ext = os.path.splitext(current_filename)
            processed_filename = f"{name}_processed{ext}"
            self.processed_files.add(processed_filename) # Mark as processed
        if self.current_image_index < len(self.png_files) - 1:
            self.current_image_index += 1
            self.load_image()
        else:
            messagebox.showinfo("Info", "No more images to skip.")
            self.update_progress()  # 更新最终进度

    def save_image(self):
        self._save_current_image(show_message=True)
        # After manual save, move to next image
        self.load_next_image()

if __name__ == "__main__":
    root = tk.Tk()
    # 可以在创建应用时指定预设文件夹路径
    app = ImageProcessorApp(root)
    root.mainloop()