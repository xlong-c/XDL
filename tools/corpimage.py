import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import glob

class ImageCropper:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("图片裁切工具 - 批量处理")
        self.root.geometry("1280x800")
        
        # 图片相关变量
        self.original_image = None
        self.display_image = None
        self.photo = None
        
        # 文件夹和图片列表
        self.input_folder = ""
        self.output_folder = ""
        self.image_files = []
        self.current_index = 0
        
        # 裁切框参数 (4:3比例, 宽3高4), 增大可视裁切框
        self.crop_width = 450
        self.crop_height = 600
        
        # 图片显示参数
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 鼠标拖拽相关
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        
        self.input_folder_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        
        self.is_processing = False
        self.keep_settings_var = tk.BooleanVar(value=False)
        self.start_index_var = tk.StringVar(value="1")
        
        self.setup_ui()
        self.bind_events()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        # file_menu.add_command(label="选择文件夹", command=self.select_folders) # 移除弹窗选择
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建画布
        self.canvas = tk.Canvas(main_frame, bg='gray', width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 创建控制面板
        control_frame = tk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)
        
        # 添加说明标签
        tk.Label(control_frame, text="操作说明:", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        instructions = [
            "• 鼠标拖拽移动图片",
            "• WASD键移动图片",
            "• Q/E键或滚轮缩放图片",
            "• 空格键保存并切换下一张",
            "• 裁切比例: 3:4 (宽:高)"
        ]
        
        for instruction in instructions:
            tk.Label(control_frame, text=instruction, justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        
        # 添加输入框和加载按钮
        tk.Label(control_frame, text="输入文件夹:", justify=tk.LEFT).pack(anchor=tk.W, pady=(20,0))
        self.input_entry = tk.Entry(control_frame, textvariable=self.input_folder_var, width=28)
        self.input_entry.pack(fill=tk.X, pady=2, padx=2)

        tk.Label(control_frame, text="输出文件夹:", justify=tk.LEFT).pack(anchor=tk.W, pady=(5,0))
        self.output_entry = tk.Entry(control_frame, textvariable=self.output_folder_var, width=28)
        self.output_entry.pack(fill=tk.X, pady=2, padx=2)

        tk.Label(control_frame, text="起始图片序号:", justify=tk.LEFT).pack(anchor=tk.W, pady=(5,0))
        self.start_index_entry = tk.Entry(control_frame, textvariable=self.start_index_var, width=10)
        self.start_index_entry.pack(anchor=tk.W, pady=2, padx=2)

        tk.Button(control_frame, text="加载图片", command=self.load_images, width=15).pack(pady=(10, 5))
        
        # 添加按钮
        # tk.Button(control_frame, text="选择文件夹", command=self.select_folders, width=15).pack(pady=(20, 5))
        tk.Button(control_frame, text="上一张", command=self.prev_image, width=15).pack(pady=5)
        tk.Button(control_frame, text="下一张", command=self.next_image, width=15).pack(pady=5)
        tk.Button(control_frame, text="跳过", command=self.next_image, width=15).pack(pady=5)
        tk.Button(control_frame, text="重置视图", command=self.reset_view, width=15).pack(pady=5)
        tk.Button(control_frame, text="保存裁切", command=self.save_crop, width=15).pack(pady=5)
        
        tk.Checkbutton(control_frame, text="沿用上一张设置", variable=self.keep_settings_var, anchor=tk.W).pack(pady=(10, 0), fill=tk.X)
        
        # 显示当前缩放比例和进度
        self.scale_label = tk.Label(control_frame, text="PPI: 72")
        self.scale_label.pack(pady=(20, 0))
        
        self.progress_label = tk.Label(control_frame, text="进度: 0/0")
        self.progress_label.pack(pady=(5, 0))
        
        # 显示当前文件名
        self.filename_label = tk.Label(control_frame, text="", wraplength=180, justify=tk.LEFT)
        self.filename_label.pack(pady=(10, 0))
        
    def bind_events(self):
        """绑定事件"""
        self.canvas.bind("<Button-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag_image)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
        self.canvas.bind("<MouseWheel>", self.zoom_image)
        self.root.bind("<KeyPress-space>", self.save_and_next)
        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("<KeyPress-q>", self.zoom_in_key)
        self.root.bind("<KeyPress-e>", self.zoom_out_key)
        self.root.bind("<KeyPress-a>", self.move_left)
        self.root.bind("<KeyPress-d>", self.move_right)
        self.root.bind("<KeyPress-w>", self.move_up)
        self.root.bind("<KeyPress-s>", self.move_down)
        self.canvas.focus_set()  # 确保画布可以接收键盘事件
        
    def move_left(self, event=None):
        """向左移动图片"""
        # self.offset_x += self.crop_width / 10
        self.prev_image()

    def move_right(self, event=None):
        """向右移动图片"""
        # self.offset_x -= self.crop_width / 10
        self.next_image()

    def move_up(self, event=None):
        """向上移动图片"""
        self.offset_y += self.crop_height / 10
        self.update_display()

    def move_down(self, event=None):
        """向下移动图片"""
        self.offset_y -= self.crop_height / 10
        self.update_display()
        
    def load_images(self):
        """从输入框中的路径加载图片"""
        input_folder = self.input_folder_var.get()
        output_folder = self.output_folder_var.get()

        if not input_folder or not os.path.isdir(input_folder):
            messagebox.showerror("错误", "请输入有效的输入文件夹路径！")
            return
        
        if not output_folder:
            messagebox.showerror("错误", "请输入输出文件夹路径！")
            return

        if not os.path.isdir(output_folder):
            # 询问是否创建目录
            if messagebox.askyesno("确认", f"输出文件夹 '{output_folder}' 不存在或不是一个目录, 是否创建？"):
                try:
                    os.makedirs(output_folder, exist_ok=True)
                except OSError as e:
                    messagebox.showerror("错误", f"无法创建输出文件夹：{e}")
                    return
            else:
                # 如果用户不创建, 则中止
                messagebox.showwarning("操作取消", "未指定有效的输出文件夹。")
                return

        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
        found_files = []
        
        for ext in image_extensions:
            found_files.extend(glob.glob(os.path.join(input_folder, ext)))
            found_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
        # 去除因Windows下glob不区分大小写而可能产生的重复项
        self.image_files = sorted(list(set(found_files)))
        
        if not self.image_files:
            messagebox.showwarning("警告", "在选择的文件夹中未找到图片文件！")
            return
            
        # 解析起始序号
        try:
            start_index = int(self.start_index_var.get())
            if 1 <= start_index <= len(self.image_files):
                self.current_index = start_index - 1
            else:
                messagebox.showwarning("警告", f"起始序号必须在 1 和 {len(self.image_files)} 之间。将从第一张开始。")
                self.current_index = 0
        except ValueError:
            messagebox.showwarning("警告", "无效的起始序号。将从第一张开始。")
            self.current_index = 0
        
        messagebox.showinfo("成功", f"找到 {len(self.image_files)} 张图片, 将从第 {self.current_index + 1} 张开始。")
        
        # 加载第一张图片
        self.load_current_image()
        
    def load_current_image(self):
        """加载当前索引的图片"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
            
        try:
            image_path = self.image_files[self.current_index]
            img = Image.open(image_path)

            # 限制加载图片的最大分辨率, 防止卡顿
            max_dim = 4000
            if img.width > max_dim or img.height > max_dim:
                # 使用 thumbnail 来按比例缩小, 它会直接修改图像对象
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            self.original_image = img
            
            if not self.keep_settings_var.get():
                self.reset_view()
            else:
                self.update_display() # 沿用现有缩放和偏移
            
            # 更新界面信息
            filename = os.path.basename(image_path)
            self.filename_label.config(text=f"当前文件:\n{filename}")
            self.progress_label.config(text=f"进度: {self.current_index + 1}/{len(self.image_files)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
    
    def prev_image(self, event=None):
        """切换到上一张图片"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self, event=None):
        """切换到下一张图片"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
        elif self.current_index >= len(self.image_files) - 1:
            messagebox.showinfo("完成", "已处理完所有图片！")
    
    def save_and_next(self, event=None):
        """保存当前图片并切换到下一张"""
        if self.is_processing:
            return  # 防止重复触发

        self.is_processing = True
        try:
            if self.save_crop(auto_filename=True):
                # 使用 after 确保在当前事件处理完成后再调用 next_image
                # 这可以避免潜在的UI更新延迟问题
                self.root.after(50, self.next_image_and_unlock)
            else:
                self.is_processing = False # 如果保存失败, 也需要解锁
        except Exception as e:
            self.is_processing = False # 确保出错时也能解锁
            messagebox.showerror("错误", f"处理时发生错误: {e}")

    def next_image_and_unlock(self):
        """切换到下一张图片并解锁"""
        self.next_image()
        self.is_processing = False
    
    def reset_view(self):
        """重置视图"""
        if self.original_image:
            # 限制初始显示宽度为600px
            target_width = 600
            img_width = self.original_image.width
            
            if img_width > target_width:
                self.scale = target_width / img_width
            else:
                self.scale = 1.0

            self.offset_x = 0
            self.offset_y = 0
            self.update_display()
    
    def start_drag(self, event):
        """开始拖拽"""
        self.dragging = True
        self.last_x = event.x
        self.last_y = event.y
        self.canvas.focus_set()  # 确保画布获得焦点以接收键盘事件
    
    def drag_image(self, event):
        """拖拽图片"""
        if self.dragging and self.original_image:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            
            self.offset_x += dx
            self.offset_y += dy
            
            self.last_x = event.x
            self.last_y = event.y
            
            self.update_display()
    
    def stop_drag(self, event):
        """停止拖拽"""
        self.dragging = False
    
    def zoom(self, factor):
        """通过给定的因子缩放图片"""
        if self.original_image:
            new_scale = self.scale * factor
            
            # 移除缩放范围限制
            self.scale = new_scale
            self.update_display()

    def zoom_in_key(self, event=None):
        """按下 'q' 键放大"""
        self.zoom(1.25)

    def zoom_out_key(self, event=None):
        """按下 'e' 键缩小"""
        self.zoom(0.75)

    def zoom_image(self, event):
        """缩放图片"""
        if self.original_image:
            # 计算缩放因子, 使用更精确的值
            zoom_factor = 1.05 if event.delta > 0 else 0.95
            self.zoom(zoom_factor)
    
    def update_display(self):
        """更新显示"""
        if not self.original_image:
            return
        
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # 1. 计算缩放后的完整图片尺寸
        scaled_width = self.original_image.width * self.scale
        scaled_height = self.original_image.height * self.scale

        # 2. 计算完整图片在画布上的中心和左上角位置
        img_center_x = canvas_width / 2 + self.offset_x
        img_center_y = canvas_height / 2 + self.offset_y
        
        img_left_on_canvas = img_center_x - scaled_width / 2
        img_top_on_canvas = img_center_y - scaled_height / 2

        # 3. 计算画布与图片的交集, 以确定要渲染的原图区域
        # 交集相对于缩放后图片的左上角坐标
        visible_part_left_rel = max(0, -img_left_on_canvas)
        visible_part_top_rel = max(0, -img_top_on_canvas)
        visible_part_right_rel = min(scaled_width, canvas_width - img_left_on_canvas)
        visible_part_bottom_rel = min(scaled_height, canvas_height - img_top_on_canvas)

        # 4. 将交集区域映射回原图坐标
        crop_x1 = visible_part_left_rel / self.scale
        crop_y1 = visible_part_top_rel / self.scale
        crop_x2 = visible_part_right_rel / self.scale
        crop_y2 = visible_part_bottom_rel / self.scale

        # 如果图片完全在画布外, 则不进行渲染
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            self.draw_crop_frame()
            self.scale_label.config(text=f"缩放: {int(self.scale * 100)}%")
            return

        # 5. 从原图中裁切出需要显示的部分
        image_to_render = self.original_image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))

        # 6. 将裁切出的部分缩放到其在画布上的显示尺寸
        display_width = int(image_to_render.width * self.scale)
        display_height = int(image_to_render.height * self.scale)

        if display_width > 0 and display_height > 0:
            resized_image = image_to_render.resize((display_width, display_height), Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(resized_image)

            # 7. 计算渲染图片在画布上的左上角位置
            draw_x = max(0, img_left_on_canvas)
            draw_y = max(0, img_top_on_canvas)

            self.canvas.create_image(draw_x, draw_y, image=self.photo, anchor=tk.NW)
        
        # 绘制裁切框 (固定在画布中心)
        self.draw_crop_frame()
        
        # 更新缩放标签
        ppi = int(self.scale * 72) # 假设屏幕基础PPI为72
        self.scale_label.config(text=f"PPI: {ppi}")
        
    def draw_crop_frame(self):
        """绘制裁切框"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 计算裁切框位置 (固定在画布中心)
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        x1 = center_x - self.crop_width // 2
        y1 = center_y - self.crop_height // 2
        x2 = center_x + self.crop_width // 2
        y2 = center_y + self.crop_height // 2
        
        # 绘制裁切框
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="crop_frame")
        
        # 绘制遮罩 (半透明覆盖)
        # 上方遮罩
        self.canvas.create_rectangle(0, 0, canvas_width, y1, fill="black", stipple="gray50", tags="mask")
        # 下方遮罩
        self.canvas.create_rectangle(0, y2, canvas_width, canvas_height, fill="black", stipple="gray50", tags="mask")
        # 左侧遮罩
        self.canvas.create_rectangle(0, y1, x1, y2, fill="black", stipple="gray50", tags="mask")
        # 右侧遮罩
        self.canvas.create_rectangle(x2, y1, canvas_width, y2, fill="black", stipple="gray50", tags="mask")
        
        # 添加标签
        self.canvas.create_text(center_x, y1 - 20, text="裁切区域 (3:4)", fill="red", font=("Arial", 12, "bold"))
    
    def save_crop(self, event=None, auto_filename=False):
        """保存裁切的图片"""
        if not self.original_image:
            messagebox.showwarning("警告", "请先加载图片！")
            return False
            
        if not self.output_folder:
            messagebox.showwarning("警告", "请先选择输出文件夹！")
            return False
        
        try:
            # 计算裁切区域在原图中的坐标
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # 画布中心点
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            # 图片在画布上的中心点
            img_center_x = canvas_center_x + self.offset_x
            img_center_y = canvas_center_y + self.offset_y
            
            # 裁切框在画布上的坐标
            crop_x1 = canvas_center_x - self.crop_width // 2
            crop_y1 = canvas_center_y - self.crop_height // 2
            
            # 转换为原图坐标
            scaled_width = int(self.original_image.width * self.scale)
            scaled_height = int(self.original_image.height * self.scale)
            
            # 图片左上角在画布上的位置
            img_left = img_center_x - scaled_width // 2
            img_top = img_center_y - scaled_height // 2
            
            # 裁切框相对于图片左上角的坐标
            rel_x1 = crop_x1 - img_left
            rel_y1 = crop_y1 - img_top
            
            # 转换为原图坐标, 并确保尺寸精确
            orig_x1 = int(rel_x1 / self.scale)
            orig_y1 = int(rel_y1 / self.scale)
            orig_x2 = int(orig_x1 + self.crop_width / self.scale)
            orig_y2 = int(orig_y1 + self.crop_height / self.scale)

            # 裁切框在原图上的尺寸
            crop_w = orig_x2 - orig_x1
            crop_h = orig_y2 - orig_y1

            if crop_w <= 0 or crop_h <= 0:
                messagebox.showwarning("警告", "无效的裁切区域！")
                return False

            # 创建一个最终的图片, 尺寸与裁切框在原图上对应的一致, 背景为白色
            final_image = Image.new('RGB', (crop_w, crop_h), 'white')

            # 计算原图和裁切框的交集
            actual_crop_x1 = max(0, orig_x1)
            actual_crop_y1 = max(0, orig_y1)
            actual_crop_x2 = min(self.original_image.width, orig_x2)
            actual_crop_y2 = min(self.original_image.height, orig_y2)

            # 如果有交集
            if actual_crop_x2 > actual_crop_x1 and actual_crop_y2 > actual_crop_y1:
                # 从原图中裁切出交集部分
                image_to_paste = self.original_image.crop((actual_crop_x1, actual_crop_y1, actual_crop_x2, actual_crop_y2))

                # 计算这部分应该粘贴到 final_image 的什么位置
                paste_x = actual_crop_x1 - orig_x1
                paste_y = actual_crop_y1 - orig_y1

                final_image.paste(image_to_paste, (paste_x, paste_y))
            
            # 生成保存路径
            if auto_filename and self.image_files:
                original_filename = os.path.basename(self.image_files[self.current_index])
                name, ext = os.path.splitext(original_filename)
                save_filename = f"{name}_cropped.jpg"
                save_path = os.path.join(self.output_folder, save_filename)
            else:
                save_path = filedialog.asksaveasfilename(
                    title="保存裁切图片",
                    defaultextension=".jpg",
                    filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
                    initialdir=self.output_folder
                )
                
            if save_path:
                final_image.save(save_path, quality=95)
                if not auto_filename:
                    messagebox.showinfo("成功", f"图片已保存到: {save_path}")
                return True
                
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
            return False
        
        return False
    
    def run(self):
        """运行应用"""
        # 显示初始说明
        if not self.image_files:
            self.canvas.create_text(
                400, 300, 
                text="请选择输入和输出文件夹开始批量裁切\n\n操作说明:\n• 鼠标拖拽移动图片\n• 滚轮缩放图片\n• 空格键保存并切换下一张\n• 左右箭头键切换图片",
                font=("Arial", 14),
                justify=tk.CENTER
            )
        
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageCropper()
    app.run()