#!/usr/bin/env python3
"""
批量图片查看器 (Batch Image Viewer)

功能说明:
    基于 tkinter 的 GUI 批量图片查看器,支持网格布局和选择删除。
    适用于快速浏览大量图片并进行筛选删除的场景。

核心功能:
    - 网格布局显示: 可配置列数,自动计算行数以填充屏幕
    - 图片选择: 点击选择/取消选择,红色 X 标记已选图片
    - 批量删除: 支持删除选中的图片文件
    - 缩放控制: 通过列数调整实现缩放(列数越少,图片越大)
    - 翻页浏览: 支持上一页/下一页,可跳转到指定索引

快捷键:
    ← (左箭头)  - 上一页
    → (右箭头)  - 下一页
    ↑ (上箭头)  - 放大(减少列数)
    ↓ (下箭头)  - 缩小(增加列数)
    Delete       - 删除选中的图片
    Escape       - 退出程序

使用方法:
    修改脚本底部的 start_dir 和 aspect_ratio 参数:

    app = BatchImageViewer(
        root,
        start_dir="/path/to/images",  # 图片文件夹路径
        aspect_ratio=4/1              # 图片显示比例
    )

依赖:
    - tkinter (Python 标准库)
    - Pillow (PIL)

作者: xdl 项目
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


class BatchImageViewer:
    def __init__(self, root, start_dir=".", aspect_ratio=16 / 9):
        self.root = root
        self.root.title("Batch Image Viewer")

        # Start maximized or fullscreen
        # Linux/X11 specific: try to maximize
        try:
            self.root.attributes("-zoomed", True)
        except tk.TclError:
            self.root.state("zoomed")

        self.image_dir = start_dir
        self.aspect_ratio = aspect_ratio
        self.image_paths = []
        self.selected_images = set()  # Store selected image paths
        self.current_batch_start = 0
        self.current_batch_layout = []  # Store layout info: {'x':, 'y':, 'w':, 'h':, 'path':}
        self.deleted_in_this_batch_count = 0  # Track deletions to adjust pagination

        # Layout settings
        self.cols = 2  # Initial columns
        self.rows = 2  # Initial rows (calculated or fixed?) - we will calculate rows to fit screen or fixed
        # Let's define zoom level by number of columns.
        # Rows will be determined to fill screen vertically.

        self.slice_enabled = True
        self.slice_indices = [2, 3]
        self.slice_parts = 4

        self.pad_x = 0
        self.pad_y = 0

        self.images_cache = []  # Keep references to PhotoImages to prevent garbage collection

        # UI Setup
        self.setup_ui()

        # Load initial images
        self.load_images_from_dir(self.image_dir)

        # Bind keys
        self.root.bind("<Left>", lambda e: self.prev_batch())
        self.root.bind("<Right>", lambda e: self.next_batch())
        self.root.bind("<Up>", lambda e: self.zoom_in())
        self.root.bind("<Down>", lambda e: self.zoom_out())
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.bind("<Delete>", lambda e: self.delete_selected())

    def setup_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, bg="white")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Image Display Area (Canvas)
        self.canvas = tk.Canvas(self.main_frame, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bottom Control Panel
        self.controls_frame = tk.Frame(self.root, bg="gray90", height=50)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Buttons
        btn_style = {
            "bg": "gray80",
            "fg": "black",
            "font": ("Arial", 10),
            "bd": 1,
            "relief": tk.RAISED,
        }

        self.btn_open = tk.Button(
            self.controls_frame,
            text="Open Folder",
            command=self.open_folder,
            **btn_style,
        )
        self.btn_open.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_delete = tk.Button(
            self.controls_frame,
            text="Delete Selected",
            command=self.delete_selected,
            **btn_style,
        )
        self.btn_delete.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_prev = tk.Button(
            self.controls_frame,
            text="<< Prev Batch",
            command=self.prev_batch,
            **btn_style,
        )
        self.btn_prev.pack(side=tk.LEFT, padx=10, pady=10)

        self.lbl_status = tk.Label(
            self.controls_frame, text="0/0", bg="gray90", fg="black", font=("Arial", 10)
        )
        self.lbl_status.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_next = tk.Button(
            self.controls_frame,
            text="Next Batch >>",
            command=self.next_batch,
            **btn_style,
        )
        self.btn_next.pack(side=tk.LEFT, padx=10, pady=10)

        # Jump to index
        tk.Label(self.controls_frame, text="Jump to:", bg="gray90").pack(
            side=tk.LEFT, padx=(20, 5)
        )
        self.entry_jump = tk.Entry(self.controls_frame, width=6)
        self.entry_jump.pack(side=tk.LEFT, padx=5)
        self.entry_jump.bind("<Return>", lambda e: self.jump_to_index())

        tk.Button(
            self.controls_frame, text="Go", command=self.jump_to_index, **btn_style
        ).pack(side=tk.LEFT, padx=5)

        self.btn_zoom_out = tk.Button(
            self.controls_frame, text="Zoom Out (-)", command=self.zoom_out, **btn_style
        )
        self.btn_zoom_out.pack(side=tk.RIGHT, padx=10, pady=10)

        self.btn_zoom_in = tk.Button(
            self.controls_frame, text="Zoom In (+)", command=self.zoom_in, **btn_style
        )
        self.btn_zoom_in.pack(side=tk.RIGHT, padx=10, pady=10)

        tk.Label(self.controls_frame, text="Slice:", bg="gray90").pack(
            side=tk.LEFT, padx=(20, 5)
        )
        self.slice_var = tk.StringVar(value="2,3")
        self.slice_entry = tk.Entry(
            self.controls_frame, width=8, textvariable=self.slice_var
        )
        self.slice_entry.pack(side=tk.LEFT, padx=5)
        self.slice_entry.bind("<Return>", lambda e: self.on_slice_change())

        tk.Label(self.controls_frame, text="PadX:", bg="gray90").pack(
            side=tk.LEFT, padx=(20, 5)
        )
        self.pad_x_var = tk.StringVar(value=str(self.pad_x))
        self.pad_x_entry = tk.Entry(
            self.controls_frame, width=4, textvariable=self.pad_x_var
        )
        self.pad_x_entry.pack(side=tk.LEFT, padx=5)
        self.pad_x_entry.bind("<Return>", lambda e: self.on_pad_change())

        tk.Label(self.controls_frame, text="PadY:", bg="gray90").pack(
            side=tk.LEFT, padx=(10, 5)
        )
        self.pad_y_var = tk.StringVar(value=str(self.pad_y))
        self.pad_y_entry = tk.Entry(
            self.controls_frame, width=4, textvariable=self.pad_y_var
        )
        self.pad_y_entry.pack(side=tk.LEFT, padx=5)
        self.pad_y_entry.bind("<Return>", lambda e: self.on_pad_change())

        tk.Label(self.controls_frame, text="Ratio:", bg="gray90").pack(
            side=tk.LEFT, padx=(20, 5)
        )
        self.ratio_var = tk.StringVar(value="4")
        self.ratio_entry = tk.Entry(
            self.controls_frame, width=6, textvariable=self.ratio_var
        )
        self.ratio_entry.pack(side=tk.LEFT, padx=5)
        self.ratio_entry.bind("<Return>", lambda e: self.on_ratio_change())

        tk.Label(self.controls_frame, text="BG:", bg="gray90").pack(
            side=tk.LEFT, padx=(20, 5)
        )
        self.bg_color_var = tk.StringVar(value="#000000")
        self.bg_color_entry = tk.Entry(
            self.controls_frame, width=8, textvariable=self.bg_color_var
        )
        self.bg_color_entry.pack(side=tk.LEFT, padx=5)
        self.bg_color_entry.bind("<Return>", lambda e: self.on_bg_color_change())

        # Handle window resize
        self.canvas.bind("<Configure>", self.on_resize)

        # Handle canvas click for selection
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", lambda e: self.next_batch())

    def load_images_from_dir(self, directory):
        self.image_dir = directory
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
        try:
            self.image_paths = sorted(
                [
                    os.path.join(directory, f)
                    for f in os.listdir(directory)
                    if f.lower().endswith(valid_exts)
                ]
            )
            self.current_batch_start = 0
            self.display_batch()
        except Exception as e:
            print(f"Error reading directory: {e}")
            messagebox.showerror("Error", f"Could not read directory: {e}")

    def open_folder(self):
        dir_path = filedialog.askdirectory(initialdir=self.image_dir)
        if dir_path:
            self.load_images_from_dir(dir_path)

    def get_batch_size(self):
        # Dynamically calculate rows based on screen aspect and cols
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return self.cols * self.rows  # Fallback

        # Estimate cell width
        cell_w = canvas_w / self.cols
        # Use provided aspect ratio
        cell_h = cell_w / self.aspect_ratio

        # How many rows fit?
        self.rows = max(1, int(canvas_h / cell_h))

        return self.cols * self.rows

    def display_batch(self):
        if not self.image_paths:
            self.lbl_status.config(text="No images")
            self.canvas.delete("all")
            return

        self.canvas.delete("all")
        self.images_cache = []
        self.current_batch_layout = []
        self.deleted_in_this_batch_count = 0

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 100:
            self.root.after(100, self.display_batch)
            return

        num_slices = len(self.slice_indices) if self.slice_enabled else 1

        effective_cols = (
            self.cols * self.slice_parts // num_slices
            if self.slice_enabled
            else self.cols
        )

        self.cell_w = (canvas_w - (effective_cols + 1) * self.pad_x) / effective_cols
        self.cell_h = (canvas_h - (self.rows + 1) * self.pad_y) / self.rows

        batch_size = effective_cols * self.rows

        start_idx = self.current_batch_start
        end_idx = min(start_idx + batch_size, len(self.image_paths))

        self.current_batch_paths = self.image_paths[start_idx:end_idx]

        for i, img_path in enumerate(self.current_batch_paths):
            row = i // effective_cols
            col = i % effective_cols

            x = self.pad_x + col * (self.cell_w + self.pad_x)
            y = self.pad_y + row * (self.cell_h + self.pad_y)

            try:
                pil_img = Image.open(img_path)

                if self.slice_enabled and num_slices > 0:
                    img_width, img_height = pil_img.size
                    slice_width = img_width // self.slice_parts
                    slices = []
                    for slice_idx in self.slice_indices:
                        idx = slice_idx - 1
                        left = idx * slice_width
                        right = left + slice_width
                        slices.append(pil_img.crop((left, 0, right, img_height)))
                    if slices:
                        total_width = sum(s.width for s in slices)
                        max_height = max(s.height for s in slices)
                        pil_img = Image.new("RGB", (total_width, max_height))
                        x_offset = 0
                        for s in slices:
                            pil_img.paste(s, (x_offset, 0))
                            x_offset += s.width

                pil_img.thumbnail(
                    (int(self.cell_w), int(self.cell_h)), Image.Resampling.LANCZOS
                )

                tk_img = ImageTk.PhotoImage(pil_img)
                self.images_cache.append(tk_img)

                img_w = tk_img.width()
                img_h = tk_img.height()

                off_x = (self.cell_w - img_w) / 2
                off_y = (self.cell_h - img_h) / 2

                self.current_batch_layout.append(
                    {
                        "path": img_path,
                        "x": x + off_x,
                        "y": y + off_y,
                        "w": img_w,
                        "h": img_h,
                    }
                )

                self.canvas.create_image(
                    x + off_x, y + off_y, anchor=tk.NW, image=tk_img
                )

                if img_path in self.selected_images:
                    self.draw_selection_overlay(i)

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                self.current_batch_layout.append(None)

        actual_displayed = min(batch_size, len(self.image_paths) - start_idx)
        self.lbl_status.config(
            text=f"{start_idx + 1}-{start_idx + actual_displayed} / {len(self.image_paths)} (cols:{effective_cols})"
        )

    def get_effective_cols(self):
        num_slices = len(self.slice_indices) if self.slice_enabled else 1
        return (
            self.cols * self.slice_parts // num_slices
            if self.slice_enabled
            else self.cols
        )

    def next_batch(self):
        if self.selected_images:
            self.delete_selected(confirm=False)

        if self.selected_images:
            self.selected_images.clear()

        effective_cols = self.get_effective_cols()
        batch_size = effective_cols * self.rows
        step = batch_size - self.deleted_in_this_batch_count

        if self.current_batch_start + step < len(self.image_paths):
            self.current_batch_start += step
            self.display_batch()
        elif self.deleted_in_this_batch_count > 0 and len(self.image_paths) > 0:
            if self.current_batch_start >= len(self.image_paths):
                self.current_batch_start = max(0, len(self.image_paths) - batch_size)
            self.display_batch()

    def prev_batch(self):
        effective_cols = self.get_effective_cols()
        batch_size = effective_cols * self.rows
        if self.current_batch_start > 0:
            self.current_batch_start = max(0, self.current_batch_start - batch_size)
            self.display_batch()

    def jump_to_index(self):
        try:
            val = self.entry_jump.get()
            if not val:
                return
            idx = int(val)
            idx = idx - 1
            if 0 <= idx < len(self.image_paths):
                effective_cols = self.get_effective_cols()
                batch_size = effective_cols * self.rows

                page_start = (idx // batch_size) * batch_size

                self.current_batch_start = page_start
                self.display_batch()

                self.canvas.focus_set()
            else:
                messagebox.showwarning(
                    "Invalid Index",
                    f"Please enter a number between 1 and {len(self.image_paths)}",
                )
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number.")

    def zoom_in(self):
        # Fewer columns = larger images
        if self.cols > 1:
            self.cols -= 1
            # Recalculate rows happens in get_batch_size / display loop if we make it dynamic
            # But here we used fixed rows logic in display_batch based on `self.rows`?
            # Actually display_batch uses self.rows. We need to update self.rows.

            # Simple heuristic: maintain roughly same aspect ratio of the grid or screen fill
            # If we decrease cols, we should probably decrease rows to make images bigger vertically too.
            # Let's re-eval rows based on screen height
            self.update_grid_dims()
            self.display_batch()

    def zoom_out(self):
        # More columns = smaller images
        if self.cols < 20:
            self.cols += 1
            self.update_grid_dims()
            self.display_batch()

    def update_grid_dims(self):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 0:
            return

        cell_w = canvas_w / self.cols
        cell_h = cell_w / self.aspect_ratio

        self.rows = max(1, int(canvas_h / cell_h))

    def on_slice_change(self):
        try:
            slice_str = self.slice_var.get()
            indices = []
            for part in slice_str.split(","):
                idx = int(part.strip())
                if 1 <= idx <= self.slice_parts:
                    indices.append(idx)
            if indices:
                self.slice_indices = sorted(set(indices))
                self.display_batch()
        except ValueError:
            pass

    def on_pad_change(self):
        try:
            new_pad_x = int(self.pad_x_var.get())
            new_pad_y = int(self.pad_y_var.get())
            self.pad_x = max(0, new_pad_x)
            self.pad_y = max(0, new_pad_y)
            self.display_batch()
        except ValueError:
            pass

    def on_ratio_change(self):
        try:
            ratio_str = self.ratio_var.get()
            if "/" in ratio_str:
                parts = ratio_str.split("/")
                self.aspect_ratio = float(parts[0]) / float(parts[1])
            else:
                self.aspect_ratio = float(ratio_str)
            self.update_grid_dims()
            self.display_batch()
        except (ValueError, ZeroDivisionError):
            pass

    def on_bg_color_change(self):
        color = self.bg_color_var.get()
        if len(color) == 7 and color.startswith("#"):
            try:
                int(color[1:], 16)
                self.canvas.config(bg=color)
            except ValueError:
                pass

    def draw_selection_overlay(self, index):
        if index >= len(self.current_batch_layout):
            return

        layout = self.current_batch_layout[index]
        if not layout:
            return

        x, y, w, h = layout["x"], layout["y"], layout["w"], layout["h"]
        tag = f"sel_{index}"

        # Draw X
        self.canvas.create_line(x, y, x + w, y + h, fill="red", width=3, tags=tag)
        self.canvas.create_line(x, y + h, x + w, y, fill="red", width=3, tags=tag)

    def on_canvas_click(self, event):
        if (
            not hasattr(self, "cell_w")
            or not hasattr(self, "cell_h")
            or self.cell_w <= 0
            or self.cell_h <= 0
        ):
            return

        col = int(event.x / self.cell_w)
        row = int(event.y / self.cell_h)

        num_slices = len(self.slice_indices) if self.slice_enabled else 1
        effective_cols = (
            self.cols * self.slice_parts // num_slices
            if self.slice_enabled
            else self.cols
        )

        index = row * effective_cols + col

        if 0 <= index < len(self.current_batch_paths):
            img_path = self.current_batch_paths[index]
            self.toggle_selection(index, img_path)

    def on_resize(self, event):
        # When window resizes, re-calculate grid layout
        # Debounce could be good, but simple is fine for now
        self.update_grid_dims()
        self.display_batch()

    def toggle_selection(self, index, img_path):
        if img_path in self.selected_images:
            self.selected_images.remove(img_path)
            # Remove overlay
            self.canvas.delete(f"sel_{index}")
        else:
            self.selected_images.add(img_path)
            # Add overlay
            self.draw_selection_overlay(index)

    def delete_selected(self, confirm=True):
        if not self.selected_images:
            return

        deleted_count = 0
        indices_to_clear = []

        # Identify visual items to clear
        for i, layout in enumerate(self.current_batch_layout):
            if layout and layout["path"] in self.selected_images:
                indices_to_clear.append(i)

        for img_path in list(self.selected_images):
            try:
                os.remove(img_path)
                if img_path in self.image_paths:
                    self.image_paths.remove(img_path)
                    # Count if it was in the current batch (part of the view)
                    # We can check if it was in current_batch_layout
                    # But simpler: we know we are deleting from selection.
                    # We need to increment deleted_in_this_batch_count if it was on screen.
                    # We checked layout above.
                self.selected_images.remove(img_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {img_path}: {e}")

        # Update visuals without full reload
        for i in indices_to_clear:
            self.deleted_in_this_batch_count += 1
            # Remove image
            # We didn't store image item ID in layout, but we can find by location or tag?
            # We didn't tag images.
            # Let's just clear the area?
            # Or better: `display_batch` should store item IDs.
            # Since we didn't store IDs, we have to rely on clearing selection tags.
            # And for the image itself... we can't easily delete just the image object without ID.
            # Wait, we can find_closest or find_overlapping?
            layout = self.current_batch_layout[i]
            if layout:
                items = self.canvas.find_overlapping(
                    layout["x"],
                    layout["y"],
                    layout["x"] + layout["w"],
                    layout["y"] + layout["h"],
                )
                for item in items:
                    self.canvas.delete(item)

            # Clear layout entry so we don't interact with it anymore
            self.current_batch_layout[i] = None

        if confirm:
            # Only show message if manual trigger
            # messagebox.showinfo("Result", f"Deleted {deleted_count} images.")
            # User requested "don't refill", so we just leave it empty.
            pass

        # Ensure focus returns to canvas or root for keybinds
        self.canvas.focus_set()


if __name__ == "__main__":
    root = tk.Tk()
    # Default to current directory if not provided
    # app = BatchImageViewer(root, start_dir="/mnt/f/dataset/1203_clein")
    app = BatchImageViewer(root, start_dir="./", aspect_ratio=4 / 1)
    root.mainloop()
