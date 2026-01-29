#!/usr/bin/env python3
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
        except:
            self.root.state("zoomed")

        self.image_dir = start_dir
        self.aspect_ratio = aspect_ratio
        self.image_paths = []
        self.selected_images = set()  # Store selected image paths
        self.current_batch_start = 0
        self.current_batch_layout = []  # Store layout info: {'x':, 'y':, 'w':, 'h':, 'path':}
        self.deleted_in_this_batch_count = 0  # Track deletions to adjust pagination

        # Layout settings

        # Layout settings
        self.cols = 5  # Initial columns
        self.rows = 4  # Initial rows (calculated or fixed?) - we will calculate rows to fit screen or fixed
        # Let's define zoom level by number of columns.
        # Rows will be determined to fill screen vertically.

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

        # Handle window resize
        self.canvas.bind("<Configure>", self.on_resize)

        # Handle canvas click for selection
        self.canvas.bind("<Button-1>", self.on_canvas_click)

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
        self.images_cache = []  # Clear cache
        self.current_batch_layout = []  # Reset layout info
        self.deleted_in_this_batch_count = 0

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # ... (keep existing checks and layout calculation)

        # Wait for layout if too small (init)
        if canvas_w < 100:
            self.root.after(100, self.display_batch)
            return

        # Calculate Grid
        # Zero padding to maximize space
        pad = 0

        # Effective width per cell
        self.cell_w = (canvas_w - (self.cols + 1) * pad) / self.cols
        self.cell_h = (canvas_h - (self.rows + 1) * pad) / self.rows

        batch_size = self.cols * self.rows

        start_idx = self.current_batch_start
        end_idx = min(start_idx + batch_size, len(self.image_paths))

        self.current_batch_paths = self.image_paths[start_idx:end_idx]

        for i, img_path in enumerate(self.current_batch_paths):
            row = i // self.cols
            col = i % self.cols

            x = pad + col * (self.cell_w + pad)
            y = pad + row * (self.cell_h + pad)

            # Load and Resize Image
            try:
                pil_img = Image.open(img_path)

                # Smart resize: fit within cell_w x cell_h while maintaining aspect ratio
                # Use thumbnail for speed and aspect ratio preservation
                pil_img.thumbnail(
                    (int(self.cell_w), int(self.cell_h)), Image.Resampling.LANCZOS
                )

                tk_img = ImageTk.PhotoImage(pil_img)
                self.images_cache.append(tk_img)  # Keep ref

                # Center in cell
                img_w = tk_img.width()
                img_h = tk_img.height()

                off_x = (self.cell_w - img_w) / 2
                off_y = (self.cell_h - img_h) / 2

                # Store layout for fast updates
                self.current_batch_layout.append(
                    {
                        "path": img_path,
                        "x": x + off_x,
                        "y": y + off_y,
                        "w": img_w,
                        "h": img_h,
                    }
                )

                # Draw image
                self.canvas.create_image(
                    x + off_x, y + off_y, anchor=tk.NW, image=tk_img
                )

                # Visual feedback for selection: Red X on top
                if img_path in self.selected_images:
                    self.draw_selection_overlay(i)

                # Optional: Add filename text below? Might clutter.

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Placeholder in layout to keep indices aligned
                self.current_batch_layout.append(None)

        # Update Status
        self.lbl_status.config(
            text=f"{start_idx + 1}-{end_idx} / {len(self.image_paths)}"
        )

    def next_batch(self):
        # Check for pending selections
        if self.selected_images:
            if messagebox.askyesno(
                "Confirm Deletion",
                f"Delete {len(self.selected_images)} selected images?",
            ):
                self.delete_selected(confirm=False)
            else:
                # If no, clear selection? Or keep?
                # Usually if moving page, selection on previous page might be lost or confusing.
                # Let's clear it to be safe.
                self.selected_images.clear()
                self.display_batch()  # Refresh to remove red Xs
                return  # display_batch resets things, but we want to move next.
                # Actually, if user says NO, we should probably just unselect and move on.

        # Re-check selection clearing if needed, but for now let's proceed.
        # If we just called display_batch above, we are still on same page.
        # So correct flow:
        # 1. Ask. 2. Delete or Clear. 3. Calculate step. 4. Move.

        if self.selected_images:  # User said No (and delete_selected wasn't called)
            self.selected_images.clear()

        batch_size = self.cols * self.rows
        step = batch_size - self.deleted_in_this_batch_count

        if self.current_batch_start + step < len(self.image_paths):
            self.current_batch_start += step
            self.display_batch()
        elif self.deleted_in_this_batch_count > 0 and len(self.image_paths) > 0:
            # Case where we deleted items, and we are at the end, but maybe we need to refresh to fill?
            # Or just stay?
            # If we are at end, but list shrunk, ensure we are valid.
            if self.current_batch_start >= len(self.image_paths):
                self.current_batch_start = max(0, len(self.image_paths) - batch_size)
            self.display_batch()
        else:
            # Optionally loop? Or just stop.
            pass

    def prev_batch(self):
        batch_size = self.cols * self.rows
        if self.current_batch_start > 0:
            self.current_batch_start = max(0, self.current_batch_start - batch_size)
            self.display_batch()

    def jump_to_index(self):
        try:
            val = self.entry_jump.get()
            if not val:
                return
            idx = int(val)
            # Adjust 1-based index to 0-based
            idx = idx - 1
            if 0 <= idx < len(self.image_paths):
                # Snap to start of a batch page?
                # Actually, user probably just wants to see that image.
                # Let's just set start to that index, or page alignment.
                # Page alignment is cleaner.
                batch_size = self.cols * self.rows

                # Align to nearest page start
                page_start = (idx // batch_size) * batch_size

                self.current_batch_start = page_start
                self.display_batch()

                # Maybe clear entry or keep it? Keep it.
                self.canvas.focus_set()  # Return focus to canvas for keys
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
        # Recalculate ideal rows for current cols to fill screen
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 0:
            return

        cell_w = canvas_w / self.cols
        cell_h = cell_w / self.aspect_ratio

        # Calculate how many rows fit
        self.rows = max(1, int(canvas_h / cell_h))

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

        index = row * self.cols + col

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
            if confirm:
                messagebox.showinfo("Info", "No images selected.")
            return

        count = len(self.selected_images)
        if confirm:
            if not messagebox.askyesno("Confirm", f"Delete {count} images?"):
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
                # Find items at center of cell
                cx = layout["x"] + layout["w"] / 2
                cy = layout["y"] + layout["h"] / 2
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
    app = BatchImageViewer(
        root, start_dir="/mnt/f/dataset/1203_clein", aspect_ratio=4 / 1
    )
    root.mainloop()
