"""图片像素级对比器 — 支持双图对比 & 拼合图拆分对比。

用法:
  # 双图对比模式
  python img_compare.py img1.png img2.png
  python img_compare.py                            # 启动后选择文件

  # 拼合图拆分对比模式
  python img_compare.py result.png --grid --cols 5
  python img_compare.py result.png --grid --cols 5 --compare 0 3

操作:
  鼠标拖拽分界线   — 移动分界线
  滚轮             — 微调分界线位置
  ← ↑ → ↓ 方向键  — 微调分界线 (1px)
  Shift+方向键     — 快速移动分界线 (10px)
  H                — 切换横线 / 竖线
  F                — 切换适应窗口 / 原始大小
  S                — 保存当前对比截图
  R                — 交换左右/上下图片
  O / P            — 打开图片1 / 图片2 (双图模式)
  M                — 切换双图模式 / 拼合图模式
  1~9 数字键       — 拼合图模式: 快速选择对比块
  Esc / Q          — 退出
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk, ImageDraw


UNICODE_ESCAPE_RE = re.compile(r"\\+([uU])([0-9a-fA-F]{4}|[0-9a-fA-F]{8})")


def normalize_display_text(value: object) -> str:
    """Decode literal \\uXXXX / \\UXXXXXXXX sequences for display."""
    text = "" if value is None else str(value)
    if "\\u" not in text and "\\U" not in text:
        return text

    def _replace(match: re.Match[str]) -> str:
        prefix, digits = match.groups()
        expected_len = 4 if prefix == "u" else 8
        if len(digits) != expected_len:
            return match.group(0)
        try:
            return chr(int(digits, 16))
        except ValueError:
            return match.group(0)

    return UNICODE_ESCAPE_RE.sub(_replace, text)


class ImageComparer:
    def __init__(
        self,
        img1_path: str = None,
        img2_path: str = None,
        grid_mode: bool = False,
        cols: int = 5,
        compare: tuple | None = None,
    ):
        self.root = tk.Tk()
        self.root.title("图片对比器")
        self.root.configure(bg="#1e1e1e")
        self._resolve_fonts()

        # ── 模式: "single" = 双图对比, "grid" = 拼合图拆分对比 ──
        self.mode = "grid" if grid_mode else "single"

        # ── 双图模式状态 ──
        self.img1_path = None
        self.img2_path = None
        self.img1_pil: Image.Image | None = None
        self.img2_pil: Image.Image | None = None
        self.img1_tk: ImageTk.PhotoImage | None = None
        self.img2_tk: ImageTk.PhotoImage | None = None

        # ── 拼合图模式状态 ──
        self.source_path = None
        self.source_pil: Image.Image | None = None
        self.slices: list[Image.Image] = []
        self.cols = cols
        self.idx_left = 0
        self.idx_right = min(1, cols - 1) if cols > 1 else 0

        # ── 分界线 ──
        self.divider_x = 0.5  # 位置比例 0~1
        self.horizontal = False  # False=竖线, True=横线
        self.dragging = False
        self.fit_to_window = True

        # ── 窗口 ──
        self.root.geometry("1200x800")
        self.root.minsize(400, 300)

        # 工具栏
        self.orient_btn = None
        self.mode_btn = None
        self._build_toolbar()

        # 底部块选择面板（拼合图模式）
        self.block_frame = None
        self.block_buttons = []
        if self.mode == "grid":
            self._build_block_panel()

        # 画布
        self.canvas = tk.Canvas(
            self.root, bg="#2d2d2d", highlightthickness=0, cursor="sb_h_double_arrow"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ── 事件绑定 ──
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Configure>", self._on_resize)

        # 键盘
        self.root.bind("<Left>", lambda e: self._nudge_divider(-1))
        self.root.bind("<Right>", lambda e: self._nudge_divider(1))
        self.root.bind("<Up>", lambda e: self._nudge_divider(-1))
        self.root.bind("<Down>", lambda e: self._nudge_divider(1))
        self.root.bind("<Shift-Left>", lambda e: self._nudge_divider(-10))
        self.root.bind("<Shift-Right>", lambda e: self._nudge_divider(10))
        self.root.bind("<Shift-Up>", lambda e: self._nudge_divider(-10))
        self.root.bind("<Shift-Down>", lambda e: self._nudge_divider(10))
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.bind("<q>", lambda e: self.root.quit())
        self.root.bind("<o>", lambda e: self._open_image(1))
        self.root.bind("<p>", lambda e: self._open_image(2))
        self.root.bind("<f>", lambda e: self._toggle_fit())
        self.root.bind("<s>", lambda e: self._save_screenshot())
        self.root.bind("<r>", lambda e: self._swap_images())
        self.root.bind("<h>", lambda e: self._toggle_orientation())
        self.root.bind("<m>", lambda e: self._toggle_mode())

        # 数字键快速选块（拼合图模式）
        for i in range(10):
            self.root.bind(str(i), lambda e, idx=i: self._quick_select(idx))

        # ── 加载初始图片 ──
        if self.mode == "single":
            if img1_path:
                self._load_image(1, img1_path)
            if img2_path:
                self._load_image(2, img2_path)
        else:
            if img1_path:
                self._load_grid_image(img1_path)
            if compare and len(compare) >= 2:
                self.idx_left = max(0, min(compare[0], len(self.slices) - 1))
                self.idx_right = max(0, min(compare[1], len(self.slices) - 1))

        self._update_info_label()
        self.root.mainloop()

    def _resolve_fonts(self) -> None:
        available = set(tkfont.families(self.root))

        sans_priority = [
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Noto Sans SC",
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "文泉驿微米黑",
            "文泉驿正黑",
            "LXGW WenKai",
            "Microsoft YaHei",
            "微软雅黑",
            "PingFang SC",
            "SimHei",
            "黑体",
            "DengXian",
            "等线",
            "Noto Sans",
        ]
        mono_priority = [
            "Noto Sans Mono CJK SC",
            "Sarasa Mono SC",
            "Source Han Mono SC",
            "LXGW WenKai Mono",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Micro Hei",
            "文泉驿等宽微米黑",
            "文泉驿等宽正黑",
            "DejaVu Sans Mono",
            "Consolas",
            "Courier New",
            "Microsoft YaHei",
            "微软雅黑",
            "PingFang SC",
            "SimHei",
            "黑体",
        ]

        def _pick(candidates: list[str], default: str) -> str:
            for family in candidates:
                if family in available:
                    return family
            return default

        sans = _pick(sans_priority, "TkDefaultFont")
        mono = _pick(mono_priority, sans)

        self.FONT = (sans, 10)
        self.FONT_SM = (sans, 9)
        self.FONT_BOLD = (sans, 10, "bold")
        self.FONT_MONO = (mono, 10)

        named_fonts: dict[str, tuple] = {
            "TkDefaultFont": self.FONT,
            "TkTextFont": self.FONT,
            "TkMenuFont": self.FONT,
            "TkHeadingFont": self.FONT_BOLD,
            "TkCaptionFont": self.FONT,
            "TkIconFont": self.FONT,
            "TkTooltipFont": self.FONT_SM,
            "TkFixedFont": self.FONT_MONO,
        }
        for name, spec in named_fonts.items():
            try:
                named_font = tkfont.nametofont(name)
            except tk.TclError:
                continue
            config: dict[str, object] = {
                "family": spec[0],
                "size": spec[1],
            }
            if len(spec) > 2:
                config["weight"] = spec[2]
            named_font.configure(**config)

        self.root.option_add("*Font", f"{{{self.FONT[0]}}} {self.FONT[1]}")
        self.root.option_add("*Label.Font", f"{{{self.FONT[0]}}} {self.FONT[1]}")
        self.root.option_add("*Button.Font", f"{{{self.FONT[0]}}} {self.FONT[1]}")
        self.root.option_add("*Entry.Font", f"{{{self.FONT_MONO[0]}}} {self.FONT_MONO[1]}")
        self.root.option_add("*Menu.Font", f"{{{self.FONT[0]}}} {self.FONT[1]}")

        print(f"GUI 字体: 正文={self.FONT[0]}, 等宽={self.FONT_MONO[0]}")
        try:
            sans_match = subprocess.check_output(
                ["fc-match", "sans-serif:lang=zh-cn"], text=True
            ).strip()
            mono_match = subprocess.check_output(
                ["fc-match", "monospace:lang=zh-cn"], text=True
            ).strip()
            print(f"字体匹配: sans={sans_match}")
            print(f"字体匹配: mono={mono_match}")
        except Exception:
            pass

    # ═══════════════════════════════════════════════
    #  工具栏
    # ═══════════════════════════════════════════════

    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg="#2d2d2d")
        bar.pack(fill=tk.X, side=tk.TOP)

        btn_style = {
            "bg": "#3d3d3d",
            "fg": "#cccccc",
            "relief": tk.FLAT,
            "activebackground": "#505050",
            "activeforeground": "#ffffff",
            "bd": 0,
            "padx": 10,
            "pady": 4,
            "font": self.FONT,
        }

        # ── 拼合图模式设置区 ──
        self.grid_frame = tk.Frame(bar, bg="#2d2d2d")

        tk.Label(
            self.grid_frame, text="切分:", bg="#2d2d2d", fg="#888888", font=self.FONT_SM
        ).pack(side=tk.LEFT, padx=(8, 2))
        self.cols_var = tk.StringVar(value=str(self.cols))
        self.cols_entry = tk.Entry(
            self.grid_frame,
            textvariable=self.cols_var,
            width=4,
            bg="#3d3d3d",
            fg="#cccccc",
            insertbackground="#cccccc",
            relief=tk.FLAT,
            font=self.FONT_MONO,
        )
        self.cols_entry.pack(side=tk.LEFT, padx=2)
        self.cols_entry.bind("<Return>", lambda e: self._apply_grid_split())

        tk.Button(
            self.grid_frame, text="切分", command=self._apply_grid_split, **btn_style
        ).pack(side=tk.LEFT, padx=2)

        if self.mode == "grid":
            self.grid_frame.pack(side=tk.LEFT, padx=4, pady=2)

        # ── 通用按钮 ──
        self.mode_btn = tk.Button(
            bar, text=self._mode_btn_text(), command=self._toggle_mode, **btn_style
        )
        self.mode_btn.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Button(
            bar, text="打开图1 (O)", command=lambda: self._open_image(1), **btn_style
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(
            bar, text="打开图2 (P)", command=lambda: self._open_image(2), **btn_style
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(
            bar, text="适应/原始 (F)", command=self._toggle_fit, **btn_style
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(bar, text="交换 (R)", command=self._swap_images, **btn_style).pack(
            side=tk.LEFT, padx=2, pady=2
        )
        tk.Button(
            bar, text="截图保存 (S)", command=self._save_screenshot, **btn_style
        ).pack(side=tk.LEFT, padx=2, pady=2)
        self.orient_btn = tk.Button(
            bar, text="竖线 ▯ (H)", command=self._toggle_orientation, **btn_style
        )
        self.orient_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.info_label = tk.Label(
            bar, text="", bg="#2d2d2d", fg="#888888", font=self.FONT_SM
        )
        self.info_label.pack(side=tk.RIGHT, padx=10)

    def _mode_btn_text(self):
        return "拼合图模式 (M)" if self.mode == "single" else "双图模式 (M)"

    # ═══════════════════════════════════════════════
    #  底部块选择面板 (拼合图模式)
    # ═══════════════════════════════════════════════

    def _build_block_panel(self):
        if self.block_frame:
            self.block_frame.destroy()
        self.block_frame = tk.Frame(self.root, bg="#252525")
        self.block_frame.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(
            self.block_frame,
            text="点击选择对比块:",
            bg="#252525",
            fg="#888888",
            font=self.FONT_SM,
        ).pack(side=tk.LEFT, padx=8, pady=4)

        self.block_buttons = []
        for i in range(len(self.slices)):
            btn = tk.Button(
                self.block_frame,
                text=str(i + 1),
                bg="#3d3d3d",
                fg="#cccccc",
                relief=tk.FLAT,
                activebackground="#505050",
                activeforeground="#ffffff",
                bd=0,
                padx=14,
                pady=3,
                font=self.FONT,
                command=lambda idx=i: self._on_block_click(idx),
            )
            btn.pack(side=tk.LEFT, padx=1, pady=4)
            self.block_buttons.append(btn)

        self._update_block_buttons()

    def _destroy_block_panel(self):
        if self.block_frame:
            self.block_frame.destroy()
            self.block_frame = None
        self.block_buttons = []

    def _update_block_buttons(self):
        for i, btn in enumerate(self.block_buttons):
            if i == self.idx_left:
                btn.config(bg="#007acc", fg="#ffffff")
            elif i == self.idx_right:
                btn.config(bg="#cc5500", fg="#ffffff")
            else:
                btn.config(bg="#3d3d3d", fg="#cccccc")

    def _on_block_click(self, idx: int):
        if idx == self.idx_left:
            self.idx_left = self.idx_right
            self.idx_right = idx
        elif idx == self.idx_right:
            pass
        else:
            self.idx_left = self.idx_right
            self.idx_right = idx
        self._update_block_buttons()
        self._redraw()
        self._update_info_label()

    def _quick_select(self, idx: int):
        """数字键快速选择对比块 (1-based, 0=10)"""
        if self.mode != "grid" or not self.slices:
            return
        n = len(self.slices)
        if idx == 0:
            idx = 10
        idx -= 1
        if 0 <= idx < n:
            self._on_block_click(idx)

    # ═══════════════════════════════════════════════
    #  获取当前左右图
    # ═══════════════════════════════════════════════

    def _get_img_left(self):
        if self.mode == "grid":
            if 0 <= self.idx_left < len(self.slices):
                return self.slices[self.idx_left]
            return None
        return self.img1_pil

    def _get_img_right(self):
        if self.mode == "grid":
            if 0 <= self.idx_right < len(self.slices):
                return self.slices[self.idx_right]
            return None
        return self.img2_pil

    # ═══════════════════════════════════════════════
    #  图片加载
    # ═══════════════════════════════════════════════

    def _load_image(self, side: int, path: str):
        """双图模式: side=1 左边图, side=2 右边图"""
        if not os.path.isfile(path):
            messagebox.showerror("错误", normalize_display_text(f"文件不存在:\n{path}"))
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("错误", normalize_display_text(f"无法打开图片:\n{e}"))
            return

        if side == 1:
            self.img1_path = path
            self.img1_pil = img
        else:
            self.img2_path = path
            self.img2_pil = img

        self._redraw()
        self._update_info_label()

    def _open_image(self, side: int):
        if self.mode == "grid":
            # 拼合图模式: 打开新拼合图
            path = filedialog.askopenfilename(
                title="选择拼合图片",
                filetypes=[
                    ("图片", "*.png *.jpg *.jpeg *.bmp *.webp"),
                    ("所有文件", "*.*"),
                ],
            )
            if path:
                self._load_grid_image(path)
        else:
            path = filedialog.askopenfilename(
                title=f"选择图片{side}",
                filetypes=[
                    ("图片", "*.png *.jpg *.jpeg *.bmp *.webp"),
                    ("所有文件", "*.*"),
                ],
            )
            if path:
                self._load_image(side, path)

    def _load_grid_image(self, path: str):
        """拼合图模式: 加载并切分"""
        if not os.path.isfile(path):
            messagebox.showerror("错误", normalize_display_text(f"文件不存在:\n{path}"))
            return
        try:
            self.source_pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("错误", normalize_display_text(f"无法打开图片:\n{e}"))
            return

        self.source_path = path
        self._do_split()
        self._build_block_panel()
        self._redraw()
        self._update_info_label()

    def _do_split(self):
        """按 cols 横向等分切分"""
        self.slices = []
        if self.source_pil is None:
            return

        cols = max(2, self.cols)
        slice_w = self.source_pil.width // cols
        for i in range(cols):
            x0 = i * slice_w
            x1 = x0 + slice_w if i < cols - 1 else self.source_pil.width
            self.slices.append(
                self.source_pil.crop((x0, 0, x1, self.source_pil.height))
            )

        n = len(self.slices)
        self.idx_left = max(0, min(self.idx_left, n - 1))
        self.idx_right = max(0, min(self.idx_right, n - 1))

        if n:
            print(
                f"已切分为 {n} 块 (每块 {self.slices[0].width}x{self.slices[0].height})"
            )

    def _apply_grid_split(self):
        """从输入框读取 cols 并重新切分"""
        try:
            cols = int(self.cols_var.get())
            if cols < 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "切分数量必须是不小于 2 的整数")
            return
        self.cols = cols
        self._do_split()
        self._build_block_panel()
        self._redraw()
        self._update_info_label()

    # ═══════════════════════════════════════════════
    #  绘制
    # ═══════════════════════════════════════════════

    def _redraw(self, *_):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        img_left = self._get_img_left()
        img_right = self._get_img_right()

        if img_left is None and img_right is None:
            msg = (
                "按 O / P 打开图片"
                if self.mode == "single"
                else "按 O 打开拼合图，输入切分数后回车"
            )
            self.canvas.create_text(
                w // 2,
                h // 2,
                text=msg,
                fill="#666666",
                font=(self.FONT_BOLD[0], 16, "bold"),
                justify=tk.CENTER,
            )
            return

        display_w, display_h = self._calc_display_size(w, h)

        # 绘制底图（右边/下边）
        if img_right is not None:
            r = self._resize_image(img_right, display_w, display_h)
            self.img2_tk = ImageTk.PhotoImage(r)
            self.canvas.create_image(
                0, 0, anchor=tk.NW, image=self.img2_tk, tags="right"
            )

        if self.horizontal:
            # ── 横线：上图1 下图2 ──
            div = int(display_h * self.divider_x)
            if img_left is not None and div > 0:
                r = self._resize_image(img_left, display_w, display_h)
                top = r.crop((0, 0, display_w, div))
                self.img1_tk = ImageTk.PhotoImage(top)
                self.canvas.create_image(
                    0, 0, anchor=tk.NW, image=self.img1_tk, tags="left"
                )
            self.canvas.create_line(
                0,
                div,
                display_w,
                div,
                fill="#ffffff",
                width=1,
                stipple="gray50",
                tags="divider",
            )
            hr = 4
            self.canvas.create_oval(
                w // 2 - hr,
                div - hr,
                w // 2 + hr,
                div + hr,
                fill="",
                outline="#aaaaaa",
                width=1,
                tags="divider",
            )
        else:
            # ── 竖线：左图1 右图2 ──
            div = int(display_w * self.divider_x)
            if img_left is not None and div > 0:
                r = self._resize_image(img_left, display_w, display_h)
                left = r.crop((0, 0, div, display_h))
                self.img1_tk = ImageTk.PhotoImage(left)
                self.canvas.create_image(
                    0, 0, anchor=tk.NW, image=self.img1_tk, tags="left"
                )
            self.canvas.create_line(
                div,
                0,
                div,
                display_h,
                fill="#ffffff",
                width=1,
                stipple="gray50",
                tags="divider",
            )
            hr = 4
            self.canvas.create_oval(
                div - hr,
                h // 2 - hr,
                div + hr,
                h // 2 + hr,
                fill="",
                outline="#aaaaaa",
                width=1,
                tags="divider",
            )

        # 拼合图模式: 显示块编号标签
        if self.mode == "grid":
            self.canvas.create_text(
                20,
                20,
                text=f"#{self.idx_left + 1}",
                fill="#007acc",
                font=(self.FONT_MONO[0], 14, "bold"),
                anchor=tk.NW,
            )
            self.canvas.create_text(
                display_w - 20,
                20,
                text=f"#{self.idx_right + 1}",
                fill="#cc5500",
                font=(self.FONT_MONO[0], 14, "bold"),
                anchor=tk.NE,
            )

        self.canvas.config(
            cursor="sb_v_double_arrow" if self.horizontal else "sb_h_double_arrow"
        )

    def _calc_display_size(self, canvas_w: int, canvas_h: int) -> tuple[int, int]:
        if self.fit_to_window:
            return canvas_w, canvas_h

        max_w, max_h = 1, 1
        for img in (self._get_img_left(), self._get_img_right()):
            if img:
                max_w = max(max_w, img.width)
                max_h = max(max_h, img.height)
        return min(max_w, canvas_w), min(max_h, canvas_h)

    @staticmethod
    def _resize_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        ratio = min(target_w / img.width, target_h / img.height)
        new_w = max(1, int(img.width * ratio))
        new_h = max(1, int(img.height * ratio))
        return img.resize((new_w, new_h), Image.LANCZOS)

    # ═══════════════════════════════════════════════
    #  鼠标事件
    # ═══════════════════════════════════════════════

    def _on_mouse_down(self, event):
        self.dragging = True

    def _on_mouse_drag(self, event):
        if self.dragging:
            self._set_divider_from_pos(event.x if not self.horizontal else event.y)

    def _on_mouse_up(self, event):
        self.dragging = False

    def _on_mouse_wheel(self, event):
        self._nudge_divider(1 if event.delta > 0 else -1)

    def _set_divider_from_pos(self, pos: int):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w > 0 and h > 0:
            dw, dh = self._calc_display_size(w, h)
            dim = dh if self.horizontal else dw
            if dim > 0:
                self.divider_x = max(0.0, min(1.0, pos / dim))
                self._redraw()

    def _nudge_divider(self, px: int):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w > 0 and h > 0:
            dw, dh = self._calc_display_size(w, h)
            dim = dh if self.horizontal else dw
            if dim > 0:
                self.divider_x += px / dim
                self.divider_x = max(0.0, min(1.0, self.divider_x))
                self._redraw()

    def _on_resize(self, event):
        self._redraw()

    # ═══════════════════════════════════════════════
    #  功能
    # ═══════════════════════════════════════════════

    def _toggle_mode(self):
        if self.mode == "single":
            self.mode = "grid"
            self.grid_frame.pack(side=tk.LEFT, padx=4, pady=2)
            if self.source_pil:
                self._build_block_panel()
            self._redraw()
        else:
            self.mode = "single"
            self.grid_frame.pack_forget()
            self._destroy_block_panel()
            self._redraw()
        self.mode_btn.config(text=self._mode_btn_text())
        self._update_info_label()
        print(f"模式切换: {'拼合图拆分' if self.mode == 'grid' else '双图对比'}")

    def _toggle_orientation(self):
        self.horizontal = not self.horizontal
        self.orient_btn.config(text="横线 ▬ (H)" if self.horizontal else "竖线 ▯ (H)")
        mode = "横向（上图1 下图2）" if self.horizontal else "竖向（左图1 右图2）"
        self._redraw()
        print(f"分界线方向: {mode}")

    def _toggle_fit(self):
        self.fit_to_window = not self.fit_to_window
        self._redraw()
        self._update_info_label()
        print(f"显示模式: {'适应窗口' if self.fit_to_window else '原始大小'}")

    def _swap_images(self):
        if self.mode == "grid":
            self.idx_left, self.idx_right = self.idx_right, self.idx_left
            self._update_block_buttons()
        else:
            self.img1_path, self.img2_path = self.img2_path, self.img1_path
            self.img1_pil, self.img2_pil = self.img2_pil, self.img1_pil
        self.divider_x = 1.0 - self.divider_x
        self._redraw()
        self._update_info_label()
        print("已交换")

    def _save_screenshot(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        display_w, display_h = self._calc_display_size(w, h)
        img_left = self._get_img_left()
        img_right = self._get_img_right()

        if img_left is None and img_right is None:
            return

        if img_right is not None:
            result = self._resize_image(img_right, display_w, display_h)
        else:
            result = Image.new("RGB", (display_w, display_h), (0, 0, 0))

        draw = ImageDraw.Draw(result)

        if self.horizontal:
            div = int(display_h * self.divider_x)
            if img_left is not None and div > 0:
                r = self._resize_image(img_left, display_w, display_h)
                result.paste(r.crop((0, 0, display_w, div)), (0, 0))
            draw.line([(0, div), (display_w, div)], fill="#ffffff", width=1)
        else:
            div = int(display_w * self.divider_x)
            if img_left is not None and div > 0:
                r = self._resize_image(img_left, display_w, display_h)
                result.paste(r.crop((0, 0, div, display_h)), (0, 0))
            draw.line([(div, 0), (div, display_h)], fill="#ffffff", width=1)

        path = filedialog.asksaveasfilename(
            title="保存对比截图",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if path:
            result.save(path, quality=99)
            print(f"截图已保存: {path}")

    def _update_info_label(self):
        parts = []
        if self.mode == "grid":
            if self.source_pil:
                name = os.path.basename(self.source_path or "拼合图")
                parts.append(
                    normalize_display_text(
                        f"拼合: {name} ({self.source_pil.width}x{self.source_pil.height})"
                    )
                )
            if self.slices:
                parts.append(f"共 {len(self.slices)} 块")
                parts.append(f"对比: #{self.idx_left + 1} vs #{self.idx_right + 1}")
        else:
            if self.img1_pil:
                name = os.path.basename(self.img1_path or "图1")
                parts.append(
                    normalize_display_text(
                        f"图1: {name} ({self.img1_pil.width}x{self.img1_pil.height})"
                    )
                )
            if self.img2_pil:
                name = os.path.basename(self.img2_path or "图2")
                parts.append(
                    normalize_display_text(
                        f"图2: {name} ({self.img2_pil.width}x{self.img2_pil.height})"
                    )
                )
        parts.append("适应" if self.fit_to_window else "原始")
        self.info_label.config(text=normalize_display_text("  |  ".join(parts)))


def main():
    # ============================================================
    # 配置：直接修改下方字典来设置参数
    # ============================================================
    CONFIG = {
        "mode": "single",  # "single" = 双图对比, "grid" = 拼合图拆分
        "img1": None,  # 图片1路径 (None = 启动后选择)
        "img2": None,  # 图片2路径 (仅双图模式)
        "cols": 5,  # 拼合图横向切分数
        "compare": None,  # 对比块 (0-based), e.g. (0, 3) or None
    }

    if CONFIG["mode"] == "grid" and CONFIG["cols"] < 2:
        print("错误: CONFIG['cols'] 至少为 2", file=sys.stderr)
        sys.exit(1)

    compare_tuple = tuple(CONFIG["compare"]) if CONFIG["compare"] else None

    ImageComparer(
        img1_path=CONFIG["img1"],
        img2_path=CONFIG["img2"],
        grid_mode=(CONFIG["mode"] == "grid"),
        cols=CONFIG["cols"],
        compare=compare_tuple,
    )


if __name__ == "__main__":
    main()
