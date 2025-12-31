"""
采样动画回调
在训练过程中每个epoch结束后进行采样,并最终生成动画展示生成过程变化。

适用于VAE、GAN等生成式模型的可视化训练过程。
"""

import os
from pathlib import Path
from typing import Callable, Optional, Union

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免弹出窗口
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from xdl.callbacks.base import Callback


class SamplingAnimationCallback(Callback):
    """
    采样动画回调类

    在每个epoch结束时调用模型的采样方法,收集采样结果,
    并在训练结束后生成动画展示生成质量的变化过程。

    Args:
        n_samples: 每次采样的样本数量, 默认为16
        save_dir: 动画和中间图像的保存目录, 默认为'./others/animations'
        animation_filename: 动画文件名, 默认为'sampling_animation.gif'
        animation_format: 动画格式, 支持'gif'或'mp4', 默认为'gif'
        fps: 动画帧率, 默认为2
        sample_method: 模型的采样方法名, 默认为'generate_samples'
            - 如果模型有generate_samples方法,使用该方法
            - 如果模型有inference方法,使用inference方法
            - 也可以传入自定义的可调用对象
        sample_kwargs: 传递给采样方法的额外参数, 默认为None
        save_intermediate_images: 是否保存每个epoch的中间图像, 默认为True
        figsize: 图像大小, 默认为(8, 8)
        cmap: 图像的颜色映射, 默认为'gray'
        show_epoch_label: 是否在图像上显示epoch标签, 默认为True
        dpi: 图像分辨率, 默认为100
        enable_preview: 是否在训练结束后预览动画, 默认为False

    Examples:
        >>> # 使用默认配置
        >>> callback = SamplingAnimationCallback()
        >>> trainer = Trainer(callbacks=[callback])

        >>> # 自定义配置
        >>> callback = SamplingAnimationCallback(
        ...     n_samples=16,
        ...     save_dir='./animations',
        ...     animation_format='gif',
        ...     fps=2
        ... )
        >>> trainer = Trainer(callbacks=[callback])

        >>> # 使用自定义采样函数
        >>> def custom_sampler(model, n_samples):
        ...     with torch.no_grad():
        ...         z = torch.randn(n_samples, model.latent_dim)
        ...         z = z.to(model.device)
        ...         samples = model.decode(z)
        ...         return samples.cpu().numpy()
        >>>
        >>> callback = SamplingAnimationCallback(
        ...     sample_method=custom_sampler,
        ...     n_samples=16
        ... )
    """

    def __init__(
        self,
        n_samples: int = 16,
        save_dir: Union[str, Path] = './others/animations',
        animation_filename: str = 'sampling_animation.gif',
        animation_format: str = 'gif',
        fps: int = 2,
        sample_method: Union[str, Callable] = 'generate_samples',
        sample_kwargs: Optional[dict] = None,
        save_intermediate_images: bool = True,
        figsize: tuple = (8, 8),
        cmap: str = 'gray',
        show_epoch_label: bool = True,
        dpi: int = 100,
        enable_preview: bool = False,
        priority: int = 500,
    ):
        super().__init__(priority=priority)

        self.n_samples = n_samples
        self.save_dir = Path(save_dir)
        self.animation_filename = animation_filename
        self.animation_format = animation_format.lower()
        self.fps = fps
        self.sample_method = sample_method
        self.sample_kwargs = sample_kwargs or {}
        self.save_intermediate_images = save_intermediate_images
        self.figsize = figsize
        self.cmap = cmap
        self.show_epoch_label = show_epoch_label
        self.dpi = dpi
        self.enable_preview = enable_preview

        # 内部状态
        self._samples_history = []  # 存储每个epoch的采样结果
        self._epoch_indices = []  # 存储每个epoch的索引

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer, core_module):
        """训练开始时的初始化"""
        self._samples_history = []
        self._epoch_indices = []

        # 确定采样方法
        if callable(self.sample_method):
            self._sampler = self.sample_method
        elif isinstance(self.sample_method, str):
            if hasattr(core_module, self.sample_method):
                self._sampler = getattr(core_module, self.sample_method)
            elif hasattr(core_module, 'inference'):
                print(
                    f"警告: 模型没有 {self.sample_method} 方法, 使用 inference 方法"
                )
                self._sampler = core_module.inference
            else:
                raise AttributeError(
                    f"模型没有 {self.sample_method} 或 inference 方法, "
                    "请提供有效的采样方法或自定义采样函数"
                )
        else:
            raise TypeError(
                "sample_method 必须是字符串或可调用对象"
            )

        if trainer.is_main_process():
            print(f"采样动画回调已初始化, 保存目录: {self.save_dir}")

    def on_train_epoch_end(self, trainer, core_module):
        """每个epoch结束时进行采样"""
        # 只在主进程进行采样
        if not trainer.is_main_process():
            return

        try:
            # 设置为评估模式
            core_module.eval()

            # 调用采样方法
            if callable(self._sampler):
                # 如果采样方法需要n_samples参数
                import inspect
                sig = inspect.signature(self._sampler)

                # 准备采样参数
                sample_params = {}
                if 'n_samples' in sig.parameters:
                    sample_params['n_samples'] = self.n_samples
                if 'num_samples' in sig.parameters:
                    sample_params['num_samples'] = self.n_samples

                # 添加自定义参数
                sample_params.update(self.sample_kwargs)

                # 执行采样
                samples = self._sampler(**sample_params)
            else:
                raise RuntimeError("采样方法不是可调用对象")

            # 转换为numpy数组
            if isinstance(samples, torch.Tensor):
                samples = samples.detach().cpu().numpy()

            # 确保samples是numpy数组
            samples = np.asarray(samples)

            # 保存到历史记录
            self._samples_history.append(samples.copy())
            self._epoch_indices.append(trainer.current_epoch)

            # 保存中间图像
            if self.save_intermediate_images:
                self._save_epoch_image(
                    samples,
                    trainer.current_epoch
                )

            # 恢复训练模式
            core_module.train()

            if trainer.is_main_process():
                print(
                    f"Epoch {trainer.current_epoch}: 已采集 {self.n_samples} 个样本, "
                    f"形状: {samples.shape}"
                )

        except Exception as e:
            print(f"采样失败 (Epoch {trainer.current_epoch}): {str(e)}")
            import traceback
            traceback.print_exc()

    def on_train_end(self, trainer, core_module):
        """训练结束后生成动画"""
        if not trainer.is_main_process():
            return

        if len(self._samples_history) == 0:
            print("没有采集到样本, 跳过动画生成")
            return

        if trainer.is_main_process():
            print(f"开始生成动画, 共 {len(self._samples_history)} 帧...")

        try:
            self._generate_animation()

            if trainer.is_main_process():
                print(f"动画已保存到: {self.save_dir / self.animation_filename}")

                # 预览动画
                if self.enable_preview:
                    self._preview_animation()

        except Exception as e:
            print(f"动画生成失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _save_epoch_image(self, samples, epoch_idx):
        """保存单个epoch的采样图像"""
        try:
            fig = self._create_samples_figure(samples, epoch_idx)

            # 保存图像
            image_path = self.save_dir / f'epoch_{epoch_idx:04d}.png'
            fig.savefig(image_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"保存epoch图像失败 (Epoch {epoch_idx}): {str(e)}")

    def _create_samples_figure(self, samples, epoch_idx=None):
        """创建样本图像的figure"""
        n_samples = samples.shape[0]

        # 计算网格大小
        n_cols = int(np.ceil(np.sqrt(n_samples)))
        n_rows = int(np.ceil(n_samples / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        # 绘制每个样本
        for idx in range(n_samples):
            row = idx // n_cols
            col = idx % n_cols

            ax = axes[row, col]

            # 处理不同维度的样本
            sample = samples[idx]

            # 将torch.Tensor转换为numpy数组
            if isinstance(sample, torch.Tensor):
                sample = sample.detach().cpu().numpy()

            # 处理图像维度
            if sample.ndim == 1:
                # 展平的图像,需要reshape
                side = int(np.sqrt(sample.shape[0]))
                if side * side == sample.shape[0]:
                    sample = sample.reshape(side, side)
                else:
                    # 不是完美平方,假设是1D数据
                    ax.plot(sample)
                    ax.set_title(f'Sample {idx+1}')
                    ax.axis('off')
                    continue
            elif sample.ndim == 3:
                # 3D图像 (C, H, W) 或 (H, W, C)
                # 如果是 (1, H, W) 或 (H, W, 1)，squeeze掉通道维度
                if sample.shape[0] == 1:  # (1, H, W)
                    sample = sample.squeeze(0)
                elif sample.shape[-1] == 1:  # (H, W, 1)
                    sample = sample.squeeze(-1)
                elif sample.shape[0] in [3, 4]:  # (C, H, W)，需要转置
                    sample = np.transpose(sample, (1, 2, 0))
                # 如果是 (H, W, C)，保持不变

            # 绘制图像
            ax.imshow(sample, cmap=self.cmap)
            ax.axis('off')

            # 添加样本编号
            if n_samples <= 64:  # 只在样本数不多时显示编号
                ax.set_title(f'{idx+1}', fontsize=8)

        # 隐藏多余的子图
        for idx in range(n_samples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        # 添加epoch标签
        if self.show_epoch_label and epoch_idx is not None:
            fig.suptitle(f'Epoch {epoch_idx}', fontsize=14, fontweight='bold')

        plt.tight_layout()

        return fig

    def _generate_animation(self):
        """生成动画"""
        if self.animation_format == 'gif':
            self._generate_gif()
        elif self.animation_format == 'mp4':
            self._generate_mp4()
        else:
            raise ValueError(f"不支持的动画格式: {self.animation_format}")

    def _generate_gif(self):
        """生成GIF动画"""
        fig, ax = plt.subplots(figsize=self.figsize)

        # 初始化图像
        first_samples = self._samples_history[0]
        im = self._plot_samples_on_axis(ax, first_samples, self._epoch_indices[0])

        def update(frame_idx):
            ax.clear()
            samples = self._samples_history[frame_idx]
            epoch = self._epoch_indices[frame_idx]
            self._plot_samples_on_axis(ax, samples, epoch)
            return ax,

        # 创建动画
        anim = FuncAnimation(
            fig,
            update,
            frames=len(self._samples_history),
            interval=1000 // self.fps,
            blit=False
        )

        # 保存GIF
        gif_path = self.save_dir / self.animation_filename
        writer = PillowWriter(fps=self.fps)
        anim.save(gif_path, writer=writer, dpi=self.dpi)

        plt.close(fig)

    def _generate_mp4(self):
        """生成MP4动画"""
        try:
            import matplotlib.animation as animation
            fig, ax = plt.subplots(figsize=self.figsize)

            def update(frame_idx):
                ax.clear()
                samples = self._samples_history[frame_idx]
                epoch = self._epoch_indices[frame_idx]
                self._plot_samples_on_axis(ax, samples, epoch)
                return ax,

            # 创建动画
            anim = FuncAnimation(
                fig,
                update,
                frames=len(self._samples_history),
                interval=1000 // self.fps,
                blit=False
            )

            # 保存MP4
            mp4_path = self.save_dir / self.animation_filename
            writer = FFMpegWriter(fps=self.fps, codec='libx264')
            anim.save(mp4_path, writer=writer, dpi=self.dpi)

            plt.close(fig)

        except Exception as e:
            print(f"生成MP4失败, 请确保已安装FFmpeg: {str(e)}")
            print("尝试使用GIF格式替代...")
            self.animation_filename = self.animation_filename.replace('.mp4', '.gif')
            self._generate_gif()

    def _plot_samples_on_axis(self, ax, samples, epoch_idx):
        """在单个axis上绘制样本"""
        n_samples = samples.shape[0]

        # 计算网格大小
        n_cols = int(np.ceil(np.sqrt(n_samples)))
        n_rows = int(np.ceil(n_samples / n_cols))

        # 清空axis并创建子图网格
        ax.clear()

        # 绘制样本
        for idx in range(n_samples):
            row = idx // n_cols
            col = idx % n_cols

            # 计算子图位置
            x0 = col / n_cols
            y0 = 1 - (row + 1) / n_rows
            width = 1 / n_cols
            height = 1 / n_rows

            # 创建子图
            sub_ax = ax.inset_axes([x0, y0, width, height])

            # 处理不同维度的样本
            sample = samples[idx]

            # 将torch.Tensor转换为numpy数组
            if isinstance(sample, torch.Tensor):
                sample = sample.detach().cpu().numpy()

            # 处理图像维度
            if sample.ndim == 1:
                # 展平的图像,需要reshape
                side = int(np.sqrt(sample.shape[0]))
                if side * side == sample.shape[0]:
                    sample = sample.reshape(side, side)
                else:
                    # 不是完美平方,假设是1D数据，跳过
                    sub_ax.plot(sample)
                    sub_ax.axis('off')
                    if n_samples <= 64:
                        sub_ax.set_title(f'{idx+1}', fontsize=6)
                    continue
            elif sample.ndim == 3:
                # 3D图像 (C, H, W) 或 (H, W, C)
                # 如果是 (1, H, W) 或 (H, W, 1)，squeeze掉通道维度
                if sample.shape[0] == 1:  # (1, H, W)
                    sample = sample.squeeze(0)
                elif sample.shape[-1] == 1:  # (H, W, 1)
                    sample = sample.squeeze(-1)
                elif sample.shape[0] in [3, 4]:  # (C, H, W)，需要转置
                    sample = np.transpose(sample, (1, 2, 0))
                # 如果是 (H, W, C)，保持不变

            sub_ax.imshow(sample, cmap=self.cmap)
            sub_ax.axis('off')

            # 添加样本编号
            if n_samples <= 64:
                sub_ax.set_title(f'{idx+1}', fontsize=6)

        # 添加标题
        if self.show_epoch_label:
            ax.set_title(f'Epoch {epoch_idx}', fontsize=12, fontweight='bold')

        ax.axis('off')

        return ax

    def _preview_animation(self):
        """预览动画（在Jupyter notebook中）"""
        try:
            from IPython.display import HTML, display
            animation_path = self.save_dir / self.animation_filename
            if animation_path.exists():
                display(HTML(f'<img src="{animation_path}">'))
        except Exception:
            print("无法预览动画（非Jupyter环境或缺少依赖）")
