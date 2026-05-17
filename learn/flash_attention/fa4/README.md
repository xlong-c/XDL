# FlashAttention-4 极简阅读版

这个目录只保留 6 个最核心的阅读文件, 目标不是运行, 而是快速抓住 FA4 forward
主线的代码组织。

## 适合什么时候看

- 你已经看过 `01` 到 `06`
- 你不想先陷进完整工程细节
- 你只想先建立 “FA4 forward 是怎么分层的” 直觉

## 文件顺序

1. `flash_fwd.py`
   - 通用 forward 基类, 能看清 tile / softmax / mask / scheduler 怎么拼起来。
2. `softmax.py`
   - online softmax 状态更新。
3. `mask.py`
   - causal / local / block sparse 的掩码逻辑。
4. `tile_scheduler.py`
   - tile 调度策略。
5. `flash_fwd_sm90.py`
   - Hopper 版本 forward。
6. `flash_fwd_sm100.py`
   - Blackwell 版本 forward。

## 注意

- 这是阅读摘录, 不是可直接运行的完整子包。
- 文件内部还会保留一部分上游 import 痕迹, 这里的目标是读主线结构, 不是保证本目录单独可运行。
- 如果后面你想继续追辅助模块, 再回官方仓库或我继续按需补文件就行。
