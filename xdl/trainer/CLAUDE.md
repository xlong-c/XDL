# xdl/trainer — Claude 工作说明

## 作用

这里是 XDL 的运行时中枢，所有生命周期问题都先回到这里确认。

## 工作重点

- setup、training_step、callback、optimizer 时序是否正确
- 设备迁移和状态更新是否符合已有契约
- 变更会不会破坏现有训练入口

## 本地约束

- 谨慎改生命周期
- 先理解主循环，再下手改
