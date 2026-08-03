#!/usr/bin/env bash
# quiet.sh — dbus-run-session 启动 GUI 工具, 静默已知无害的桌面/驱动噪声
#
# 用法:
#   ./quiet.sh <script.py> [args...]
#   相对路径先在 CWD 找, 再在 quiet.sh 同目录找, 最后用绝对路径.
#
# 等价于:
#   dbus-run-session python <script> [...]
#   但 stderr 中 xdg-desktop-portal / PipeWire / RealtimeKit / MESA ZINK /
#   max-threads-overflow / dbus-daemon 激活日志 / SpiRegistry 这些
#   已知无害警告被过滤.  Python 真 traceback 不会匹配这些模式, 仍会显示.
#
# 平台: Linux 容器 / SSH / 无完整桌面 portal 的环境
# 警告: 仅"个人用 / 自己机"工具.  服务器/分发场景不要静默, 让用户看到所有问题.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 过滤模式 (扩展正则, ^ 锚定行首).  增删请保持 1 行 1 条, 必要时加尾部注释.
FILTERS=(
    '^dbus-daemon\['                # dbus-daemon 激活日志
    'xdg-desktop-portal.*WARNING'   # portal 找不到 backend
    'RealtimeKit'                   # PipeWire 调度器
    'PipeWire'                      # PipeWire 音频服务
    'ZINK'                          # MESA ZINK 驱动日志
    'max threads value'             # MESA max-threads-overflow
    '^Ignoring invalid max threads' # 同上, 另一条格式
    'pw\.conf'                      # PipeWire 配置加载失败
    'SpiRegistry daemon'            # GTK SPI 注册失败
    'dconf-WARNING'                 # dconf 提交失败 (容器/SSH 常见)
)

# 解析 TARGET: 相对路径先在 CWD 找, 再在 SCRIPT_DIR 找, 找不到报错退出.
resolve_target() {
    local t="$1"
    [[ -f "$t" ]] && { printf '%s\n' "$t"; return 0; }
    [[ -f "$SCRIPT_DIR/$t" ]] && { printf '%s\n' "$SCRIPT_DIR/$t"; return 0; }
    echo "找不到脚本: $t (CWD 与 $SCRIPT_DIR 都没有)" >&2
    return 1
}

TARGET="${1:?用法: $0 <script.py> [args...]}"
shift
TARGET="$(resolve_target "$TARGET")"

# 把 FILTERS 数组展开成 grep 的 -e 参数.
GREP_ARGS=( -v )
for pat in "${FILTERS[@]}"; do
    GREP_ARGS+=( -e "$pat" )
done

# pipefail: dbus-run-session 失败能正确返回非 0.
# line-buffered: grep 不缓冲, 输出实时显示.
# LC_ALL=C: 强制字节级匹配, 避免 locale 影响.
exec dbus-run-session python "$TARGET" "$@" 2>&1 \
    | LC_ALL=C grep --line-buffered "${GREP_ARGS[@]}"
