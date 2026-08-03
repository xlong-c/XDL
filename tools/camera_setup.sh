#!/usr/bin/env bash
# ============================================================
# 一键设置 USB 摄像头参数 (v4l2-ctl)
#
# 用法:
#   bash tools/camera_setup.sh                 # 默认 /dev/video0
#   DEVICE=/dev/video1 bash tools/camera_setup.sh
# ============================================================
set -euo pipefail

DEVICE="${DEVICE:-/dev/video0}"

# ---------- 参数配置（修改这里即可）----------
# 格式: [参数名]=值
# bool 类型: 1=true, 0=false
# menu 类型: 填序号
declare -A CONTROLS=(
  # === User Controls ===
  [brightness]=32
  [contrast]=32
  [saturation]=32
  [hue]=32
  [white_balance_automatic]=1
  [gamma]=30
  [gain]=0
  [power_line_frequency]=1      # 0=Disabled 1=50Hz 2=60Hz
  [white_balance_temperature]=5000
  [sharpness]=32
  [backlight_compensation]=1

  # === Camera Controls ===
  [auto_exposure]=1             # 0=Auto 1=Manual 2=Shutter Priority 3=Aperture Priority
  [exposure_time_absolute]=280
  [pan_absolute]=0
  [tilt_absolute]=0
  [focus_absolute]=512
  [focus_automatic_continuous]=1
  [zoom_absolute]=0
)
# -------------------------------------------------

echo "[INFO] 设备: $DEVICE"
echo "[INFO] 共 ${#CONTROLS[@]} 个参数待设置"

# 构建参数列表
CTL_ARGS=()
for name in "${!CONTROLS[@]}"; do
  CTL_ARGS+=(-c "${name}=${CONTROLS[$name]}")
done

# 执行
if v4l2-ctl -d "$DEVICE" "${CTL_ARGS[@]}"; then
  echo "[OK] 摄像头参数设置完成。"
else
  echo "[ERROR] v4l2-ctl 执行失败 (退出码: $?)" >&2
  exit 1
fi
