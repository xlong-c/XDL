#!/bin/bash
# ============================================================
# 脚本: make_clonezilla_usb.sh
# 用途: 一键制作再生龙自动备份U盘
# 用法: sudo bash make_clonezilla_usb.sh /dev/sdX
# WARNING: 会清空目标U盘所有数据!
# ============================================================

set -euo pipefail

# ==================== 配置区 ====================
CLONEZILLA_ISO="clonezilla-live-3.1.3-16-amd64.iso"
CLONEZILLA_URL="https://dl.tw.org/drbl-live/stable/${CLONEZILLA_ISO}"
WORK_DIR="/tmp/clonezilla_usb_$$"

# ==================== 检查参数 ====================
if [ -z "${1:-}" ]; then
    echo "Usage: sudo bash $0 /dev/sdX"
    echo "example: sudo bash $0 /dev/sda"
    exit 1
fi

USB_DEV="$1"

if [ ! -b "$USB_DEV" ]; then
    echo "ERROR: $USB_DEV 不是有效的块设备"
    exit 1
fi

echo "=========================================="
echo " WARNING: 这将清空 ${USB_DEV} 上的所有数据!"
echo " 目标设备: ${USB_DEV}"
lsblk "$USB_DEV"
echo "=========================================="
read -rp "确认继续? (输入 YES 继续): " confirm
[ "$confirm" != "YES" ] && echo "已取消" && exit 0

# ==================== 下载 Clonezilla ====================
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -f "$CLONEZILLA_ISO" ]; then
    echo "[1/5] 下载 Clonezilla ISO..."
    wget -q --show-progress "$CLONEZILLA_URL" -O "$CLONEZILLA_ISO"
else
    echo "[1/5] ISO 已存在, 跳过下载"
fi

# ==================== 分区 ====================
echo "[2/5] 对 U 盘分区..."

# 卸载所有现有分区
for part in $(lsblk -ln -o NAME "$USB_DEV" | grep -v "^$(basename "$USB_DEV")$"); do
    umount "/dev/${part}" 2>/dev/null || true
done

# 创建 GPT 分区表 + 两个分区
sgdisk --zap-all "$USB_DEV"
sgdisk -n 1:0:+4G -t 1:EF00 "$USB_DEV"   # EFI 分区 4G
sgdisk -n 2:0:0   -t 2:8300 "$USB_DEV"   # 数据分区 (剩余空间)

# 等待内核刷新分区表
sleep 2
partprobe "$USB_DEV" 2>/dev/null || true
sleep 1

# 确定分区名 (处理 /dev/sda -> /dev/sda1, /dev/nvme0n1 -> /dev/nvme0n1p1)
if [[ "$USB_DEV" =~ nvme ]]; then
    PART1="${USB_DEV}p1"
    PART2="${USB_DEV}p2"
else
    PART1="${USB_DEV}1"
    PART2="${USB_DEV}2"
fi

# ==================== 格式化 ====================
echo "[3/5] 格式化分区..."
mkfs.vfat -F32 "$PART1"
fatlabel "$PART1" CLONEZILLA
mkfs.ext4 -F "$PART2"
e2label "$PART2" BACKUP_IMG

# ==================== 安装 Clonezilla 到 U盘 ====================
echo "[4/5] 安装 Clonezilla 到 U盘..."

MOUNT_EFI="/mnt/clonezilla_efi_$$"
mkdir -p "$MOUNT_EFI"
mount "$PART1" "$MOUNT_EFI"

# 挂载 ISO 并复制文件
mkdir -p /mnt/iso_$$
mount -o loop "$CLONEZILLA_ISO" /mnt/iso_$$
cp -a /mnt/iso_/* "$MOUNT_EFI/"
umount /mnt/iso_$$
rmdir /mnt/iso_$$

# ==================== 写入自动备份配置 ====================
echo "[5/5] 配置无人值守自动备份..."

# 修改 syslinux/grub 启动参数, 添加自动运行 ocs-custom 脚本
# 关键: ocs_prerun 指向我们的自动备份脚本
cat > "$MOUNT_EFI/syslinux/ocs-custom.conf" << 'SYSLINUX_EOF'
# 无人值守自动备份 - 启动后直接执行
label Clonezilla Auto Backup
  menu label Clonezilla: Auto Backup to USB
  kernel /live/vmlinuz
  append initrd=/live/initrd.img boot=live union=overlay username=user config components quiet nosplash ocs_prerun="/lib/live/mount/medium/auto_backup.sh" ocs_live_run="ocs-sr" ocs_live_extra_param="--batch -p choose -q2 -j2 -z1p -srel -scr" ocs_live_batch="yes" keyboard-layouts="us" locales="zh_CN.UTF-8" noprompt
SYSLINUX_EOF

# 同样的配置给 grub
cat > "$MOUNT_EFI/boot/grub/ocs-custom.cfg" << 'GRUB_EOF'
menuentry "Clonezilla: Auto Backup to USB" {
    search --set=root --fs-uuid __UUID__
    linux /live/vmlinuz boot=live union=overlay username=user config components quiet nosplash ocs_prerun="/lib/live/mount/medium/auto_backup.sh" ocs_live_run="ocs-sr" ocs_live_extra_param="--batch -p choose -q2 -j2 -z1p -srel -scr" ocs_live_batch="yes" keyboard-layouts="us" locales="zh_CN.UTF-8" noprompt
    initrd /live/initrd.img
}
GRUB_EOF

# ==================== 核心: 自动备份脚本 ====================
cat > "$MOUNT_EFI/auto_backup.sh" << 'AUTOBAK_EOF'
#!/bin/bash
# ============================================================
# Clonezilla 无人值守自动备份脚本
# 此脚本在 Clonezilla Live 环境中运行
# ============================================================

set -e

LOG="/var/log/auto_backup.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo " Clonezilla 自动备份开始"
echo " 时间: $(date)"
echo "=========================================="

# ==================== 配置区 ====================
# 备份目标分区 (U盘第二个分区, ext4 格式)
BACKUP_PART="/dev/disk/by-label/BACKUP_IMG"

# 要备份的源盘 (通常是系统盘, 根据实际情况修改)
# 常见: /dev/nvme0n1 (NVMe), /dev/sda (SATA)
SOURCE_DISK="/dev/nvme0n1"

# 备份镜像名称 (会自动加日期)
BACKUP_NAME="system_backup_$(date +%Y%m%d_%H%M%S)"

# 压缩方式: -z1p (pigz并行gzip), -z3p (pigz parallel zstd), -z0 (不压缩)
COMPRESS="-z3p"

# ==================== 检测可用磁盘 ====================
echo ""
echo "--- 系统磁盘信息 ---"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,LABEL
echo ""

# ==================== 挂载备份目标分区 ====================
echo "--- 挂载备份分区 ---"
mkdir -p /home/partimag
mount "$BACKUP_PART" /home/partimag || {
    echo "ERROR: 无法挂载 $BACKUP_PART, 尝试手动查找..."
    # 如果 label 找不到, 尝试按大小查找
    blkid | grep -E "ext4|ntfs" || true
    exit 1
}

# 检查空间
available=$(df -BG /home/partimag | awk 'NR==2 {print $4}' | sed 's/G//')
echo "备份目标可用空间: ${available}G"

# ==================== 执行备份 ====================
echo ""
echo "--- 开始备份 $SOURCE_DISK ---"
echo "镜像名称: $BACKUP_NAME"
echo "压缩方式: $COMPRESS"

# ocs-sr: Clonezilla 的命令行保存/恢复工具
# -q2: 使用 partclone (推荐, 只备份已用块)
# -j2: 克隆前检查文件系统并修复
# -srel: 不等待确认, 自动开始
# -scr: 修复后继续 (即使修复失败)
# -p choose: 自动选择"保存磁盘"模式
ocs-sr \
    --batch \
    -p choose \
    -q2 \
    -j2 \
    "$COMPRESS" \
    -srel \
    -scr \
    savedisk \
    "$BACKUP_NAME" \
    "$SOURCE_DISK"

# ==================== 清理 ====================
echo ""
echo "--- 备份完成 ---"
echo "镜像位置: /home/partimag/${BACKUP_NAME}/"
ls -lh /home/partimag/"$BACKUP_NAME"/

umount /home/partimag

echo ""
echo "=========================================="
echo " 备份完成! 时间: $(date)"
echo " 系统将在 10 秒后关机"
echo "=========================================="

# 完成后自动关机 (去掉下面注释以启用)
# sleep 10 && poweroff
AUTOBAK_EOF

chmod +x "$MOUNT_EFI/auto_backup.sh"

# ==================== 清理 ====================
umount "$MOUNT_EFI"
rmdir "$MOUNT_EFI"

echo ""
echo "=========================================="
echo " 制作完成!"
echo " U盘分区: ${PART1} (Clonezilla Live)"
echo " 备份分区: ${PART2} (镜像存储)"
echo ""
echo " 使用方法:"
echo " 1. 插入U盘, 从U盘启动"
echo " 2. 选择 'Clonezilla: Auto Backup to USB'"
echo " 3. 自动开始备份, 无需人工干预"
echo ""
echo " 如需恢复: 启动后选择 Clonezilla 交互模式,"
echo "           使用 ocs-sr restoredisk 命令"
echo "=========================================="

# 清理临时文件
rm -rf "$WORK_DIR"
