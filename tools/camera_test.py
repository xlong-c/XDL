import cv2
import sys

# 尝试导入matplotlib,如果失败则标记为不可用
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("注意: matplotlib不可用,将只使用OpenCV显示")

def set_camera_resolution(cap, width, height):
    """设置摄像头分辨率"""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 检查设置是否成功
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if actual_width != width or actual_height != height:
        print(f"警告: 无法设置分辨率 {width}x{height},实际分辨率: {int(actual_width)}x{int(actual_height)}")
        return False
    else:
        print(f"成功设置分辨率: {int(width)}x{int(height)}")
        return True

def get_camera_property(cap, prop_id, prop_name):
    """安全地获取摄像头属性"""
    try:
        value = cap.get(prop_id)
        return value if value != -1 else "N/A"
    except:
        return "N/A"

def detect_supported_resolutions(cap):
    """检测摄像头支持的分辨率"""
    common_resolutions = [
        (320, 240),   # QVGA
        (640, 480),   # VGA
        (800, 600),   # SVGA
        (1024, 768),  # XGA
        (1280, 720),  # HD
        (1280, 960),  # 1.3MP
        (1920, 1080), # FHD
        (2560, 1440), # QHD
    ]
    
    supported_resolutions = []
    print("检测摄像头支持的分辨率...")
    
    # 保存当前设置
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    for width, height in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width == width and actual_height == height:
            supported_resolutions.append((width, height))
            print(f"  [OK] {width}x{height}")
        else:
            # 有些摄像头可能只近似支持某些分辨率
            if abs(actual_width - width) < 50 and abs(actual_height - height) < 50:
                supported_resolutions.append((actual_width, actual_height))
                print(f"  [~] {width}x{height} (实际: {actual_width}x{actual_height})")
    
    # 恢复原始设置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
    
    return supported_resolutions

def test_camera(resolution=None, auto_detect=False):
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return
    
    # 如果需要自动检测分辨率
    if auto_detect:
        supported_resolutions = detect_supported_resolutions(cap)
        if supported_resolutions:
            # 选择最高分辨率
            best_resolution = max(supported_resolutions, key=lambda res: res[0] * res[1])
            print(f"选择最佳分辨率: {best_resolution[0]}x{best_resolution[1]}")
            set_camera_resolution(cap, best_resolution[0], best_resolution[1])
        else:
            print("未检测到支持的分辨率")
    # 如果指定了分辨率,则尝试设置
    elif resolution:
        width, height = resolution
        print(f"尝试设置摄像头分辨率为: {width}x{height}")
        set_camera_resolution(cap, width, height)
    else:
        # 尝试设置默认的高清分辨率
        print("尝试设置默认分辨率为: 1280x720")
        set_camera_resolution(cap, 1280, 720)
    
    # 获取并打印摄像头信息
    print("摄像头信息:")
    print(f"  - 后端API: {cap.getBackendName()}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  - 分辨率: {width} x {height}")
    print(f"  - 帧率: {get_camera_property(cap, cv2.CAP_PROP_FPS, '帧率')}")
    print(f"  - 编码格式: {get_camera_property(cap, cv2.CAP_PROP_FOURCC, '编码格式')}")
    print(f"  - 亮度: {get_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, '亮度')}")
    print(f"  - 对比度: {get_camera_property(cap, cv2.CAP_PROP_CONTRAST, '对比度')}")
    print(f"  - 饱和度: {get_camera_property(cap, cv2.CAP_PROP_SATURATION, '饱和度')}")
    print(f"  - 色调: {get_camera_property(cap, cv2.CAP_PROP_HUE, '色调')}")
    print(f"  - 增益: {get_camera_property(cap, cv2.CAP_PROP_GAIN, '增益')}")
    print(f"  - 曝光: {get_camera_property(cap, cv2.CAP_PROP_EXPOSURE, '曝光')}")
    print("-" * 40)
    
    # 尝试使用OpenCV显示(如果支持GUI)
    try:
        print("正在尝试使用OpenCV显示...")
        opencv_display(cap)
    except cv2.error as e:
        print("OpenCV GUI不支持,切换到matplotlib显示...")
        matplotlib_display(cap)
    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        # 释放摄像头
        cap.release()

def opencv_display(cap):
    """使用OpenCV显示视频流"""
    print("按 'q' 键退出预览")
    while True:
        # 读取帧
        ret, frame = cap.read()
        
        # 检查是否成功读取帧
        if not ret:
            print("无法接收帧(流结束？)")
            break
            
        # 获取当前帧的实际尺寸
        height, width = frame.shape[:2]
        print(f"当前帧尺寸: {width} x {height}", end='\r')
            
        # 显示帧
        cv2.imshow('Camera Preview', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 关闭窗口
    cv2.destroyAllWindows()

def matplotlib_display(cap):
    """使用matplotlib显示视频流"""
    if not MATPLOTLIB_AVAILABLE:
        print("错误: matplotlib不可用,无法使用此显示方式")
        return
        
    print("使用matplotlib显示视频流...")
    print("关闭matplotlib窗口或按Ctrl+C退出")
    
    # 设置matplotlib
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    ax.set_title('Camera Preview')
    ax.axis('off')
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧")
        return
    
    # 转换颜色格式(BGR to RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 显示图像
    im = ax.imshow(frame_rgb)
    
    def update_frame(frame_num):
        ret, frame = cap.read()
        if ret:
            # 获取当前帧的实际尺寸
            height, width = frame.shape[:2]
            print(f"当前帧尺寸: {width} x {height}", end='\r')
            
            # 转换颜色格式(BGR to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im.set_array(frame_rgb)
        return [im]
    
    # 创建动画
    ani = FuncAnimation(fig, update_frame, interval=50, blit=True, cache_frame_data=False)
    
    # 显示窗口
    plt.tight_layout()
    plt.show()
    
    # 等待窗口关闭
    try:
        plt.pause(0.1)  # 初始暂停
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)

if __name__ == "__main__":
    # 支持的分辨率选项
    resolutions = {
        'hd': (1280, 720),
        'fhd': (1920, 1080),
        'vga': (640, 480),
        'qvga': (320, 240)
    }
    
    # 解析命令行参数
    resolution = None
    auto_detect = False
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == '--auto' or arg == '-a':
            auto_detect = True
            print("启用自动检测最佳分辨率模式")
        elif arg in resolutions:
            resolution = resolutions[arg]
            print(f"使用指定分辨率: {arg} ({resolution[0]}x{resolution[1]})")
        else:
            print(f"未知参数: {arg}")
            print("使用方法:")
            print("  python camera_test.py           # 使用默认分辨率")
            print("  python camera_test.py hd        # 使用HD分辨率 (1280x720)")
            print("  python camera_test.py fhd       # 使用FHD分辨率 (1920x1080)")
            print("  python camera_test.py vga       # 使用VGA分辨率 (640x480)")
            print("  python camera_test.py qvga      # 使用QVGA分辨率 (320x240)")
            print("  python camera_test.py --auto    # 自动检测最佳分辨率")
            sys.exit(1)
    
    test_camera(resolution, auto_detect)