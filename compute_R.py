import cv2
import numpy as np
from scipy.optimize import curve_fit

# ==================== 固定参数（写死） ====================
WAVELENGTH = 589.3e-6  # 钠光波长 589.3 nm → 0.0005893 mm
PIXEL_TO_MM = 0.005    # 假设 1 像素 = 0.005 mm（需要你实际标定）
RINGS_TO_USE = [5, 7, 9, 11, 13]  # 使用第5,7,9,11,13级暗环

# ==================== 函数：从图像找圆心 ====================
def find_center(img_gray):
    """
    用边缘检测 + 最小外接圆拟合找圆心。
    假设牛顿环大致在图像中央，且是类圆形。
    """
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # Canny边缘检测
    edges = cv2.Canny(blurred, 30, 100)
    # 找到所有轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到任何轮廓，请检查输入图像。")
    # 取最大轮廓（假设牛顿环是主要物体）
    largest_contour = max(contours, key=cv2.contourArea)
    # 用最小外接圆拟合轮廓，圆心就是外接圆圆心
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    return (int(x), int(y))

# ==================== 函数：沿径向提取强度曲线 ====================
def extract_radial_intensity(img_gray, center, max_radius=None):
    """
    从圆心出发，沿半径方向采样灰度值，取圆周平均。
    返回：半径数组（像素）、对应的平均灰度数组。
    """
    h, w = img_gray.shape
    if max_radius is None:
        max_radius = int(min(h, w) / 2)
    
    radii = np.arange(1, max_radius)
    intensities = []
    for r in radii:
        # 创建圆形掩膜
        mask = np.zeros_like(img_gray, dtype=np.uint8)
        cv2.circle(mask, center, r, 255, 1)
        # 取圆周上所有像素的平均灰度
        pixels = img_gray[mask == 255]
        if len(pixels) > 0:
            intensities.append(np.mean(pixels))
        else:
            intensities.append(0)
    return radii, np.array(intensities)

# ==================== 函数：找到暗环位置 ====================
def find_dark_rings(radii, intensity):
    """
    在径向强度曲线上找局部极小值，对应暗环。
    返回：暗环半径列表（像素）。
    """
    from scipy.signal import argrelextrema
    # 找局部极小值
    min_indices = argrelextrema(intensity, np.less)[0]
    dark_radii = radii[min_indices]
    return dark_radii

# ==================== 函数：计算曲率半径 R ====================
def compute_R(dark_radii_pixel, rings_to_use):
    """
    根据指定级次的暗环半径（像素），用逐差法计算 R（mm）。
    rings_to_use：使用的暗环级次列表，比如 [5,7,9,11,13]。
    """
    if len(dark_radii_pixel) < max(rings_to_use) + 1:
        raise ValueError(f"检测到的暗环数量不足，需要至少 {max(rings_to_use)+1} 个环。")
    
    # 取对应级次的半径（像素）
    radii_pixel = [dark_radii_pixel[k] for k in rings_to_use]
    # 转换为物理半径（mm）
    radii_mm = [r * PIXEL_TO_MM for r in radii_pixel]
    # 直径（mm）
    diameters_mm = [2 * r for r in radii_mm]
    
    # 用逐差法计算 R
    # 使用 m - n = 2 的组合（比如 11与5，13与7），也可以自定义
    m_idx = len(rings_to_use) - 1
    n_idx = 0
    Dm = diameters_mm[m_idx]
    Dn = diameters_mm[n_idx]
    m = rings_to_use[m_idx]
    n = rings_to_use[n_idx]
    
    R = (Dm**2 - Dn**2) / (4 * (m - n) * WAVELENGTH)
    return R

# ==================== 主函数：计算一张图像的 R ====================
def compute_R_from_image(image_path):
    """
    输入：牛顿环裁剪图像的路径。
    输出：曲率半径 R（mm）。
    """
    # 读取图像并转为灰度
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 找圆心
    center = find_center(gray)
    print(f"检测到圆心坐标：{center}")
    
    # 提取径向强度
    max_r = int(min(gray.shape) / 2)
    radii, intensity = extract_radial_intensity(gray, center, max_radius=max_r)
    
    # 找暗环半径
    dark_radii = find_dark_rings(radii, intensity)
    print(f"检测到 {len(dark_radii)} 个暗环，半径（像素）：{dark_radii[:10]}...")
    
    # 计算 R
    R = compute_R(dark_radii, RINGS_TO_USE)
    return R

# ==================== 测试入口 ====================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：python3 compute_R.py <牛顿环裁剪图片路径>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    try:
        R_value = compute_R_from_image(img_path)
        print(f"曲率半径 R = {R_value:.2f} mm")
    except Exception as e:
        print(f"计算失败：{e}")
