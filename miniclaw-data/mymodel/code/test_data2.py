import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, binary_dilation
from tqdm import tqdm


# ----------------------------
# Perlin 噪声辅助函数（无外部依赖）
# ----------------------------

def generate_perlin_noise_2d(shape, res):
    """生成二维 Perlin 噪声（归一化到 [0,1]）"""

    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    noise = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
    return (noise - noise.min()) / (noise.max() - noise.min())


def add_cloud_occlusion(image_float,
                        cloud_coverage=0.3,  # 云覆盖率（0~1）
                        cloud_opacity=0.7,  # 云不透明度（0.5~0.9）
                        num_clouds=8,  # 云团数量
                        min_size=40,  # 最小云团半径
                        max_size=150,  # 最大云团半径
                        dtype_max=255.0):
    """
    在遥感图像上添加逼真的半透明云覆盖

    参数:
        image_float: (H, W, C) float32, [0, 255]
        cloud_coverage: 目标云覆盖比例（如 0.3 = 30% 区域被云影响）
        cloud_opacity: 云的不透明度（0.5=薄云，0.9=厚云）
        num_clouds: 初始云团数量（越多越分散）
        min_size/max_size: 控制云团尺度
    """
    h, w = image_float.shape[:2]
    cloud_mask = np.zeros((h, w), dtype=np.float32)

    # Step 1: 随机生成多个圆形云核
    for _ in range(num_clouds):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        radius = np.random.uniform(min_size, max_size)

        # 创建圆形距离场
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # 高斯型衰减（模拟云边缘模糊）
        cloud_blob = np.exp(-dist ** 2 / (2 * (radius / 2) ** 2))
        cloud_mask = np.maximum(cloud_mask, cloud_blob)

    # Step 2: 用 Perlin 噪声扰动云边界（更自然）
    perlin = generate_perlin_noise_2d((h, w), (4, 4))  # 低频噪声
    cloud_mask = np.clip(cloud_mask + 0.3 * (perlin - 0.5), 0, 1)

    # Step 3: 调整云量至目标覆盖率
    current_coverage = np.mean(cloud_mask > 0.1)
    if current_coverage > 0:
        scale_factor = cloud_coverage / current_coverage
        cloud_mask = np.clip(cloud_mask * scale_factor, 0, 1)

    # Step 4: 应用最终不透明度（0.5~0.9）
    cloud_alpha = cloud_mask * cloud_opacity

    # Step 5: 云的颜色 —— 使用白色偏亮（符合真实云）
    cloud_color = np.full_like(image_float, dtype_max * 0.95)  # 略低于纯白

    # 混合：I = (1 - α) * I_original + α * I_cloud
    alpha_3ch = np.expand_dims(cloud_alpha, axis=-1)
    cloudy = image_float * (1 - alpha_3ch) + cloud_color * alpha_3ch

    return cloudy


def save_tif(image_float, profile, out_path):
    image_uint = np.clip(image_float, 0, 255).astype(profile['dtype'])
    image_chw = np.transpose(image_uint, (2, 0, 1))
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(image_chw)


# ====== 修改你的路径 ======
# input_dir = r"F:\Bo Yu\data\sichuan_jiangyou\test\img"
# output_dir = r"F:\Bo Yu\data\sichuan_jiangyou\test\add\train_image"
input_dir = r"F:\Bo Yu\data\HL\test\dadd\BlurFog"
output_dir = r"F:\Bo Yu\data\HL\test\dadd\BlurFogCloud"
# =========================

os.makedirs(output_dir, exist_ok=True)
tif_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]

for fname in tqdm(tif_files, desc="Cloud Occlusion"):
    with rasterio.open(os.path.join(input_dir, fname)) as src:
        profile = src.profile.copy()
        img = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)

    # 添加云覆盖（可调参数）
    cloudy = add_cloud_occlusion(
        img,
        cloud_coverage=0.40,  # 25% 区域被云影响
        cloud_opacity=0.75,  # 中等厚度云
        num_clouds=8,
        min_size=30,
        max_size=120
    )

    save_tif(cloudy, profile, os.path.join(output_dir, fname))

print("✅ 云覆盖模拟完成！")