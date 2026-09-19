# import os
# from PIL import Image
# import numpy as np
# from noise import pnoise2
#
# def add_haze_with_noise(image, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, A=0.9):
#     hazy_image = np.zeros_like(image)
#     rows, cols, _ = image.shape
#
#     for y in range(rows):
#         for x in range(cols):
#             n = pnoise2(y / scale, x / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity,
#                         repeatx=rows, repeaty=cols, base=0)
#             t = 0.6 + (n * 0.2)  # 控制雾浓度
#             hazy_image[y, x, :] = image[y, x, :] * t + A * (1 - t)
#
#     return np.clip(hazy_image, 0, 1.0)
#
# def process_and_save(input_path, output_path):
#     img = Image.open(input_path).convert('RGB')
#     image = np.array(img).astype(np.float32) / 255.0
#
#     hazy_image = add_haze_with_noise(image, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, A=0.95)
#     hazy_uint8 = (hazy_image * 255).astype(np.uint8)
#     hazy_img = Image.fromarray(hazy_uint8, mode='RGB')
#
#     hazy_img.save(output_path, format='TIFF')
#
# def batch_process_images(input_dir, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.tif'):
#             input_path = os.path.join(input_dir, filename)
#             output_path = os.path.join(output_dir, filename)
#
#             print(f"Processing {filename}...")
#             process_and_save(input_path, output_path)
#
# # 示例调用
# # input_directory = 'F:/Bo Yu/data/Xinjiang/test/img'  # 替换为你的输入文件夹路径
# # output_directory = 'F:/Bo Yu/data/Xinjiang/test/img_noise'  # 替换为你想要保存结果的文件夹路径
# # input_directory = 'F:/Bo Yu/data/HL/train/img'  # 替换为你的输入文件夹路径
# # output_directory = 'F:/Bo Yu/data/HL/train/img_noise'  # 替换为你想要保存结果的文件夹路径
# input_directory = 'F:\Bo Yu\data\sichuan_jiangyou\\train\image'  # 替换为你的输入文件夹路径
# output_directory = 'F:\Bo Yu\data\sichuan_jiangyou\\train\image_noise'  # 替换为你想要保存结果的文件夹路径
# batch_process_images(input_directory, output_directory)


import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# ==============================
# 修正版退化函数（全部内部使用 float 计算）
# ==============================

def add_blur(image_float, w=5):
    """输入应为 float32 [0, 255]"""
    sigma = (w - 1) / 6.0
    blurred = np.stack([
        gaussian_filter(image_float[:, :, i], sigma=sigma)
        for i in range(image_float.shape[2])
    ], axis=-1)
    return blurred


def add_gaussian_noise(image_float, sigma=20.0):
    """输入应为 float32 [0, 255]"""
    noise = np.random.normal(0, sigma, image_float.shape)
    return image_float + noise


def add_stripe_noise(image_float, r=0.2, direction='horizontal', dtype_max=255.0):
    """输入应为 float32 [0, 255]"""
    noisy = image_float.copy()
    h, w = image_float.shape[:2]
    max_offset = r * dtype_max

    if direction == 'horizontal':
        for i in range(h):
            if np.random.rand() < 0.1:
                offset = np.random.uniform(-max_offset, max_offset)
                noisy[i, :] += offset  # 现在是 float + float，安全！
    elif direction == 'vertical':
        for j in range(w):
            if np.random.rand() < 0.1:
                offset = np.random.uniform(-max_offset, max_offset)
                noisy[:, j] += offset
    return noisy


def add_fog(image_float, A=0.9, t0=0.6, dtype_max=255.0):
    """输入应为 float32 [0, 255]"""
    A_abs = A * dtype_max
    return image_float * t0 + A_abs * (1 - t0)


def save_tif(image_float, profile, out_path):
    """将 float 图像裁剪并转回原始 dtype（如 uint8）后保存"""
    # 裁剪到合法范围并转为原始数据类型
    image_uint = np.clip(image_float, 0, 255).astype(profile['dtype'])
    image_chw = np.transpose(image_uint, (2, 0, 1))  # (C, H, W)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(image_chw)


# ==============================
# 主程序
# ==============================

# 🔧 设置路径（仅需改这两行）
input_dir = r"F:\Bo Yu\data\HL\test\dadd\BlurCloud"
output_base_dir = r"F:\Bo Yu\data\HL\test\dadd\BlurCloudStripe"

for sub in ['train_mask', 'train_image', 'train_image', 'train_image']:
    os.makedirs(os.path.join(output_base_dir, sub), exist_ok=True)

tif_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]

print(f"共 {len(tif_files)} 张图像待处理...")

for fname in tqdm(tif_files, desc="处理图像"):
    input_path = os.path.join(input_dir, fname)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        img_data = src.read()  # (C, H, W), dtype=uint8
        img_hwc = np.transpose(img_data, (1, 2, 0))  # (H, W, C)

    # ✅ 关键步骤：转换为 float32 进行所有运算
    img_float = img_hwc.astype(np.float32)

    # 应用四种退化（全部在 float 上操作）
    blurred = add_blur(img_float, w=5)
    noisy = add_gaussian_noise(img_float, sigma=20.0)
    striped = add_stripe_noise(img_float, r=0.2, direction='horizontal')
    hazy = add_fog(img_float, A=0.9, t0=0.6)
    # blurred = add_blur(img_float, w=6)
    # noisy = add_gaussian_noise(img_float, sigma=25.0)
    # striped = add_stripe_noise(img_float, r=0.4, direction='horizontal')
    # hazy = add_fog(img_float, A=0.8, t0=0.5)

    # 保存（内部会转回 uint8）
    # save_tif(blurred, profile, os.path.join(output_base_dir, 'train_mask', fname))
    # save_tif(noisy, profile, os.path.join(output_base_dir, 'train_image', fname))
    save_tif(striped, profile, os.path.join(output_base_dir, 'train_image', fname))
    # save_tif(hazy, profile, os.path.join(output_base_dir, 'train_image', fname))

print("✅ 处理完成！")
