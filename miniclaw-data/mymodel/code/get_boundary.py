import os
import cv2
import numpy as np
from tqdm import tqdm


def generate_boundaries_from_masks(mask_dir, save_dir, edge_thickness=3):
    """
    从掩码图像批量生成边界图
    :param mask_dir: 原始掩码文件夹路径
    :param save_dir: 边界图保存文件夹路径
    :param edge_thickness: 边界的像素宽度 (建议 2-5 之间)
    """
    print(f"正在读取掩码路径: {mask_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有图片文件
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(valid_ext)]

    if len(mask_files) == 0:
        print("错误：掩码文件夹中没有找到图片！")
        return

    # 生成一个用于形态学操作的核（决定了线条的粗细）
    kernel = np.ones((edge_thickness, edge_thickness), np.uint8)

    print(f"共找到 {len(mask_files)} 张掩码，开始提取边界...")

    for filename in tqdm(mask_files, desc="生成进度"):
        mask_path = os.path.join(mask_dir, filename)
        save_path = os.path.join(save_dir, filename)

        # 1. 以灰度模式读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 2. 严格二值化 (确保只有 0 和 255，去除可能的抗锯齿灰边)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 3. 提取边界：使用形态学梯度 (膨胀图 减去 腐蚀图)
        # 这会生成一条跨越掩码内外边缘的边界线
        boundary = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)

        # 4. 保存边界图 (文件名与原图保持完全一致)
        cv2.imwrite(save_path, boundary)

    print(f"\n生成完毕！所有边界图已保存至: {save_dir}")


if __name__ == "__main__":
    # ================= 配置区 =================
    # 1. 你的原始掩码文件夹
    MASK_DIR = r"/hdd2/Bo Yu/data/HL/test/mask/"

    # 2. 你想把生成的边界图存在哪里 (如果不存在会自动创建)
    SAVE_BOUNDARY_DIR = r"/home/yubo/Bsinet_yb/Mymodel/boundary_data/HL/"

    # 3. 边界线条的粗细 (核心参数！)
    THICKNESS = 2
    # ==========================================

    generate_boundaries_from_masks(MASK_DIR, SAVE_BOUNDARY_DIR, edge_thickness=THICKNESS)