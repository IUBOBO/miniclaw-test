import torch
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm

# 导入你使用的网络模型类
# from model import BaselineSharedDecoderNet as MyModel
from model import Tripmodel as MyModel


def preprocess_image(img_path):
    """
    图像预处理，与训练时保持一致
    """
    img = Image.open(img_path)
    img_nd = np.array(img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    img_trans = img_trans / 255.0

    img_tensor = torch.from_numpy(img_trans).float().unsqueeze(0)
    return img_tensor


def batch_predict(model_path, test_dir, save_dir, device='cuda', threshold=0.5):
    """
    遍历文件夹，批量预测边界图
    """
    print(f"正在加载模型权重: {model_path}")

    # 1. 初始化模型并加载权重 (在循环外只执行一次)
    net = MyModel(input_channels=3, num_classes=1)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()  # 设置为评估模式

    # 2. 准备输入和输出目录
    os.makedirs(save_dir, exist_ok=True)

    # 获取测试文件夹下所有的图片文件 (过滤掉可能存在的隐藏文件如 .DS_Store)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_extensions)]

    if len(image_files) == 0:
        print(f"在 {test_dir} 中没有找到图片文件！请检查路径。")
        return

    print(f"共找到 {len(image_files)} 张图片，开始批量预测...")

    # 3. 遍历图片进行预测
    with torch.no_grad():  # 整个推理过程都不需要梯度
        for img_name in tqdm(image_files, desc="预测进度"):
            # 拼接完整的输入和输出路径
            img_path = os.path.join(test_dir, img_name)
            save_path = os.path.join(save_dir, img_name)  # 输出名与输入名保持一致

            # 预处理
            img_tensor = preprocess_image(img_path).to(device)

            # 前向推理
            pred_mask, pred_boundary = net(img_tensor)

            # 后处理边界图
            prob_boundary = torch.sigmoid(pred_boundary)
            binary_boundary = (prob_boundary > threshold).float()

            # 转换为 NumPy 并放大到 0-255
            boundary_np = binary_boundary.squeeze().cpu().numpy()
            boundary_img_cv = (boundary_np * 255.0).astype(np.uint8)

            # 保存图片
            cv2.imwrite(save_path, boundary_img_cv)

    print(f"\n批量预测完成！所有结果已保存至: {save_dir}")


if __name__ == '__main__':
    # ================= 配置区 =================

    # 1. 你保存的 .pth 权重文件路径
    MODEL_WEIGHTS = r"/home/yubo/Bsinet_yb/Mymodel/Mloss/hl/main64/epoch100.pth"

    # 2. 包含所有测试图片的文件夹路径
    TEST_DIR = r"/hdd2/Bo Yu/data/HL/test/img/"

    # 3. 预测结果批量保存的文件夹路径 (如果不存在会自动创建)
    SAVE_DIR = r"/home/yubo/Bsinet_yb/Mymodel/pred/boundary/HL/main64-hl-b-03-new"

    # 设定设备和二值化阈值
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    THRESHOLD = 0.3
    # ==========================================

    batch_predict(MODEL_WEIGHTS, TEST_DIR, SAVE_DIR, device=DEVICE, threshold=THRESHOLD)