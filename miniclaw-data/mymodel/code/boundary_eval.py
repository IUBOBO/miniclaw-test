import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


# ==========================================
# 1. 核心评估类 (保持不变)
# ==========================================
class BoundaryEvaluator:
    def __init__(self, tolerance=2.0):
        self.tolerance = tolerance

    def _get_distances(self, pred_boundary, gt_boundary):
        if np.sum(pred_boundary) == 0 or np.sum(gt_boundary) == 0:
            return None, None
        dist_to_gt = distance_transform_edt(gt_boundary == 0)
        dist_to_pred = distance_transform_edt(pred_boundary == 0)
        distances_pred_to_gt = dist_to_gt[pred_boundary == 1]
        distances_gt_to_pred = dist_to_pred[gt_boundary == 1]
        return distances_pred_to_gt, distances_gt_to_pred

    def compute_boundary_f_measure(self, pred_boundary, gt_boundary):
        distances_pred_to_gt, distances_gt_to_pred = self._get_distances(pred_boundary, gt_boundary)
        if distances_pred_to_gt is None:
            return 0.0, 0.0, 0.0
        tp_pred = np.sum(distances_pred_to_gt <= self.tolerance)
        precision = tp_pred / len(distances_pred_to_gt) if len(distances_pred_to_gt) > 0 else 0.0
        tp_gt = np.sum(distances_gt_to_pred <= self.tolerance)
        recall = tp_gt / len(distances_gt_to_pred) if len(distances_gt_to_pred) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1_score

    def compute_hd95_and_assd(self, pred_boundary, gt_boundary):
        distances_pred_to_gt, distances_gt_to_pred = self._get_distances(pred_boundary, gt_boundary)
        if distances_pred_to_gt is None:
            max_dist = np.sqrt(pred_boundary.shape[0] ** 2 + pred_boundary.shape[1] ** 2)
            return max_dist, max_dist
        hd95 = max(np.percentile(distances_pred_to_gt, 95), np.percentile(distances_gt_to_pred, 95))
        assd = (np.mean(distances_pred_to_gt) + np.mean(distances_gt_to_pred)) / 2.0
        return hd95, assd


# ==========================================
# 2. 批量处理函数
# ==========================================
def evaluate_folder(pred_dir, gt_dir, tolerance=2.0):
    print(f"正在评估文件夹: \n预测路径: {pred_dir}\n真实路径: {gt_dir}")

    # 实例化评估器
    evaluator = BoundaryEvaluator(tolerance=tolerance)

    # 获取预测文件夹中的所有图片
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(valid_ext)]

    if len(pred_files) == 0:
        print("错误：预测文件夹中没有找到图片！")
        return

    # 初始化累计变量
    total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0
    total_hd95, total_assd = 0.0, 0.0
    valid_image_count = 0
    missing_files = []

    # 遍历计算
    for img_name in tqdm(pred_files, desc="计算指标中"):
        pred_path = os.path.join(pred_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name)

        # 检查真实标签中是否存在同名文件
        if not os.path.exists(gt_path):
            missing_files.append(img_name)
            continue

        # 读取为灰度图
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # 确保尺寸一致
        if pred_img.shape != gt_img.shape:
            # 如果尺寸不一致，将预测图 resize 到和 gt 一样大（或者报错）
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 二值化 (0 和 1)
        pred_binary = (pred_img > 127).astype(np.uint8)
        gt_binary = (gt_img > 127).astype(np.uint8)

        # 计算当前图像的指标
        p, r, f1 = evaluator.compute_boundary_f_measure(pred_binary, gt_binary)
        hd95, assd = evaluator.compute_hd95_and_assd(pred_binary, gt_binary)

        # 累加
        total_precision += p
        total_recall += r
        total_f1 += f1
        total_hd95 += hd95
        total_assd += assd
        valid_image_count += 1

    # 报告缺失文件
    if len(missing_files) > 0:
        print(f"\n警告：共有 {len(missing_files)} 张预测图在真实标签文件夹中找不到同名文件，已跳过。")

    # ==========================================
    # 3. 输出平均结果
    # ==========================================
    if valid_image_count == 0:
        print("错误：没有成功匹配并计算的图像对！")
        return

    avg_precision = total_precision / valid_image_count
    avg_recall = total_recall / valid_image_count
    avg_f1 = total_f1 / valid_image_count
    avg_hd95 = total_hd95 / valid_image_count
    avg_assd = total_assd / valid_image_count
    print("\n" + "=" * 40)
    print(f"数据集总体评估结果(共{valid_image_count}张图像)")
    print(f"   像素容差设为: {tolerance} px")
    print("=" * 40)
    print(f"Boundary Precision: {avg_precision:.4f}")
    print(f"Boundary Recall:    {avg_recall:.4f}")
    print(f"Boundary F-measure: {avg_f1:.4f}   (越高越好，满分 1.0)")
    print("-" * 40)
    print(f"HD95 (95%豪斯多夫): {avg_hd95:.4f} px (越低越好)")
    print(f"ASSD (平均表面距离): {avg_assd:.4f} px (越低越好)")
    print("=" * 40)


if __name__ == "__main__":
    # ================= 配置区 =================

    # 1. 你刚用 batch_predict 生成的所有预测边界图所在的文件夹
    PRED_DIR = r"/home/yubo/Bsinet_yb/Mymodel/pred/boundary/HL/main64-hl-b-04/"

    # 2. 你的测试集真实标签(Ground Truth)边界图所在的文件夹
    # (注意文件名必须和预测图一一对应，例如都是 001.png)
    GT_DIR = r"/home/yubo/Bsinet_yb/Mymodel/boundary_data/HL/"

    # ==========================================

    evaluate_folder(PRED_DIR, GT_DIR, tolerance=2.0)
