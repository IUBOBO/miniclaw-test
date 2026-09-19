import torch
from PIL import Image
import numpy as np
import os


def compute_iou(pred_image, target_image):

    # 将图像转换为张量
    pred = torch.from_numpy(np.array(pred_image)).float()
    target = torch.from_numpy(np.array(target_image)).float()

    # 将预测结果和真实标签转换为二值图像
    pred = pred >= 128# 这里假设使用了类似sigmoid的激活函数，并将预测结果阈值化为二值图像
    target = target > 0

    # 计算交集和并集
    intersection = torch.logical_and(pred, target).float().sum()
    union = torch.logical_or(pred, target).float().sum()
    # print(intersection,union-1e-7)
    # 计算IoU
    iou = intersection / (union+1e-10) # 添加一个小的常数，防止除零错误

    return iou.item()


def calculate_iou(folder1, folder2):
    total_iou = 0
    file_count = 0
    count = 0

    # 获取两个文件夹中的文件列表
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    for file1 in files1:
        if file1 in files2:
            file_count += 1
            file_path1 = os.path.join(folder1, file1)
            file_path2 = os.path.join(folder2, file1)

            # 加载预测结果图像和真实标签图像
            pred_image = Image.open(file_path1)
            target_image = Image.open(file_path2)

            # 调用compute_iou函数计算IoU
            iou = compute_iou(pred_image, target_image)
            if iou > 0:
                count += 1
            print(f"{file1}:{iou}")
            total_iou += iou

    # 计算平均IoU
    average_iou = total_iou / count

    return average_iou


# # 用于计算IoU的两个文件夹路径

# test_path = r'F:\Bo Yu\data\Xinjiang\test\mask'
# pred_path = r"F:\Bo Yu\data\Xinjiang\test\new7_plus"

test_path = r'F:\Bo Yu\data\Xinjiang\test\mask'
# test_path  = r'F:\Bo Yu\data\HL\test\mask'
# pred_path = r"F:\Bo Yu\Bsinet_yb\pred\TAFbgm6\xj\mask1_lss"
pred_path = r'F:\Bo Yu\Bsinet_yb\pred\TAFbgm4\xj\mask1_lss'
# pred_path = r"F:\Bo Yu\Bsinet_yb\pred\TAF13148\xj\mask2"
# pred_path = r'F:\Bo Yu\Net\deeplabv3\yb_test\pred\xj\mask1_lss'
# pred_path = r'F:\Bo Yu\Net\BsiNet\yb_test\pred\xj\mask1_lss'
# pred_path = r"F:\Bo Yu\Net\SEANet\yb_test\pred\xj\mask1_lss"
# pred_path = r"F:\Bo Yu\Net\REAUNet\yb_test\pred\xj\mask"
# pred_path = r"F:\Bo Yu\Net\HBGNet\yb_test\pred"
# 调用calculate_iou函数计算IoU
iou = calculate_iou(test_path, pred_path)

# 输出结果
print("Average IoU:", iou)
print(f'path: {pred_path}')
"""
if __name__ == '__main__':


    pred_image = Image.open("D:\\GIS\\train_test\\T_T\\UNet3plus_pth-master\\data\\true_musk\\clipped_13.tif")
    target_image = Image.open("D:\\GIS\\train_test\\T_T\\UNet3plus_pth-master\\data\\res_v1_100\\clipped_13.tif")

    iou = compute_iou(pred_image, target_image)

    # 输出结果
    print("IoU:", iou)
"""
