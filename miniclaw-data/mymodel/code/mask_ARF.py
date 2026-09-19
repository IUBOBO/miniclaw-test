import os
import cv2
import numpy as np
from scipy import stats


def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input > expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input < expected_dims:
        np_output = np.expand_dims(np_input, 0)
    assert len(np_output.shape) == expected_dims
    return np_output


def binary_metrics(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred >= 0.5)
    label = (label >= 0.5)

    TP = float((pred * label).sum())
    FP = float((pred * (1 - label)).sum())
    FN = float(((1 - pred) * (label)).sum())
    TN = float(((1 - pred) * (1 - label)).sum())

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    IoU = TP / (TP + FP + FN + 1e-10)
    accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-10)

    return precision, recall, f1_score, IoU, accuracy


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def avg(self):
        return self.sum / self.count if self.count > 0 else 0

# 设置路径

# gt_dir = r'F:\Bo Yu\data\Xinjiang\test\mask'
gt_dir = r'/hdd2/Bo Yu/data/HL/test/mask'
# gt_dir = r'F:\Bo Yu\data\sichuan_jiangyou\test\mask'
# gt_dir = r'F:\Bo Yu\data\chongqing\test\mask'

# pred_dir = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\hl\main91'
# pred_dir = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\hl\main64'

# pred_dir = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\sd\model'
# pred_dir = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\hl\no_GMRM'
# pred_dir = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\jy\main64-7'
# pred_dir = r"F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\xj_lsn\DLV3"   # 预测结果目录
# pred_dir = r"F:\Bo Yu\Net\SEANet\yb_test\pred\jy\mask1_lss"   # 预测结果目录
# pred_dir = r"F:\Bo Yu\Net\BsiNet\yb_test\pred\jy\mask1_lss"   # 预测结果目录
# pred_dir = r'F:\Bo Yu\Net\deeplabv3\yb_test\pred\jy\mask1_lss'   # 预测结果目录
# pred_dir = r'F:\Bo Yu\Net\UNet\yb_test\pred\jy\mask1_lss'   # 预测结果目录
# pred_dir = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\jy\SEANet'
'''
train_mask
train_image
train_image
train_image
train_image
'''
'''
BlurCloud
BlurFog
BlurStripe
CloudFog
CloudStripe
FogGaussion
BlurCloudStripe
BlurCloudFog
'''
pred_dir = r'/home/yubo/Bsinet_yb/Mymodel/zwtest/HL/test1/base/'
# pred_dir = r'F:\Bo Yu\Bsinet_yb\Mymodel\enhance\HL\TF\orignal'


# 初始化平均指标记录器
precision_meter = AverageMeter()
recall_meter = AverageMeter()
f1_meter = AverageMeter()
iou_meter = AverageMeter()
accuracy_meter = AverageMeter()

file_names = sorted(os.listdir(pred_dir))

for name in file_names:
    pred_path = os.path.join(pred_dir, name)
    gt_path = os.path.join(gt_dir, name)

    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    # 归一化到 0 和 1
    pred = (pred > 127).astype(np.uint8)
    label = (label > 127).astype(np.uint8)

    pre, rec, f1, iou, acc = binary_metrics(pred, label)
    precision_meter.update(pre)
    recall_meter.update(rec)
    f1_meter.update(f1)
    iou_meter.update(iou)
    accuracy_meter.update(acc)

    print(f'{name}: Pre={pre:.4f}, Rec={rec:.4f}, F1={f1:.4f}, IoU={iou:.4f}, Acc={acc:.4f}')

print('path:{}'.format(pred_dir))
print('\nFinal Average Results:')
print('Pre={:.4f}, Rec={:.4f}, F1={:.4f}, IoU={:.4f}, Acc={:.4f}'.format(
    precision_meter.avg(),
    recall_meter.avg(),
    f1_meter.avg(),
    iou_meter.avg(),
    accuracy_meter.avg()
))

print(pred_dir)