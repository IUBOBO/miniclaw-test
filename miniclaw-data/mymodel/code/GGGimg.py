import cv2
import os
import numpy as np


def visualize_classification(image_folder, label_folder, output_folder, threshold=128):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)

    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image is None or label_img is None:
            print(f"Error reading {image_file} or {label_file}")
            continue

        # Threshold the predicted image to binary
        _, pred_bin = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # Convert single channel to 3 channels
        output_img = np.zeros((pred_bin.shape[0], pred_bin.shape[1], 3), dtype=np.uint8)

        # Correctly classified regions: pred_bin matches label_img and are 255 (white)
        correct_classified = (pred_bin == 255) & (label_img == 255)
        output_img[correct_classified] = [255, 255, 255]  # White

        # Over-classified regions (red): pred_bin is 255, label_img is 0
        over_classified = (pred_bin == 255) & (label_img == 0)
        output_img[over_classified] = [0, 0, 255]  # Red

        # Under-classified regions (blue): pred_bin is 0, label_img is 255
        under_classified = (pred_bin == 0) & (label_img == 255)
        output_img[under_classified] = [255, 0, 0]  # Blue

        # Areas below threshold remain black (no need to assign explicitly, output_img is initialized to zeros)

        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, output_img)

        print(f"Processed {image_file}, saved to {output_path}")


# sd
# image_folder =r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\sd\main64'   # 预测掩码图
# # # image_folder = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\sd\DPLV3'
# label_folder = r"F:\Bo Yu\data\Xinjiang\test\mask"                 # 真实掩码图
# output_folder = r'F:\Bo Yu\Bsinet_yb\Mymodel\visualization\sd\main64_2' # 可视化结果图

# # hl
# image_folder =r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\hl\main91'   # 预测掩码图
# # image_folder = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\hl\BsiNet'
# label_folder = r"F:\Bo Yu\data\HL\test\mask"                 # 真实掩码图
# output_folder = r'F:\Bo Yu\Bsinet_yb\Mymodel\visualization\hl\main91' # 可视化结果图
# image_folder =r'F:\Bo Yu\Bsinet_yb\experimence\55'   # 预测掩码图
# image_folder = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\hl\no_ETA'
# label_folder = r"F:\Bo Yu\data\HL\test\mask"                 # 真实掩码图
# output_folder = r'F:\Bo Yu\Bsinet_yb\experimence\visualization\no_ETA' # 可视化结果图

# image_folder =r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\5-3\double\no_Bbranch_ETA'   # 预测掩码图
# # # image_folder = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\sd\DPLV3'
# label_folder = r"F:\Bo Yu\Bsinet_yb\Mymodel\pred\5-3\mask"                 # 真实掩码图
# output_folder = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\5-3\double\no_Bbranch_ETA_v' # 可视化结果图


# sd
# image_folder =r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\jy\main64'   # 预测掩码图
# # image_folder = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\jy\UNet'
# # image_folder = r'F:\Bo Yu\data\Xinjiang\test\new7_1'
# label_folder = r"F:\Bo Yu\data\XinJiang\test\mask"                 # 真实掩码图
# output_folder = r'F:\Bo Yu\Bsinet_yb\Mymodel\visualization\sd\new7_1' # 可视化结果图

'''
train_mask
train_image
train_image
train_image
train_image
'''

image_folder = r'/home/yubo/Bsinet_yb/Mymodel/OtherNet/sc/HBG-2/'
label_folder = r"/hdd2/Bo Yu/data/HL/test/mask"                 # 真实掩码图
# label_folder = r"/hdd2/Bo Yu/data/sichuan_jiangyou/test/mask/"                 # 真实掩码图
output_folder = r'/home/yubo/Bsinet_yb/Mymodel/visualization/jy/HBG-2/' # 可视化结果图


visualize_classification(image_folder, label_folder, output_folder)
print(image_folder)


