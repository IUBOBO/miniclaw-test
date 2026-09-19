import cv2
import os
import numpy as np
from skimage.measure import regionprops, label


def calculate_oc(area_intersect, area_gt):
    """Calculate over-classification error."""
    return 1 - (area_intersect / area_gt)


def calculate_uc(area_intersect, area_pred):
    """Calculate under-classification error."""
    return 1 - (area_intersect / area_pred)


def calculate_tc(oc, uc):
    """Calculate total classification error."""
    return np.sqrt((oc ** 2 + uc ** 2) / 2)


def compute_metrics_for_images(image_folder, label_folder):
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)

    total_goc = 0
    total_guc = 0
    total_gtc = 0
    total_area = 0
    num_images = 0

    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image is None or label_img is None:
            print(f"Error reading {image_file} or {label_file}")
            continue

        # Threshold the predicted image to binary
        _, pred_bin = cv2.threshold(image, 205, 255, cv2.THRESH_BINARY)

        # Label the regions in label and pred images
        label_label = label(label_img)
        pred_label = label(pred_bin)

        # Get the properties of the labeled regions
        label_props = regionprops(label_label)
        pred_props = regionprops(pred_label)

        # Map to store maximum overlapping area
        overlap_area_map = {}

        for pred in pred_props:
            pred_id = pred.label
            pred_mask = (pred_label == pred_id)

            for gt in label_props:
                gt_id = gt.label
                gt_mask = (label_label == gt_id)

                overlap_area = np.sum(pred_mask & gt_mask)
                if (pred_id, gt_id) not in overlap_area_map:
                    overlap_area_map[(pred_id, gt_id)] = overlap_area
                else:
                    overlap_area_map[(pred_id, gt_id)] = max(overlap_area_map[(pred_id, gt_id)], overlap_area)

        # Calculate OC, UC, and weighted metrics
        img_total_goc = 0
        img_total_guc = 0
        img_total_gtc = 0
        img_total_area = 0

        for pred in pred_props:
            pred_id = pred.label
            pred_area = pred.area

            max_overlap = 0
            for gt in label_props:
                gt_id = gt.label
                gt_area = gt.area
                if (pred_id, gt_id) in overlap_area_map:
                    overlap_area = overlap_area_map[(pred_id, gt_id)]
                    if overlap_area > max_overlap:
                        max_overlap = overlap_area
                        best_gt_id = gt_id
                        best_gt_area = gt_area

            if max_overlap == 0:
                # No overlap with any GT
                continue

            oc = calculate_oc(max_overlap, best_gt_area)
            uc = calculate_uc(max_overlap, pred_area)
            tc = calculate_tc(oc, uc)

            img_total_goc += oc * pred_area
            img_total_guc += uc * pred_area
            img_total_gtc += tc * pred_area
            img_total_area += pred_area

        if img_total_area > 0:
            avg_img_goc = img_total_goc / img_total_area
            avg_img_guc = img_total_guc / img_total_area
            avg_img_gtc = img_total_gtc / img_total_area

            print(f"Metrics for {image_file}:")
            print(f"  Global Over-Classification (GOC): {avg_img_goc}")
            print(f"  Global Under-Classification (GUC): {avg_img_guc}")
            print(f"  Global Total Errors (GTC): {avg_img_gtc}")

            total_goc += img_total_goc
            total_guc += img_total_guc
            total_gtc += img_total_gtc
            total_area += img_total_area
            num_images += 1

    if total_area > 0:
        avg_goc = total_goc / total_area
        avg_guc = total_guc / total_area
        avg_gtc = total_gtc / total_area

        print(f"\nAverage Metrics for all images:")
        print(f"{'Average Global Over-Classification (GOC):':<50} {avg_goc}")
        print(f"{'Average Global Under-Classification (GUC):':<50} {avg_guc}")
        print(f"{'Average Global Total Errors (GTC):':<50} {avg_gtc}")
    else:
        print("No valid images found.")


# Example usage

# folder2_path = r'F:\Bo Yu\data\xj_lsn\xj\test\mask'
# folder2_path = r'F:\Bo Yu\data\Xinjiang\test\mask'
folder2_path = r'/hdd2/Bo Yu/data/HL/test/mask'
# folder2_path = r'F:\Bo Yu\data\sichuan_jiangyou\test\mask'
# folder2_path = r'F:\Bo Yu\data\chongqing\test\mask'
# folder2_path = r"F:\Bo Yu\data\Xinjiang\test\mask"

# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\hl\main64_test3'
# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\hl\main91'

# folder1_path = r'F:\Bo Yu\Bsinet_yb\pred\TAFModel\xj\mask1'
# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\sd\model2'
# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\jy\main64-5'
# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\jy\SEANet'
# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\xj_lsn\DLV3'
# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\jy\main64-2'
# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\pred\sd\main64_lsnxj_1'
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

folder1_path = r'/home/yubo/Bsinet_yb/Mymodel/zwtest/HL/test1/base/'
# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\sc\REAU'
# folder1_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\hl\TFNet'


compute_metrics_for_images(folder1_path, folder2_path)
print(folder1_path)
