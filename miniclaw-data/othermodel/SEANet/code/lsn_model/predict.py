import argparse
import logging
import os
os.environ['PROJ_LIB'] = r'C:\\Anaconda\\envs\\pytorch\\lib\\site-packages\\pyproj\\proj_dir\\share\\proj'
import numpy as np
import torch
import torch.nn.functional as F
import torchsummary as summary
from PIL import Image
from osgeo import gdal
from torchvision import transforms


from dataset_3task import BasicDataset
# from SNNet import model
Image.MAX_IMAGE_PIXELS = None
from SEAnet import *
# from Mednet.losses import *


def predict_img(unet_type, net, full_img, device, img_scale=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(unet_type, full_img, img_scale))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        # if net.n_classes > 1:
        #     probs = F.softmax(output, dim=1)
        # else:
        probs = torch.sigmoid(output[0])

        probs = probs.squeeze(0)
        tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(full_img.size[1]),
                                 transforms.ToTensor()])
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask


def get_args():
    class Args:
        gpu_id = 0
        # unet_type = 'Mednet'  # 'v1', 'v2', 'v3', 'UNet3Plus_DeepSup', 或 'UNet3Plus_DeepSup_CGM'
        model = 'F:\Bo Yu\\Net\SEANet\yb_test\SEA_skpts_xj\SEA_lsn\CP_epoch100_SEANet_SC_increase.pth'
        input_folder = r'F:\Bo Yu\data\Xinjiang\test\mask'  # 指定输入图像所在的文件夹路径
        output_folder = r'F:\Bo Yu\Net\SEANet\yb_test\pred\xj\mask3'  # 指定输出图像的文件夹路径
        viz = True
        no_save = False
        scale = 1
    return Args()


def get_output_filenames(args):
    in_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(('.jpg', '.png', '.tif'))]
    out_files = []

    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append('{}_OUT{}'.format(pathsplit[0], pathsplit[1]))

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def get_georeference_info(input_folder):
    input_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.tif'))]
    input_file = os.path.join(input_folder, input_files[0])  # 使用第一个输入文件获取空间参考信息
    input_dataset = gdal.Open(input_file)
    geotransform = input_dataset.GetGeoTransform()
    projection = input_dataset.GetProjection()
    return geotransform, projection


def save_mask_with_georeference(mask, output_path, geotransform, projection):
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_path, mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte)
    output_dataset.SetGeoTransform(geotransform)
    output_dataset.SetProjection(projection)
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(mask)
    output_band.FlushCache()
    output_dataset = None

def predict_large_image( net, full_img, device, crop_size=512, img_scale=1, out_threshold=0.5):
    """
    对大尺寸图像进行分块预测，并拼接结果。
    """
    net.eval()
    input_width, input_height = full_img.size

    # 结果初始化
    full_mask = np.zeros((input_height, input_width), dtype=np.float32)

    for top in range(0, input_height, crop_size):
        for left in range(0, input_width, crop_size):
            # 块的右下角坐标
            right = min(left + crop_size, input_width)
            bottom = min(top + crop_size, input_height)

            # 裁剪并预处理块
            crop_img = full_img.crop((left, top, right, bottom))
            img = torch.from_numpy(BasicDataset.preprocess(crop_img, img_scale))
            img = img.unsqueeze(0)
            img = img.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                output = net(img)
                probs = torch.sigmoid(output[0])

                probs = probs.squeeze(0)
                tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((bottom - top, right - left)),
                                         transforms.ToTensor()])
                probs = tf(probs.cpu())
                crop_mask = probs.squeeze().cpu().numpy()

            # 将预测结果填充回结果矩阵
            full_mask[top:bottom, left:right] = crop_mask

    return full_mask

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    net = SEANet()

    logging.info('Loading model {}'.format(args.model))
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info('Model loaded !')

    in_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(('.jpg', '.png', '.tif'))]
    for i, fn in enumerate(in_files):
        geotransform, projection = get_georeference_info(os.path.dirname(fn))
        logging.info('\nPredicting image {} ...'.format(fn))
        img = Image.open(fn)
        # mask = predict_img(unet_type=args.unet_type, net=net, full_img=img, img_scale=args.scale, device=device)
        mask = predict_large_image(net=net, full_img=img, crop_size=512, device=device)

        # 将预测结果映射到 0-255 范围
        mask = (mask * 255).astype(np.uint8)

        if not args.no_save:
            out_fn = os.path.join(args.output_folder, os.path.basename(fn))
            save_mask_with_georeference(mask, out_fn, geotransform, projection)
            logging.info('Mask saved to {}'.format(out_fn))

        if args.viz:
            logging.info('Visualizing results for image {}, close to continue ...'.format(fn))


