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
from models import SEANet
from dataset import BasicDataset

Image.MAX_IMAGE_PIXELS = None


def predict_img(net, full_img, device):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if net.num_classes > 1:
            probs1 = F.softmax(output[0], dim=1)
            probs2 = F.softmax(output[1], dim=1)
        else:
            probs1 = torch.sigmoid(output[0])
            probs2 = torch.sigmoid(output[1])

        probs1 = probs1.squeeze(0)
        probs2 = probs2.squeeze(0)
        tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(full_img.size[1]),
                                 transforms.ToTensor()])
        probs1 = tf(probs1.cpu())
        probs2 = tf(probs2.cpu())

        boundary_pred = probs1.squeeze().cpu().numpy()
        distance_pred = probs2.squeeze().cpu().numpy()
    return boundary_pred, distance_pred


def get_args():
    class Args:
        gpu_id = 0
        model = r'E:\Xian-Jia He\GIS\coding\DDM-Net\data\Xinjiang\ckpts\SEANet\CP_epoch150.pth'
        input_folder = r'E:\Xian-Jia He\GIS\coding\DDM-Net\data\Xinjiang\test\img'  # 指定输入图像所在的文件夹路径
        output_folder = r'E:\Xian-Jia He\GIS\coding\DDM-Net\data\Xinjiang\pred\SEANet\boundary'  # 指定输出图像的文件夹路径
        dis_folder = r'E:\Xian-Jia He\GIS\coding\DDM-Net\data\Xinjiang\pred\SEANet\mask'
        viz = True
        no_save = False
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


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = SEANet(num_classes=1)

    logging.info('Loading model {}'.format(args.model))
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info('Model loaded !')

    in_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(('.jpg', '.png', '.tif'))]
    for i, fn in enumerate(in_files):
        geotransform, projection = get_georeference_info(os.path.dirname(fn))
        logging.info('\nPredicting image {} ...'.format(fn))
        img = Image.open(fn)
        distance_pred, boundary_pred = predict_img(net=net, full_img=img, device=device)
        # boundary_pred = predict_img(net=net, full_img=img, device=device)
        # 将预测结果映射到 0-255 范围
        # boundary_pred[boundary_pred > 0.1] = 1
        # boundary_pred[boundary_pred <= 0.1] = 0
        boundary_pred = (boundary_pred * 255).astype(np.uint8)
        distance_pred = (distance_pred * 255).astype(np.uint8)

        if not args.no_save:
            out_fn1 = os.path.join(args.output_folder, os.path.basename(fn))
            out_fn2 = os.path.join(args.dis_folder, os.path.basename(fn))

            save_mask_with_georeference(boundary_pred, out_fn1, geotransform, projection)
            save_mask_with_georeference(distance_pred, out_fn2, geotransform, projection)
            logging.info('Mask saved to {}'.format(out_fn1))

        if args.viz:
            logging.info('Visualizing results for image {}, close to continue ...'.format(fn))


