'''
                              _ooOoo_
                             o8888888o
                             88" . "88
                             (| -_- |)
                             O\  =  /O
                          ____/`---'\____
                        .'  \\|     |//  `.
                       /  \\|||  :  |||//  \
                      /  _||||| -:- |||||-  \
                      |   | \\\  -  /// |   |
                      | \_|  ''\---/''  |   |
                      \  .-\__  `-`  ___/-. /
                    ___`. .'  /--.--\  `. . __
                 ."" '<  `.___\_<|>_/___.'  >'"".
                | | :  `- \`.;`\ _ /`;.`/ - ` : | |
               \  \ `-.   \_ __\ /__ _/   .-` /  /
           ======`-.____`-.___\_____/___.-`____.-'======
                              `=---='
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      佛祖保佑        永无BUG
             佛曰:
                    写字楼里写字间，写字间里程序员；
                    程序人员写程序，又拿程序换酒钱。
                    酒醒只在网上坐，酒醉还来网下眠；
                    酒醉酒醒日复日，网上网下年复年。
                    但愿老死电脑间，不愿鞠躬老板前；
                    奔驰宝马贵者趣，公交自行程序员。
                    别人笑我忒疯癫，我笑自己命太贱；
                    不见满街漂亮妹，哪个归得程序员？
'''

import argparse
import logging
import os
import sys
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from dataset import BasicDataset
from tqdm import tqdm
from models import BsiNet
import matplotlib.pyplot as plt
import warnings
from losses import *

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

dir_img = r'D:\Study\SWPU\GIS\paper\Code\parcel_delineation\Xinjiang\img/'
dir_boundary = r'D:\Study\SWPU\GIS\paper\Code\parcel_delineation\Xinjiang\boundary/'
dir_mask = r'D:\Study\SWPU\GIS\paper\Code\parcel_delineation\Xinjiang\mask/'
dir_distance = r'D:\Study\SWPU\GIS\paper\Code\parcel_delineation\Xinjiang\train\dist_contour/'
dir_checkpoint = r'D:\Study\SWPU\GIS\paper\Code\parcel_delineation\Xinjiang\ckpts\BsiNet/'
alpha = 0.5


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu_id', dest='gpu_id', metavar='G', type=int, default=0, help='GPU ID')
    parser.add_argument('-u', '--unet_type', dest='unet_type', metavar='U', type=str, default='v1',
                        help='UNet type: v1/v2/v3')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2, help='Batch size',
                        dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=8,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


def train_net(
        net,
        device,
        epochs=5,
        batch_size=1,
        lr=0.1,
        val_percent=0.1,
        save_cp=True,
        img_scale=1,
):
    dataset = BasicDataset(dir_img, dir_boundary, dir_mask, dir_distance)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    global_step = 0

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Dataset size:    {len(dataset)}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device}
        Images scaling:  {img_scale}"""
    )

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scaler = GradScaler()

    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05

    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.BCEWithLogitsLoss()
    criterion3 = nn.MSELoss()
    # 新增：记录每个epoch的平均训练和验证损失
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        net.train()
        total_train_loss = 0

        with tqdm(
                total=n_train, desc=f"Epoch {epoch + 1}/{200}", unit="img"
        ) as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                true_boundary = batch["boundary"]
                true_mask = batch["mask"]
                true_distance = batch["distance"]

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_boundary = true_boundary.to(device)
                true_mask = true_mask.to(device)
                true_distance = true_distance.to(device)

                boundary_type = torch.float32
                mask_type = torch.float32
                distance_type = torch.float32

                true_boundary = true_boundary.to(device=device, dtype=boundary_type)
                true_mask = true_mask.to(device=device, dtype=mask_type)
                true_distance = true_distance.to(device=device, dtype=distance_type)

                optimizer.zero_grad()

                with autocast():
                    pred_boundary, pred_mask, pred_distance = net(imgs)
                    loss1 = criterion1(pred_boundary, true_boundary)
                    loss2 = criterion2(pred_mask, true_mask)
                    loss3 = criterion3(pred_distance, true_distance)

                    loss = 0.4 * loss1 + 0.3 * loss2 + 0.3 * loss3

                # loss = loss.to(torch.float32)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.update(imgs.shape[0])
                global_step += 1
                total_train_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device=device, dtype=torch.float32)
                true_boundary = batch["boundary"].to(device=device, dtype=boundary_type)
                true_mask = batch["mask"].to(device=device, dtype=mask_type)
                true_distance = batch["distance"].to(device=device, dtype=mask_type)

                pred_boundary, pred_mask, pred_distance = net(imgs)
                loss1 = criterion1(pred_boundary, true_boundary)
                loss2 = criterion2(pred_mask, true_mask)
                loss3 = criterion3(pred_distance, true_distance)

                loss = 0.4 * loss1 + 0.3 * loss2 + 0.3 * loss3
                total_val_loss += loss.item()

        #
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        writer.add_scalar("Loss/validation", val_loss, global_step)

        logging.info(
            f"Epoch {epoch + 1} finished! Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )
        if save_cp:
            try:
                os.makedirs(dir_checkpoint, exist_ok=True)
                # logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'CP_epoch {epoch + 1} saved !')
    writer.close()

    # 绘制训练和验证损失图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    net = BsiNet(num_classes=1)
    net.to(device=device)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
        )
    except KeyboardInterrupt:
        # torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        sys.exit(0)
