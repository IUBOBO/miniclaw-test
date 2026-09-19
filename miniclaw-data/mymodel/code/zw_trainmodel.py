import argparse
import logging
import os
import sys
import pandas as pd
import warnings
import yaml
from datetime import datetime

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from Dataset import MyDataset
# from Bsinet_yb.Mymodel.ModelAE2.no_GMRM_Bbranch import Tripmodel
from zw_model import STDNet
from LossFunction import MultiTaskLoss
warnings.filterwarnings("ignore")


alpha = 0.6 # 掩码损失权重
beta = 0.4 # 边界损失权重

# dir_img = r'F:\Bo Yu\data\sichuan_jiangyou\train\image/'                            # 输入图像路径
# dir_boundary = r'F:\Bo Yu\data\sichuan_jiangyou\train\contour/'                    # 输入边界路径
# dir_mask = r'F:\Bo Yu\data\sichuan_jiangyou\train\mask/'
# dir_distance = r'F:\Bo Yu\data\sichuan_jiangyou\train\dist_contour_tif/'

# dir_img = r'F:\Bo Yu\data\Xinjiang\train\img/'                            # 输入图像路径
# dir_boundary = r'F:\Bo Yu\data\Xinjiang\train\boundary/'                    # 输入边界路径
# dir_mask = r'F:\Bo Yu\data\Xinjiang\train\mask/'                            # 输入掩码路径
# dir_checkpoint = r'F:\Bo Yu\Bsinet_yb\Mymodel\AE\sd\no_GMRM' # 模型保存位置

# dir_img = r'F:\Bo Yu\data\HL\train\img/'                        # 输入图像路径
# dir_boundary = r'F:\Bo Yu\data\HL\train\boundary_new/'                   # 输入边界路径
# dir_mask = r'F:\Bo Yu\data\HL\train\newmask/'
# dir_checkpoint = r'F:\Bo Yu\Bsinet_yb\Mymodel\Mloss\hl\baseline' # 模型保存位置
#
# log_csv_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\Mloss\hl\baseline\LossData.csv'         # 损失保存位置
# log_file_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\Mloss\hl\baseline\TrainData'                            # 训练过程保存位置
# config_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\Mloss\hl\baseline\ParaData'                            # 参数保存位置
# model_name = 'baseline'
dir_img = r'/hdd2/Bo Yu/data/sichuan_jiangyou/train/image'                      # 输入图像路径
dir_boundary = r'/hdd2/Bo Yu/data/sichuan_jiangyou/train/contour'                   # 输入边界路径
dir_mask = r'/hdd2/Bo Yu/data/sichuan_jiangyou/train/mask'
dir_checkpoint = r'/home/yubo/Bsinet_yb/Mymodel/zwtest/SC/test2' # 模型保存位置

log_csv_path = r'/home/yubo/Bsinet_yb/Mymodel/zwtest/SC/test2/LossData.csv'         # 损失保存位置
log_file_path = r'/home/yubo/Bsinet_yb/Mymodel/zwtest/SC/test2/TrainData'                            # 训练过程保存位置
config_path = r'/home/yubo/Bsinet_yb/Mymodel/zwtest/SC/test2/ParaData'                            # 参数保存位置
model_name = 'zw-SC-tes2'


# dir_img = r'F:\Bo Yu\data\Xinjiang\train\img/'                            # 输入图像路径
# dir_boundary = r'F:\Bo Yu\data\Xinjiang\train\train_boundary/'                    # 输入边界路径
# dir_mask = r'F:\Bo Yu\data\Xinjiang\train\train_mask/'                            # 输入掩码路径
# dir_checkpoint = r'F:\Bo Yu\Bsinet_yb\Mymodel\Mloss\sd\main64-2' # 模型保存位置
#
# log_csv_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\Mloss\sd\main64-2\LossData.csv'         # 损失保存位置
# log_file_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\Mloss\sd\main64-2\TrainData'                            # 训练过程保存位置
# config_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\Mloss\sd\main64-2\ParaData'                            # 参数保存位置
# model_name = 'main64-2'

# dir_checkpoint = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\xj_model\Net' # 模型保存位置
# log_csv_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\xj_model\Net\LossData.csv'         # 损失保存位置
# log_file_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\xj_model\Net\TrainData'                            # 训练过程保存位置
# config_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\OtherNet\xj_model\Net\ParaData'                            # 参数保存位置
# model_name = 'Net'


# dir_img = r'F:\Bo Yu\data\HL\train\img/'                        # 输入图像路径
# dir_boundary = r'F:\Bo Yu\data\HL\train\boundary_new/'                   # 输入边界路径
# dir_mask = r'F:\Bo Yu\data\HL\train\newmask/'                           # 输入掩码路径
# # dir_checkpoint = r'F:\Bo Yu\Bsinet_yb\Mymodel\AE\hl\no_GMRM_Bbranch'  # 模型保存位置
# dir_checkpoint = 'F:\Bo Yu\Bsinet_yb\Mymodel\Mloss\hl\main64_test4'
#
# log_csv_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\para\hl\main64_test4\LossData.csv'         # 损失保存位置
# log_file_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\para\hl\main64_test4\TrainData'                            # 训练过程保存位置
# config_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\para\hl\main64_test4\ParaData'                              # 参数保存位置
# model_name = 'main64_test4'

# 损失权重配置
# alpha = 0.6 # 掩码损失权重
# beta = 0.4 # 边界损失权重
# dir_img = r'F:\Bo Yu\data\Xinjiang\train\add\train_image/'                        # 输入图像路径
# dir_boundary = r'F:\Bo Yu\data\Xinjiang\train\train_boundary/'                   # 输入边界路径
# dir_mask = r'F:\Bo Yu\data\Xinjiang\train\train_mask/'                           # 输入掩码路径
# dir_checkpoint = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\Xinjiang\main64\train_image' # 模型保存位置
#
# log_csv_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\Xinjiang\main64\train_image\LossData.csv'         # 损失保存位置
# log_file_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\Xinjiang\main64\train_image\TrainData'                            # 训练过程保存位置
# config_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\Xinjiang\main64\train_image\ParaData'                              # 参数保存位置
# model_name = 'main64-train_image'

# alpha = 0.6 # 掩码损失权重
# beta = 0.4 # 边界损失权重
# dir_img = r'F:\Bo Yu\data\HL\train\add\train_image/'                        # 输入图像路径
# dir_boundary = r'F:\Bo Yu\data\HL\train\boundary_new/'                   # 输入边界路径
# dir_mask = r'F:\Bo Yu\data\HL\train\newmask/'                           # 输入掩码路径
# dir_checkpoint = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\HL\main64\fog5' # 模型保存位置
#
# log_csv_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\HL\main64\fog5\LossData.csv'         # 损失保存位置
# log_file_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\HL\main64\fog5\TrainData'                            # 训练过程保存位置
# config_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\HL\main64\fog5\ParaData'                              # 参数保存位置
# model_name = 'main64-fog5'


# 损失权重配置
# alpha = 0.6 # 掩码损失权重
# beta = 0.4 # 边界损失权重
# dir_img = r'F:\Bo Yu\data\sichuan_jiangyou\train\add\train_image/'                        # 输入图像路径
# dir_boundary = r'F:\Bo Yu\data\sichuan_jiangyou\train\contour/'                   # 输入边界路径
# dir_mask = r'F:\Bo Yu\data\sichuan_jiangyou\train\mask/'                           # 输入掩码路径
# dir_checkpoint = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\SC\main64\train_image' # 模型保存位置
#
# log_csv_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\SC\main64\train_image\LossData.csv'         # 损失保存位置
# log_file_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\SC\main64\train_image\TrainData'                            # 训练过程保存位置
# config_path = r'F:\Bo Yu\Bsinet_yb\Mymodel\noise\SC\main64\train_image\ParaData'                              # 参数保存位置
# model_name = 'main64-train_image'

def get_args():
    parser = argparse.ArgumentParser(description="Train the ymodel on images and target masks")

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,help='Learning rate', dest='lr')

    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '--val', dest='val', type=float, default=10.0,help='Percent of the data that is used as validation (0-100)')

    parser.add_argument('--gpu-id', '-g', type=int, default=1, help='GPU ID to use')
    return parser.parse_args()


def save_model_config(net, args, config_path, optimizer, criterion):
    """将模型配置、训练参数和优化器设置保存为YAML文件"""
    config = {
        'model': {
            'name': net.__class__.__name__,
            'input_channels': net.input_channels if hasattr(net, 'input_channels') else 3,
            'num_classes': net.num_classes if hasattr(net, 'num_classes') else 1,
        },
        'optimizer': {
            'type': optimizer.__class__.__name__,
            'learning_rate': args.lr,
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'eps': optimizer.param_groups[0]['eps'] if 'eps' in optimizer.param_groups[0] else None,
        },
        'loss': {
            'type': criterion.__class__.__name__,
            'alpha': alpha,  # 掩码损失权重
            'beta': beta,  # 边界损失权重
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'validation_percent': args.val,
        },
        'hardware': {
            'device': str(device),
            'gpu_id': args.gpu_id,
            'num_workers': 0  # 数据加载的worker数量
        },
        'paths': {
            'image_dir': dir_img,
            'boundary_dir': dir_boundary,
            'mask_dir': dir_mask,
            'checkpoint_dir': dir_checkpoint,
            'log_csv': log_csv_path,
            'log_file': log_file_path,
            'config_file': config_path
        },
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'notes': model_name
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    logging.info(f'Model config saved to {config_path}')


def setup_logging(log_file_path):
    """设置日志记录到文件和终端"""
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M'
    # 清除之前的日志处理器
    logging.getLogger().handlers.clear()

    # 设置新的日志配置
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),  # 每次覆盖写入
            logging.StreamHandler()  # 输出到控制台
        ]
    )

def train_net(net, device, epochs, batch_size, lr, val_percent, save_cp=True):
    # 初始化数据集
    dataset = MyDataset(dir_img, dir_boundary, dir_mask)
    n_val = int(len(dataset) * val_percent / 100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],)

    # 数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # 日志和监控
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    # 初始化训练日志DataFrame
    log_df = pd.DataFrame(columns=[
        'epoch',
        'train_loss',
        'val_loss',
        'boundary_loss',
        'mask_loss',
        'learning_rate'
    ])

    logging.info(f'''Starting training:
        Model:           {model_name}
        Loss rate:       {alpha, beta}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Checkpoints:     {save_cp}
    ''')

    # 优化器和损失函数
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
    scaler = GradScaler()

    criterion = MultiTaskLoss(alpha=alpha, beta=beta)

    # 在训练开始前保存配置
    config_path = os.path.join(os.path.dirname(log_csv_path),
                               f'model_config_{model_name}.yaml')
    save_model_config(net, args, config_path, optimizer, criterion)

    # 训练循环
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        epoch_boundary_loss = 0
        epoch_mask_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # 数据转移到设备
                images = batch['image'].to(device)
                true_boundary = batch['boundary'].to(device)
                true_mask = batch['mask'].to(device)

                # 前向传播
                optimizer.zero_grad()
                with autocast():
                    pred_mask, pred_boundary = net(images)
                    loss_dict = criterion(pred_boundary, pred_mask, true_boundary, true_mask)
                    loss = loss_dict['total_loss']

                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 记录损失
                epoch_loss += loss.item()
                epoch_boundary_loss += loss_dict['boundary_loss']
                epoch_mask_loss += loss_dict['mask_loss']
                writer.add_scalar('Loss/train', loss.item(), global_step)
                global_step += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(images.shape[0])


        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)


        # 验证阶段
        val_loss = 0
        val_boundary_loss = 0
        val_mask_loss = 0
        net.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                true_boundary = batch['boundary'].to(device)
                true_mask = batch['mask'].to(device)

                with autocast():
                    pred_mask, pred_boundary = net(images)
                    loss_dict = criterion(pred_boundary, pred_mask, true_boundary, true_mask)
                    val_loss += loss_dict['total_loss'].item()
                    val_boundary_loss += loss_dict['boundary_loss']
                    val_mask_loss += loss_dict['mask_loss']

        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        avg_val_boundary_loss = val_boundary_loss / len(val_loader)
        avg_val_mask_loss = val_mask_loss / len(val_loader)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']


        # 记录到DataFrame
        new_log = pd.DataFrame([{
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'boundary_loss': avg_val_boundary_loss,
            'mask_loss': avg_val_mask_loss,
            'learning_rate': current_lr
        }])
        log_df = pd.concat([log_df, new_log], ignore_index=True)

        # 保存到CSV文件
        log_df.to_csv(log_csv_path, index=False)

        # 日志记录
        logging.info(f'Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f} '
                     f'- Val loss: {avg_val_loss:.4f} '
                     f'- Boundary loss: {avg_val_boundary_loss:.4f} '
                     f'- Mask loss: {avg_val_mask_loss:.4f} '
                     f'- LR: {current_lr:.5f}')

        # 保存检查点
        if save_cp and (epoch + 1) % 10 == 0:
            os.makedirs(dir_checkpoint, exist_ok=True)
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint, f'epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved!')

    writer.close()


if __name__ == '__main__':
    args = get_args()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 配置日志和输出路径
    log_dir = os.path.dirname(log_csv_path)
    # current_time = datetime.now().strftime("%Y%m%d_%H%M")

    log_file_path = os.path.join(log_dir, f'training_{model_name}.log')
    config_path = os.path.join(log_dir, f'model_config_{model_name}.yaml')
    # 设置日志系统
    setup_logging(log_file_path)



    # 初始化模型
    net = STDNet(input_channels=3, num_classes=1)
    net.to(device=device)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()

    # 设备设置
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')


    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
