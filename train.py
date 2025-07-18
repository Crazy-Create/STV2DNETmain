import argparse
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter
from datasets.loader import PairLoader
import os
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from models import *
import matplotlib.pyplot as plt
import pandas as pd
import datetime  # 引入datetime模块用于获取当前时间
from torch.cuda.amp import autocast

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-6K', type=str, help='dataset name')
#parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
#parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--exp', default='reside6k', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1,2,3,4,5', type=str, help='GPUs used for training')
parser.add_argument('--pretrained_model', default='dehazeformer-s', type=str, help='预训练模型名称（不带.pth后缀）')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()
    torch.cuda.empty_cache()
    network.train()

    # 创建进度条
    pbar = tqdm(train_loader, desc="Training", ncols=100, total=len(train_loader), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]")

    try:
        for batch_idx, batch in enumerate(pbar):
            source_img = batch['source'].cuda()
            target_img = batch['target'].cuda()

            with autocast():
                output = network(source_img)
                loss = criterion(output, target_img)   #核心损失计算的地方

            losses.update(loss.item())

            optimizer.zero_grad()
            # 梯度更新
            scaler.scale(loss).backward()   # 混合精度反向传播
            scaler.step(optimizer)          # 参数更新
            scaler.update()                 # 缩放因子更新

            # 更新进度条百分比显示
            progress_percentage = (batch_idx + 1) / len(train_loader) * 100
            pbar.set_postfix(loss=losses.avg, progress=f'{progress_percentage:.2f}%')

    finally:
        # 确保在训练结束时关闭进度条
        pbar.close()

    return losses.avg

def valid(val_loader, network):
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            output = network(source_img).clamp_(-1, 1)

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()    # 评估指标计算
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()
    criterion = nn.L1Loss()   #criterion就是L1损失
    # 预训练模型加载 ========================
    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    pretrained_path = os.path.join(os.getcwd(), args.pretrained_model + '.pth')   # 绝对路径
    if os.path.exists(pretrained_path):
        print(f'\n==> Loading pretrained model from {pretrained_path}')
        checkpoint = torch.load(pretrained_path)

        # 处理DataParallel的键名问题
        state_dict = checkpoint['state_dict']
        if not isinstance(network, nn.DataParallel) and all(k.startswith('module.') for k in state_dict):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        load_result = network.load_state_dict(state_dict, strict=False)
        #load_result = network.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f'==> Load status:')
        print(f'Successfully loaded keys: {len(load_result.missing_keys)}')
        print(f'Missing keys: {load_result.missing_keys}')
        print(f'Unexpected keys: {load_result.unexpected_keys}')
    # ==========================================
    # 3.损失函数与优化器
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):
        print('==> Start training, current model name: ' + args.model)

        # 打印训练开始时的北京时间
        start_time = datetime.datetime.now()  # 获取当前时间
        print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        best_psnr = 0
        epoch_data = []  # 用于存储每个epoch的训练和验证数据

        for epoch in tqdm(range(setting['epochs']), desc="Epoch Progress", ncols=100):
            train_loss = train(train_loader, network, criterion, optimizer, scaler)   #所有训练周期在这
            writer.add_scalar('train_loss', train_loss, epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network)
                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                epoch_data.append({'epoch': epoch, 'train_loss': train_loss, 'valid_psnr': avg_psnr})

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))
                    print('best_psnr: %.3f', best_psnr)

                writer.add_scalar('best_psnr', best_psnr, epoch)

        # 保存训练和验证数据到 Excel
        df = pd.DataFrame(epoch_data)
        df.to_excel(os.path.join(save_dir, 'training_log.xlsx'), index=False)

        # 绘制训练和验证曲线
        epochs = df['epoch']
        train_loss = df['train_loss']
        valid_psnr = df['valid_psnr']

        # 训练曲线图1: train_loss
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label='Train Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.title('Training Loss per Epoch')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'train_loss_curve.png'))

        # 训练曲线图2: valid_psnr
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, valid_psnr, label='Validation PSNR', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.title('Validation PSNR per Epoch')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'valid_psnr_curve.png'))

        # 打印训练结束时的北京时间
        end_time = datetime.datetime.now()  # 获取结束时间
        print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        print('==> Existing trained model')
        exit(1)
