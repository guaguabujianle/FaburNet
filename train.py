import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import random
import numpy as np
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.meter import AverageMeter, str2bool
import time
from tensorboardX import SummaryWriter
from utils.erase_dataloader import JointTransform2D,ImageToImage2D,Fetal_dataset
from utils.criterion import MyCriterion
from utils.metric import DSC
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from models.faburnet import FaburNet,EMA
import sys



def parse_args(version):
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=f'faburnet', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='None')
    parser.add_argument('--deep_supervision', default=True, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--input_w', default=256, type=int, help='image width(default: 256)')
    parser.add_argument('--input_h', default=256, type=int, help='image height(default: 256)')

    # optimizer
    parser.add_argument('--optimizer', default='AdamW',
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR',
                        help='initial learning rate(default: 1e-3)')

    parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay(default: 1e-2)')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=30, type=int, metavar='N', help='early stopping (default: -1)')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer,ema_model):
    avg_meters = {'loss': AverageMeter(),}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for sample in train_loader:
        input = sample['image'].to(dtype=torch.float32).cuda()
        target = sample['label'].squeeze(1).cuda()

        # compute output
        weight = [0.1,0.2,0.3,0.4,0.5]
        if config['deep_supervision']:
            coarse_outputs, entropys, output = model(input)
            loss = 0
            for idx, coarse_output in enumerate(coarse_outputs):
                loss += criterion(coarse_output, target, entropys[idx]) * weight[idx]
            loss += criterion(output, target, None)
            loss /= len(coarse_outputs)
        else:
            output = model(input)
            loss = criterion(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ema_model.update()

        avg_meters['loss'].update(loss.item(), input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),])


def validate(config, val_loader, test_loader,ema_model, criterion,metric):
    avg_meters_val = {'loss': AverageMeter(),
                  'dice_ps': AverageMeter(),
                  'dice_fh': AverageMeter(),}
    avg_meters_test = {'loss': AverageMeter(),
                      'dice_ps': AverageMeter(),
                      'dice_fh': AverageMeter(), }
    was_training = ema_model.model.training
    ema_model.apply_shadow()
    ema_model.model.eval()
    try:
        with torch.no_grad():
            pbar = tqdm(total=len(val_loader))
            for sample in val_loader:
                input = sample['image'].to(dtype=torch.float32).cuda()
                target = sample['label'].squeeze(1).cuda()

                # compute output
                if config['deep_supervision']:
                    _, _, output = ema_model.model(input)
                    loss = criterion(output, target)
                    ps, fh = metric(output, target)
                else:
                    output = ema_model.model(input)

                    loss = criterion(output, target)
                    ps,fh = metric(output, target)

                avg_meters_val['loss'].update(loss.item(), input.size(0))
                avg_meters_val['dice_ps'].update(ps, input.size(0))
                avg_meters_val['dice_fh'].update(fh, input.size(0))

                postfix = OrderedDict([
                    ('loss', avg_meters_val['loss'].avg),
                    ('dice_ps', avg_meters_val['dice_ps'].avg),
                    ('dice_fh', avg_meters_val['dice_fh'].avg)
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()

        with torch.no_grad():
            pbar = tqdm(total=len(test_loader))
            for sample in test_loader:
                input = sample['image'].to(dtype=torch.float32).cuda()
                target = sample['label'].squeeze(1).cuda()

                # compute output
                if config['deep_supervision']:
                    _, _, output = ema_model.model(input)
                    loss = criterion(output, target)
                    ps, fh = metric(output, target)
                else:
                    output = ema_model.model(input)

                    loss = criterion(output, target)
                    ps,fh = metric(output, target)

                avg_meters_test['loss'].update(loss.item(), input.size(0))
                avg_meters_test['dice_ps'].update(ps, input.size(0))
                avg_meters_test['dice_fh'].update(fh, input.size(0))

                postfix = OrderedDict([
                    ('loss', avg_meters_test['loss'].avg),
                    ('dice_ps', avg_meters_test['dice_ps'].avg),
                    ('dice_fh', avg_meters_test['dice_fh'].avg)
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()
    finally:
        ema_model.restore()
        ema_model.model.train(was_training)
    
    return OrderedDict([('loss_val', avg_meters_val['loss'].avg),
                        ('loss_test', avg_meters_test['loss'].avg),
                        ('dice_ps_val', avg_meters_val['dice_ps'].avg),
                        ('dice_ps_test', avg_meters_test['dice_ps'].avg),
                        ('dice_fh_val', avg_meters_val['dice_fh'].avg),
                        ('dice_fh_test', avg_meters_test['dice_fh'].avg)])



def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataloader(img_size=256, batch_size=8, fold=0):
    tf_train = JointTransform2D(img_size=img_size, low_img_size=128, ori_size=img_size, crop=None, p_flip=0.5,
                                p_rota=0.5,p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.0, p_gama=0.0, p_distor=0.0, color_jitter_params=None,
                                long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=img_size, low_img_size=128, ori_size=img_size, crop=None, p_flip=0.0,
                              color_jitter_params=None, long_mask=True)

    train_split = f"train_fold_{fold}"
    val_split = f"val_fold_{fold}"

    trainfold = ImageToImage2D("./dataset/data/train", train_split, tf_train, img_size=img_size,erase=True)
    valfold = ImageToImage2D("./dataset/data/train", val_split, tf_val, img_size=img_size,erase=False)

    trainloader = DataLoader(trainfold, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(valfold, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    root_path = Path('./dataset/data/test/testdataset')
    image_files = np.array([(root_path / Path("image_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 701)])
    label_files = np.array([(root_path / Path("label_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 701)])
    with open("./dataset/data/test/test.txt", "r") as file:
        lines = file.readlines()
    test_index = [int(line.strip().split("/")[-1]) - 1 for line in lines]
    tf_test = JointTransform2D(img_size=img_size, low_img_size=128,
                               ori_size=img_size, crop=None, p_flip=0.0, color_jitter_params=None, long_mask=True)
    testdataset = Fetal_dataset(transform=tf_test,
                                list_dir=(image_files[np.array(test_index)], label_files[np.array(test_index)]))

    testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    return trainloader, valloader, testloader

def log_param(model):
    alphas = {}
    for i in range(1, 5):
        decoder_block = getattr(model, f"spatial_decoder{i}")
        if hasattr(decoder_block, 'alpha'):
            alphas[f"spatial_decoder{i}"] = decoder_block.alpha.item()
    return alphas

def main(version=None):
    seed_torch()
    config = vars(parse_args(version))


    os.makedirs("./tsboardx", exist_ok=True)
    os.makedirs(f"./tsboardx/{config['name']}", exist_ok=True)
    my_writer = SummaryWriter(log_dir=f"tsboardx/{config['name']}")
    param_writer = SummaryWriter(log_dir=f"tsboardx/alpha")

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs(f"./checkpoints/{config['name']}", exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f"./checkpoints/{config['name']}/config.yml", 'w') as f:
        yaml.dump(config, f)

    criterion = MyCriterion()
    evaluator = DSC()

    cudnn.benchmark = True
    # create model
    model = FaburNet().cuda()
    ema_model = EMA(model, 0.9)
    ema_model.register()


    if config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=config['lr'],
                                      betas=(0.9, 0.999), weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config['factor'],
                                                   patience=config['patience'], verbose=1,
                                                   min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    train_loader,val_loader,test_loader = get_dataloader()


    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),

        ('loss_val', []),
        ('dice_ps_val', []),
        ('dice_fh_val', []),
        ('dice_mean_val', []),

        ('loss_test', []),
        ('dice_ps_test', []),
        ('dice_fh_test', []),
    ])

    best_val_dc_mean = float('-inf')
    trigger = 0

    for epoch in range(config['epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        print('Epoch [%d/%d] - lr %.8f' % (epoch + 1, config['epochs'], current_lr))

        train_log = train(config, train_loader, model, criterion, optimizer,ema_model)
        val_log = validate(config, val_loader, test_loader,ema_model, criterion,evaluator)
        validation_dice_mean = (val_log['dice_ps_val'] + val_log['dice_fh_val']) / 2
        

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(validation_dice_mean)

        print('loss %.4f - val_loss %.4f - val_dc_ps %.4f - val_dc_fh %.4f - val_dc_mean %.4f - test_dc_ps %.4f - test_dc_fh %.4f'
              % (train_log['loss'], val_log['loss_val'], val_log['dice_ps_val'],val_log['dice_fh_val']
                 ,validation_dice_mean, val_log['dice_ps_test'],val_log['dice_fh_test']))

        log['epoch'].append(epoch)
        log['lr'].append(current_lr)
        log['loss'].append(train_log['loss'])

        log['loss_val'].append(val_log['loss_val'])
        log['dice_ps_val'].append(val_log['dice_ps_val'])
        log['dice_fh_val'].append(val_log['dice_fh_val'])
        log['dice_mean_val'].append(validation_dice_mean)

        log['loss_test'].append(val_log['loss_test'])
        log['dice_ps_test'].append(val_log['dice_ps_test'])
        log['dice_fh_test'].append(val_log['dice_fh_test'])


        pd.DataFrame(log).to_csv(f'./checkpoints/%s/log.csv' %
                                 config['name'], index=False)

        my_writer.add_scalar('loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('lr', current_lr, global_step=epoch)

        my_writer.add_scalar('loss_val', val_log['loss_val'], global_step=epoch)
        my_writer.add_scalar('dice_ps_val', val_log['dice_ps_val'], global_step=epoch)
        my_writer.add_scalar('dice_fh_val', val_log['dice_fh_val'], global_step=epoch)
        my_writer.add_scalar('dice_mean_val', validation_dice_mean, global_step=epoch)

        my_writer.add_scalar('loss_test', val_log['loss_test'], global_step=epoch)
        my_writer.add_scalar('dice_ps_test', val_log['dice_ps_test'], global_step=epoch)
        my_writer.add_scalar('dice_fh_test', val_log['dice_fh_test'], global_step=epoch)
        trigger += 1

        if validation_dice_mean > best_val_dc_mean:
            torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'ema_model': {
                name: value.detach().clone()
                for name, value in ema_model.shadow.items()
            },
            'optimizer': optimizer.state_dict()
        }, f"./checkpoints/{config['name']}/epoch{epoch+1}_{val_log['loss_val']:.5f}_{val_log['dice_ps_val']:.5f}_{val_log['dice_ps_test']:.5f}_{val_log['dice_fh_val']:.5f}_{val_log['dice_fh_test']:.5f}.pth")

            best_val_dc_mean = validation_dice_mean
            print("=> saved best model (val_dc_mean: %.4f)" % best_val_dc_mean)
            trigger = 0
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break
        
        torch.cuda.empty_cache()
        params = log_param(model)
        for name, param in params.items():
            param_writer.add_scalar(name, param, global_step=epoch)


if __name__ == '__main__':
    main()
