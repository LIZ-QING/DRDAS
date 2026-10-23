import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json
import geomloss
from fastprogress import progress_bar
from argparse import ArgumentParser
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, wide_resnet101_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50, de_wide_resnet101_2
from utils.utils_test import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from dataset.dataset import MVTecDataset_test, MVTecDataset_train, get_data_transforms

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--save_folder', default = './RD++_checkpoint_result_pv-wide101'
                                                   '', type=str)
    parser.add_argument('--batch_size', default = 4, type=int)
    parser.add_argument('--image_size', default = 512, type=int)
    parser.add_argument('--detail_training', default='note', type = str)
    parser.add_argument('--proj_lr', default = 0.001, type=float)
    parser.add_argument('--distill_lr', default = 0.005, type=float)
    parser.add_argument('--weight_proj', default = 0.2, type=float)
    parser.add_argument('--classes', nargs="+", default=["pvel"])
    pars = parser.parse_args()
    return pars

def train(_class_, pars):
    print(_class_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)

    data_root = 'mvtec'  # 你数据集的根目录
    train_path = os.path.join(data_root, _class_, 'train')
    test_path = os.path.join(data_root, _class_)
    
    if not os.path.exists(pars.save_folder + '/' + _class_):
        os.makedirs(pars.save_folder + '/' + _class_)
    save_model_path  = pars.save_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    train_data = MVTecDataset_train(root=train_path, transform=data_transform)
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=pars.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Use pretrained ImageNet for encoder
    encoder, bn = wide_resnet101_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet101_2(pretrained=False)
    decoder = decoder.to(device)
    
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    proj_loss = Revisit_RDLoss()
    optimizer_proj = torch.optim.Adam(list(proj_layer.parameters()), lr=pars.proj_lr, betas=(0.5,0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=pars.distill_lr, betas=(0.5,0.999))

    
    best_score = 0
    best_epoch = 0
    best_auroc_px = 0
    best_auroc_sp = 0
    best_aupro_px = 0
    
    auroc_px_list = []
    auroc_sp_list = []
    aupro_px_list = []
    
    loss_proj = []
    loss_distill = []
    total_loss = []
    
    history_infor = {}


    # set appropriate epochs for specific classes (Some classes converge faster than others)
    
    if _class_ in ['carpet','leather']:
        num_epoch = 10
    if _class_ in ['grid','tile']:
        num_epoch = 260
    if _class_ in ['wood']:
        num_epoch = 100   
    if _class_ in ['cable']:
        num_epoch = 240
    if _class_ in ['capsule']:
        num_epoch = 300
    if _class_ in ['hazelnut']:
        num_epoch = 160
    if _class_ in ['metal_nut']:
        num_epoch = 160
    if _class_ in ['screw']:
        num_epoch = 280
    if _class_ in ['toothbrush']:
        num_epoch = 280
    if _class_ in ['transistor']:
        num_epoch = 300  
    if _class_ in ['zipper']:
        num_epoch = 300
    if _class_ in ['pill']:
        num_epoch = 200
    if _class_ in ['bottle']:
        num_epoch = 200
    if _class_ in ['mono']:
        num_epoch = 40
    if _class_ in ['pvel']:
        num_epoch = 3

    print(f'with class {_class_}, Training with {num_epoch} Epoch')

    lam_noise = 1  # 噪声分支权重（你自己调）
    accumulation_steps = 2  # 梯度累积

    for epoch in tqdm(range(1, num_epoch + 1)):
        bn.train()
        proj_layer.train()
        decoder.train()

        total_loss_running = 0.0
        loss_proj_running = 0.0
        loss_distill_running = 0.0
        loss_distill_noise_running = 0.0

        optimizer_proj.zero_grad(set_to_none=True)
        optimizer_distill.zero_grad(set_to_none=True)

        for i, (img, img_noise, _) in enumerate(train_dataloader):
            img = img.to(device, non_blocking=True)
            img_noise = img_noise.to(device, non_blocking=True)

            # -------- teacher features (frozen encoder) --------
            with torch.no_grad():
                inputs = encoder(img)  # normal teacher feats: [f1,f2,f3]
                inputs_noise = encoder(img_noise)

            # -------- projection --------
            feature_space_noise, feature_space = proj_layer(inputs, features_noise=inputs_noise)

            # proj loss: 主要更新 proj_layer（也可能通过图更新 bn/decoder，但一般不会）
            L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)

            # ============ 1) 正常分支：bn + decoder 一起学 ============
            outputs = decoder(bn(feature_space))
            L_distill = loss_fucntion(inputs, outputs)

            # ============ 2) 噪声分支：只学 proj_layer + bn，隔离 decoder ============
            # 冻结 decoder 参数（但 forward 仍走 autograd，让梯度回到 bn/proj）
            decoder.requires_grad_(False)
            outputs_noise = decoder(bn(feature_space_noise))
            L_distill_noise = loss_fucntion(inputs, outputs_noise)
            decoder.requires_grad_(True)

            # -------- total loss --------
            loss = L_distill + lam_noise * L_distill_noise + pars.weight_proj * L_proj

            # 梯度累积：记得除
            (loss / accumulation_steps).backward()

            # step
            if (i + 1) % accumulation_steps == 0:
                optimizer_proj.step()
                optimizer_distill.step()
                optimizer_proj.zero_grad(set_to_none=True)
                optimizer_distill.zero_grad(set_to_none=True)

            # logging（这里用未除的 loss，做 epoch 平均更直观）
            total_loss_running += loss.detach().item()
            loss_proj_running += L_proj.detach().item()
            loss_distill_running += L_distill.detach().item()
            loss_distill_noise_running += L_distill_noise.detach().item()

        # 处理最后残留梯度（不满 accumulation_steps）
        if len(train_dataloader) % accumulation_steps != 0:
            optimizer_proj.step()
            optimizer_distill.step()
            optimizer_proj.zero_grad(set_to_none=True)
            optimizer_distill.zero_grad(set_to_none=True)

        # epoch 平均
        n_batches = len(train_dataloader)
        avg_total = total_loss_running / n_batches
        avg_proj = loss_proj_running / n_batches
        avg_dist = loss_distill_running / n_batches
        avg_dist_noise = loss_distill_noise_running / n_batches

        # eval
        auroc_px, _, auroc_sp, aupro_px = evaluation_multi_proj(
            encoder, proj_layer, bn, decoder, test_dataloader, device, save_pro_csv=None, sp_csv=None
        )

        print(
            f"Epoch {epoch:03d} | "
            f"L={avg_total:.6f} "
            f"(dist={avg_dist:.6f}, distN={avg_dist_noise:.6f}, proj={avg_proj:.6f}, wproj={(pars.weight_proj * avg_proj):.6f}) | "
            f"AUROC_sp={auroc_sp:.4f} AUROC_px={auroc_px:.4f} AUPRO={aupro_px:.4f}"
        )

        if (auroc_px + auroc_sp + aupro_px) / 3 > best_score:
            best_score = (auroc_px + auroc_sp + aupro_px) / 3
            
            best_auroc_px = auroc_px
            best_auroc_sp = auroc_sp
            best_aupro_px = aupro_px
            best_epoch = epoch

            torch.save({'proj': proj_layer.state_dict(),
                       'decoder': decoder.state_dict(),
                        'bn':bn.state_dict()}, save_model_path)

            history_infor['auroc_sp'] = best_auroc_sp
            history_infor['auroc_px'] = best_auroc_px
            history_infor['aupro_px'] = best_aupro_px
            history_infor['epoch'] = best_epoch
            with open(os.path.join(pars.save_folder + '/' + _class_, f'history.json'), 'w') as f:
                json.dump(history_infor, f)
    return best_auroc_sp, best_auroc_px, best_aupro_px




if __name__ == '__main__':
    pars = get_args()
    print('Training with classes: ', pars.classes)
    all_classes = [ 'carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    
    # train all_classes
    # for c in all_classes
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = train(c, pars)
        print('Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}'.format(c, auroc_sp, auroc_px, aupro_px))
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        pd.DataFrame(metrics).to_csv(f'{pars.save_folder}/metrics_results.csv', index=False)
