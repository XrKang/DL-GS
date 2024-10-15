#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
# import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from arguments import ModelParams, PipelineParams, OptimizationParams
from random import randint
from torch import nn
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from utils.depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from lpipsPyTorch import lpips
from DLDE import Train_Dataset, Test_Dataset, DLDE_Net, CharbonnierLoss, ContextualLoss
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as tf



def training(args, training_data_loader, val_set_loader):
    
    args.cuda = torch.cuda.is_available()
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    
    cudnn.benchmark = True
    print("===> Loading datasets")
    training_data_loader= training_data_loader
    progress_bar = tqdm(range(args.start_epoch, 
                              args.nEpochs * len(training_data_loader)),
                              desc="Training progress")

    print("===> Building model")
    model = DLDE_Net(args)

    # criterion = Loss_train()
    # criterion = nn.MSELoss()
    criterion = CharbonnierLoss()
    criterion_prec = ContextualLoss()

    print("===> Setting GPU")
    if args.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        criterion = criterion.cuda()
        criterion_prec = criterion_prec.cuda()
        
    load_model_path = args.load_dlde_path
    if os.path.isfile(load_model_path):
        print("=> loading checkpoint '{}'".format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6, betas=(0.9, 0.999))
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)

    print("===> Training")
    args.event_dir = os.path.join(args.log_path, args.model_name)
    if not os.path.exists(args.event_dir):
        os.makedirs(args.event_dir)
    print("===> event dir", args.event_dir)

    if TENSORBOARD_FOUND:
       print("Tensorboard is available")
       event_writer = SummaryWriter(args.event_dir)
    else:
        assert False, print("Tensorboard not available: not logging progress")

    model_out_path = os.path.join(args.save_path, args.model_name)
    print("===> model_path", model_out_path)
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
    print()

    Best_val = 0
    total_iter = 0
    PSNR_val = valid(args, model=None, val_set_loader=val_set_loader)
    # is_best_val = PSNR_val > Best_val
    # Best_val = max(PSNR_val, Best_val)
    for epoch in range(args.start_epoch, args.nEpochs + 1):

        lr_scheduler.step()
        loss_epoch = 0
        PSNR_epoch = 0
        model.train()
        for iteration, (render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1) in enumerate(training_data_loader):
            total_iter = total_iter + 1

            if args.cuda:
                render_img = render_img.cuda()
                tele_align = tele_align.cuda()
                wide_sr = wide_sr.cuda()
                wide_lr = wide_lr.cuda()
                W_ref_0 = W_ref_0.cuda()
                T_ref_0 = T_ref_0.cuda()
                W_ref_1 = W_ref_1.cuda()
                T_ref_1 = T_ref_1.cuda()
            pred = model(render_img, W_ref_0, T_ref_0, W_ref_1, T_ref_1)    
            pred = torch.clamp(pred, 0., 1.)
            
            # loss_mse_sr =  0.5 * args.lambda_list[0] * criterion(pred, wide_sr)
            # loss_mse_tele =   0.8 * args.lambda_list[0] * criterion(pred, tele_align)
            loss_mse_sr =  0.8 * args.lambda_list[0] * criterion(pred, wide_sr)
            loss_mse_tele =   0.5 * args.lambda_list[0] * criterion(pred, tele_align)
            loss_mse = loss_mse_sr + loss_mse_tele
            loss_ssim = args.lambda_list[1] * (1.0 - ssim(pred, tele_align))
            # loss_down = args.lambda_list[2] * criterion(torch.nn.functional.interpolate(pred, scale_factor=0.5, mode='bicubic'), wide_lr)
            loss_perc = 0.05 * criterion_prec(pred, tele_align)
            # loss = loss_mse + loss_ssim + loss_down + loss_perc
            loss = loss_mse + loss_ssim  + loss_perc

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

            with torch.no_grad():
                PSNR = psnr(pred, tele_align).mean().double()
                PSNR_epoch += PSNR
                if total_iter % 10 ==0:
                    progress_bar.set_postfix({"lr": '{:.6f}'.format(optimizer.param_groups[0]["lr"]), "loss": '{:.6f}'.format(loss.item()),  "loss_mse": '{:.6f}'.format(loss_mse.item()),"loss_ssim": '{:.6f}'.format(loss_ssim.item()),"loss_perc": '{:.6f}'.format(loss_perc.item())})

                    progress_bar.update(10)

             
                if iteration % 10 == 0:
                    event_writer.add_scalar('Loss', loss.item(), total_iter)
                    event_writer.add_scalar('loss_mse', loss_mse.item(), total_iter)
                    event_writer.add_scalar('loss_ssim', loss_ssim.item(), total_iter)
                    # event_writer.add_scalar('loss_down', loss_down.item(), total_iter)
                    event_writer.add_scalar('loss_perc', loss_perc.item(), total_iter)

                    event_writer.add_scalar('PSNR', PSNR.mean().double().item(), total_iter)


        if epoch % 1 == 0:
            with torch.no_grad():
                PSNR_val = valid(args, model, val_set_loader)
                is_best_val = PSNR_val > Best_val
                Best_val = max(PSNR_val, Best_val)
                if is_best_val:
                    model_save = os.path.join(model_out_path, "model_epoch_{}_val{}.pth".format(epoch, Best_val))
                    torch.save(model.state_dict(), model_save)
                    print("Checkpoint saved to {}".format(model_save))
        
    progress_bar.close()



def valid(args, model, val_set_loader):
    torch.cuda.empty_cache()
    val_set_loader = val_set_loader
    if model!=None:
        model.eval()
    save_path = args.event_dir
    ssims = []
    psnrs = []
    lpipss = []
    for iteration, (render_img, gt_image, W_ref_0, T_ref_0, W_ref_1, T_ref_1, file_name) in enumerate(val_set_loader):

        if args.cuda:
            render_img = render_img.cuda()
            gt_image = gt_image.cuda()
            W_ref_0 = W_ref_0.cuda()
            T_ref_0 = T_ref_0.cuda()
            W_ref_1 = W_ref_1.cuda()
            T_ref_1 = T_ref_1.cuda()
             
        if model==None:
            pred = render_img
        else:
            pred = model(render_img, W_ref_0, T_ref_0, W_ref_1, T_ref_1)        
        pred = torch.clamp(pred, 0., 1.)
        _mask = None

        gt_image = transforms.ToPILImage()(torch.squeeze(gt_image.data.cpu(), 0))
        pred = transforms.ToPILImage()(torch.squeeze(pred.data.cpu(), 0))

        if iteration%4==0:
            gt_image.save(save_path + '/{}_gt.jpg'.format(str(file_name[0])))

            pred.save(save_path + '/{}_recon.jpg'.format(str(file_name[0])))

            render_img = transforms.ToPILImage()(torch.squeeze(render_img.data.cpu(), 0))
            render_img.save(save_path + '/{}_render.jpg'.format(str(file_name[0])))

        pred = np.array(pred)
        gt_image = np.array(gt_image)
        pred = tf.to_tensor(pred).unsqueeze(0)[:, :3, :, :].cuda()
        gt_image = tf.to_tensor(gt_image).unsqueeze(0)[:, :3, :, :].cuda()
        _mask = None
        _psnr = psnr(pred, gt_image, _mask).mean().double().item()
        _ssim = ssim(pred, gt_image, _mask).mean().double().item()
        _lpips = lpips(pred, gt_image, _mask, net_type='vgg').item()

        psnrs.append(_psnr)
        ssims.append(_ssim)
        lpipss.append(_lpips)
    
    psnr_test = np.array(psnrs).mean()
    ssim_test = np.array(ssims).mean()
    lpips_test = np.array(lpipss).mean()
    print("VAL===> Val_Avg. PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f} ".format(psnr_test, ssim_test, lpips_test))
    print('--------------------------------')
    return  psnr_test


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training DLDE")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")

    # data loader
    parser.add_argument("--train_render_dir", type=str,
                        default='',
                        help="HyperSet path")
    
    parser.add_argument("--sr_wide_dir", type=str,
                        default='',
                        help="RGBSet path")

    parser.add_argument("--lr_wide_dir", type=str,
                        default='',
                        help="RGBSet path")
    
    parser.add_argument("--tele_align_dir", type=str,
                        default='',
                        help="RGBSet path")
    
    parser.add_argument("--test_render_dir", type=str,
                        default='',
                        help="HyperSet path")
    parser.add_argument("--gt_dir", type=str,
                        default='',
                        help="RGBSet path")
    # train setting
    parser.add_argument('--lambda_list', type=list, default=[1, 0.2])

    parser.add_argument("--patch_size", type=int, default=160, help="patch_size")
    parser.add_argument("--num_patch", type=int, default=5, help="num_patch")
    parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
    parser.add_argument("--nEpochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--lr", type=float, default=2 * 1e-4, help="Learning Rate. Default=1e-4")
    parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
    parser.add_argument("--milestones", type=list, default=[2], help="how many epoch to reduce the lr")
    parser.add_argument("--gamma", type=int, default=0.5, help="how much to reduce the lr each time")

    # model&events path
    parser.add_argument("--load_dlde_path", type=str, default = '')
    parser.add_argument('--log_path', default='./DLDE_log', help='log path')
    parser.add_argument('--save_path', default="./DLDE_save_path", help='log path')
    parser.add_argument('--model_name', default='', help='model')

    args = parser.parse_args()
    # --------------------------------------------
    if "10v" in args.train_render_dir:
        args.num_patch = 50
        args.nEpochs = 16
    if "20v" in args.train_render_dir:
        args.num_patch = 25
        args.nEpochs = 16
    if "90v" in args.train_render_dir:
        args.num_patch = 2
        args.nEpochs_stage2 = 18
    # --------------------------------------------

    # # DLDE model setting
    args.embed_ch = 32
    args.n_rcablocks = 5
    args.front_RBs = 3
    args.back_RBs = 3

    # --------------------------------------------
    #" # Initialize system state (RNG)
    safe_state(args.quiet)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    from torch.utils.data import DataLoader

    # --------------------------------------------
    train_set = Train_Dataset(args, scene)
    valid_set = Test_Dataset(args, scene)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batchSize, shuffle=False)
    valid_data_loader = DataLoader(dataset=valid_set, num_workers=0, batch_size=1, shuffle=False)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, training_data_loader, valid_data_loader)
