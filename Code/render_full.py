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
import matplotlib.pyplot as plt
import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import time
from tqdm import tqdm

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from utils.general_utils import vis_depth
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from lpipsPyTorch import lpips
from DLDE import Train_Dataset, Test_Dataset, DLDE_Net, CharbonnierLoss, ContextualLoss, image_selection
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args, scene, model_refine):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)
        gt = view.original_image[0:3, :, :]
        render_img = rendering["render"]
        W_ref_0, T_ref_0, W_ref_1, T_ref_1 = image_selection(view, scene.getTrainCameras())
        # print(render_img.shape, W_ref_0.shape, T_ref_0.shape, W_ref_1.shape, T_ref_1.shape)

        if args.cuda:
            render_img = render_img.unsqueeze(0).cuda()
            W_ref_0 = W_ref_0.unsqueeze(0).cuda()
            T_ref_0 = T_ref_0.unsqueeze(0).cuda()
            W_ref_1 = W_ref_1.unsqueeze(0).cuda()
            T_ref_1 = T_ref_1.unsqueeze(0).cuda()
        # print(render_img.shape, W_ref_0.shape, T_ref_0.shape, W_ref_1.shape, T_ref_1.shape)

        render_img_refine = model_refine(render_img, W_ref_0, T_ref_0, W_ref_1, T_ref_1)
        render_img_refine = torch.clamp(render_img_refine, 0., 1.)
        
        torchvision.utils.save_image(render_img_refine[0].detach().cpu(), os.path.join(render_path, view.image_name + '.jpg'))

        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".jpg"))
# 
        if args.render_depth:
            depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
            np.save(os.path.join(render_path, view.image_name + '_depth.npy'), rendering['depth'][0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)



def render_video(scene, source_path, model_path, iteration, views, gaussians, pipeline, background, fps=30, model_refine=None):
    render_path = os.path.join(model_path, 'video_OurGS+DLDE', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]

    if args.pose == 'spiral':
        render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
    else:
        render_poses = generate_ellipse_path(views)

    size = (view.original_image.shape[2], view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.R = pose[:3, :3]
        view.T = pose[:3, 3]
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)
        img = torch.clamp(rendering["render"], min=0., max=1.)

        render_img = img
        W_ref_0, T_ref_0, W_ref_1, T_ref_1 = image_selection(view, scene.getTrainCameras())
        if args.cuda:
            render_img = render_img.unsqueeze(0).cuda()
            W_ref_0 = W_ref_0.unsqueeze(0).cuda()
            T_ref_0 = T_ref_0.unsqueeze(0).cuda()
            W_ref_1 = W_ref_1.unsqueeze(0).cuda()
            T_ref_1 = T_ref_1.unsqueeze(0).cuda()
        
        render_img_refine = model_refine(render_img, W_ref_0, T_ref_0, W_ref_1, T_ref_1)
        render_img_refine = torch.clamp(render_img_refine, 0., 1.)


        torchvision.utils.save_image(render_img_refine[0].detach().cpu(), os.path.join(render_path, '{0:05d}'.format(idx) + ".jpg"))

        video_img = (render_img_refine[0].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]

        final_video.write(video_img)

    final_video.release()



def render_sets(dataset : ModelParams, pipeline : PipelineParams, args, model_refine):

    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.video:
            render_video(scene, dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTestCameras(),
                         gaussians, pipeline, background, args.fps, model_refine)

        if not args.skip_train:
            render_set(dataset.model_path, "train_OurGS+DLDE", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args, scene, model_refine)
        if not args.skip_test:
            render_set(dataset.model_path, "test_OurGS+DLDE", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args, scene, model_refine)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--pose", default="spiral", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--load_dlde_path", type=str, default = '')

    args = get_combined_args(parser)

    args.cuda = torch.cuda.is_available()

    # # DLDE model setting
    args.embed_ch = 32
    args.n_rcablocks = 5
    args.front_RBs = 3
    args.back_RBs = 3
    print("Rendering " + args.model_path)
    model_refine = DLDE_Net(args)
    if args.cuda:
        DLDE = model_refine.cuda()
        model_refine = torch.nn.DataParallel(model_refine)
    
    load_model_path = args.load_dlde_path
    if os.path.isfile(load_model_path):
        print("=> loading checkpoint '{}'".format(load_model_path))
        model_refine.load_state_dict(torch.load(load_model_path))
    else:
        assert False, print("=> no checkpoint found at '{}'".format(load_model_path))
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args, model_refine)