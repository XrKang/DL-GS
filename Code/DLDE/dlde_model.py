# from scene import Scene
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import  glob 
import os
import math
import DLDE.utils_arch as arch
# import utils_arch as arch
import functools
import torchvision.ops as ops


class Tele_align(nn.Module):
    def __init__(self, args):
        super(Tele_align, self).__init__()
        self.conv_hard_0 = nn.Conv2d(args.embed_ch*2, args.embed_ch, 3, 1, 1, bias=True)
        self.conv_hard_1 = nn.Conv2d(args.embed_ch*2, args.embed_ch, 3, 1, 1, bias=True)
        self.out_0 = nn.Conv2d(args.embed_ch, args.embed_ch, 3, 1, 1, bias=True)
        self.out_1 = nn.Conv2d(args.embed_ch, args.embed_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)
    
    def forward(self, render_image, T_ref_0, T_ref_1):

        # downsample for low cost 
        T_ref_0_down = F.interpolate(T_ref_0, scale_factor=1/4,  mode='bilinear', align_corners=True)
        T_ref_1_down = F.interpolate(T_ref_1, scale_factor=1/4,  mode='bilinear', align_corners=True)
        render_image_down = F.interpolate(render_image, size=(T_ref_0_down.size(2), T_ref_0_down.size(3)), 
                                     mode='bilinear', align_corners=True)
       
        # unfold for relative map
        render_image_unfold  = F.unfold(render_image_down, kernel_size=(3, 3), padding=1)

        T_ref0_unfold = F.unfold(T_ref_0_down, kernel_size=(3, 3), padding=1)
        T_ref0_unfold = T_ref0_unfold.permute(0, 2, 1)
        T_ref1_unfold = F.unfold(T_ref_1_down, kernel_size=(3, 3), padding=1)
        T_ref1_unfold = T_ref1_unfold.permute(0, 2, 1)

        # normalization
        T_ref0_unfold = F.normalize(T_ref0_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        T_ref1_unfold = F.normalize(T_ref1_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        render_image_unfold  = F.normalize(render_image_unfold, dim=1) # [N, C*k*k, H*W]

        # relative map, index 
        R_0 = torch.bmm(T_ref0_unfold, render_image_unfold) #[N, Hr*Wr, H*W]
        R_0_star, R_0_star_arg = torch.max(R_0, dim=1) #[N, H*W]
        R_1 = torch.bmm(T_ref1_unfold, render_image_unfold) #[N, Hr*Wr, H*W]
        R_1_star, R_1_star_arg = torch.max(R_1, dim=1) #[N, H*W]

        # hard attention 
        T_ref0_unfold_Hard = F.unfold(T_ref_0, kernel_size=(12, 12), padding=4, stride=4)
        T_ref1_unfold_Hard = F.unfold(T_ref_1, kernel_size=(12, 12), padding=4, stride=4)

        Hard_0_unfold = self.bis(T_ref0_unfold_Hard, 2, R_0_star_arg)
        Hard_1_unfold = self.bis(T_ref1_unfold_Hard, 2, R_1_star_arg)

        Hard_0 = F.fold(Hard_0_unfold, output_size=T_ref_0.size()[-2:], kernel_size=(12,12), padding=4, stride=4) / (3.*3.)
        Hard_1 = F.fold(Hard_1_unfold, output_size=T_ref_1.size()[-2:], kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

        # intorpolate for High resolution
        h, w = render_image.shape[2:]
        Hard_0 = F.interpolate(Hard_0, size=(h, w), mode='bilinear', align_corners=True)
        Hard_1 = F.interpolate(Hard_1, size=(h, w), mode='bilinear', align_corners=True)

        Hard_0_fusion = torch.cat([render_image, Hard_0], dim=1)
        Hard_0_fusion = self.lrelu(self.conv_hard_0(Hard_0_fusion))

        Hard_1_fusion = torch.cat([render_image, Hard_1], dim=1)
        Hard_1_fusion = self.lrelu(self.conv_hard_1(Hard_1_fusion))

        # soft attention
        Soft_map_0 = R_0_star.view(R_0_star.size(0), 1, render_image_down.size(2), render_image_down.size(3))
        Soft_map_1 = R_1_star.view(R_1_star.size(0), 1, render_image_down.size(2), render_image_down.size(3))
        
        Soft_map_0 = F.interpolate(Soft_map_0, size=(h, w), mode='bilinear', align_corners=True)
        Soft_map_1 = F.interpolate(Soft_map_1, size=(h, w), mode='bilinear', align_corners=True)

        Soft_0 = Hard_0_fusion * Soft_map_0
        Soft_1 = Hard_1_fusion * Soft_map_1

        Soft_0 = self.lrelu(self.out_0(Soft_0))
        Soft_1 = self.lrelu(self.out_1(Soft_1))

        return Soft_0, Soft_1

class Wide_align(nn.Module):
    def __init__(self, args):
        super(Wide_align, self).__init__()
        # warp & attention
        self.DFconv_0 = ops.DeformConv2d(args.embed_ch, args.embed_ch, 3, stride=1, padding=1, dilation=1, bias=False, groups=4)
        self.DFconv_1 = ops.DeformConv2d(args.embed_ch, args.embed_ch, 3, stride=1, padding=1, dilation=1, bias=False, groups=4)
        
        self.offset_0_conv0 = nn.Conv2d(args.embed_ch*2, 18, 3, 1, 1, bias=True)
        self.offset_1_conv0 = nn.Conv2d(args.embed_ch*2, 18, 3, 1, 1, bias=True)
        
        self.Q = nn.Conv2d(args.embed_ch, args.embed_ch, 3, 1, 1, bias=True)

        self.K_0 = nn.Conv2d(args.embed_ch, args.embed_ch, 3, 1, 1, bias=True)
        self.K_1 = nn.Conv2d(args.embed_ch, args.embed_ch, 3, 1, 1, bias=True)

        self.V_0 = nn.Conv2d(args.embed_ch, args.embed_ch, 3, 1, 1, bias=True)
        self.V_1 = nn.Conv2d(args.embed_ch, args.embed_ch, 3, 1, 1, bias=True)

        self.fusion_0 = nn.Conv2d(args.embed_ch, args.embed_ch, 3, 1, 1, bias=True)
        self.fusion_1 = nn.Conv2d(args.embed_ch, args.embed_ch, 3, 1, 1, bias=True)
        self.fusion_out_0 = nn.Conv2d(args.embed_ch*2, args.embed_ch, 3, 1, 1, bias=True)
        self.fusion_out_1 = nn.Conv2d(args.embed_ch*2, args.embed_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, render_image, W_ref_0, W_ref_1):
        # warp
        offset_0 = torch.cat([render_image, W_ref_0], dim=1)
        offset_0 = self.lrelu(self.offset_0_conv0(offset_0))

        offset_1 = torch.cat([render_image, W_ref_1], dim=1)
        offset_1 = self.lrelu(self.offset_1_conv0(offset_1))

        # print(offset_0.shape, offset_1.shape)
        W_ref_0 = self.lrelu(self.DFconv_0(W_ref_0, offset_0))
        W_ref_1 = self.lrelu(self.DFconv_1(W_ref_1, offset_1))

        # attention
        Q_render = self.lrelu(self.Q(render_image))

        K_Wref_0 = self.lrelu(self.K_0(W_ref_0))
        K_Wref_1 = self.lrelu(self.K_1(W_ref_1))

        V_Wref_0 = self.lrelu(self.V_0(W_ref_0))
        V_Wref_1 = self.lrelu(self.V_1(W_ref_1))
        
        att_wide_0 = torch.sum(Q_render * K_Wref_0, 1).unsqueeze(1)  # B, 1, H, W
        att_wide_0 = torch.sigmoid(att_wide_0)                       # B, 1, H, W
        att_wide_1 = torch.sum(Q_render * K_Wref_1, 1).unsqueeze(1)  # B, 1, H, W
        att_wide_1 = torch.sigmoid(att_wide_1)                       # B, 1, H, W

        fusion_0 = self.lrelu(self.fusion_0(V_Wref_0 * att_wide_0))
        fusion_1 = self.lrelu(self.fusion_1(V_Wref_1 * att_wide_1))

        fusion_0 = torch.cat([fusion_0, render_image], dim=1)
        fusion_1 = torch.cat([fusion_1, render_image], dim=1)

        fusion_0 = self.lrelu(self.fusion_out_0(fusion_0))
        fusion_1 = self.lrelu(self.fusion_out_1(fusion_1))
    
        return fusion_0, fusion_1

class DL_alignment(nn.Module):
    def __init__(self, args):
        super(DL_alignment, self).__init__()
        self.wide_align = Wide_align(args)
        self.tele_align = Tele_align(args)
        
    def forward(self, rend_image, W_ref_0, T_ref_0, W_ref_1, T_ref_1):
        fusion_0_wide, fusion_1_wide = self.wide_align(rend_image, W_ref_0, W_ref_1)
        fusion_0_tele, fusion_1_tele = self.tele_align(rend_image, T_ref_0, T_ref_1)

        return fusion_0_wide, fusion_0_tele, fusion_1_wide, fusion_1_tele


    
class Adaptive_fusion(nn.Module):
    def __init__(self, args):
        super(Adaptive_fusion, self).__init__()
        self.sAtt_1 = nn.Conv2d(args.embed_ch * 5, args.embed_ch, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(args.embed_ch * 2, args.embed_ch, 1, 1, bias=True)

        self.fusion = nn.Conv2d(args.embed_ch * 5, args.embed_ch, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, render_image, W_ref_0, T_ref_0, W_ref_1, T_ref_1):
        # spatial attention
        fea_fusion = torch.cat([render_image, W_ref_0, T_ref_0, W_ref_1, T_ref_1], dim=1)
        att = self.lrelu(self.sAtt_1(fea_fusion))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)

        fea_fusion = torch.cat([render_image, W_ref_0, T_ref_0, W_ref_1, T_ref_1], dim=1)
        fea_fusion = self.lrelu(self.fusion(fea_fusion))
        fea_fusion = fea_fusion * att

        return fea_fusion

class DLDE_Net(nn.Module):
    def __init__(self, args):
        super(DLDE_Net, self).__init__()
        # args.embed_ch: number of feature channels
        # args.n_rcablocks: number of residual channel attention blocks
        # args.front_RBs: number of residual groups for feature extraction
        # args.back_RBs: number of residual groups for reconstruction
        self.conv_first = nn.Conv2d(3, args.embed_ch, 3, 1, 1, bias=True)
        RGblock = functools.partial(arch.ResidualGroup, arch.CA_conv, args.embed_ch, kernel_size=3, reduction=16,
                                       act=nn.ReLU, res_scale=1, n_resblocks=args.n_rcablocks)
        self.feature_extraction = arch.make_layer(RGblock, args.front_RBs)
        self.DL_alignment = DL_alignment(args)
        self.Adaptive_fusion = Adaptive_fusion(args)
        self.recon_trunk = arch.make_layer(RGblock, args.back_RBs)
        self.conv_last = nn.Conv2d(args.embed_ch, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.weight_init(self.conv_first)
        # self.weight_init(self.RGblock)
        self.weight_init(self.feature_extraction)
        self.weight_init(self.DL_alignment)
        self.weight_init(self.Adaptive_fusion)
        self.weight_init(self.recon_trunk)
        # self.weight_init(self.conv_last)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.kaiming_normal_(m.bias, mode='fan_out')

    def forward(self, render_image, W_ref_0, T_ref_0, W_ref_1, T_ref_1):
        res = render_image
        render_image = self.lrelu(self.conv_first(render_image))
        W_ref_0 = self.lrelu(self.conv_first(W_ref_0))
        T_ref_0 = self.lrelu(self.conv_first(T_ref_0))
        W_ref_1 = self.lrelu(self.conv_first(W_ref_1))
        T_ref_1 = self.lrelu(self.conv_first(T_ref_1))

        feat_render = self.feature_extraction(render_image)
        feat_Wref_0 = self.feature_extraction(W_ref_0)
        feat_Wref_1 = self.feature_extraction(W_ref_1)
        feat_Tref_0 = self.feature_extraction(T_ref_0)
        feat_Tref_1 = self.feature_extraction(T_ref_1)

        feat_Wref_0_align, feat_Wref_1_align, feat_Tref_0_align, feat_Tref_1_align = \
            self.DL_alignment(feat_render, feat_Wref_0, feat_Tref_0, feat_Wref_1, feat_Tref_1)
        feat_fusion = self.Adaptive_fusion(feat_render, feat_Wref_0_align, feat_Tref_0_align, feat_Wref_1_align, feat_Tref_1_align)
        feat_recon = self.recon_trunk(feat_fusion)
        out =  self.lrelu(self.conv_last(feat_recon)) + res
        # out =  self.lrelu(self.conv_last(feat_recon))

        return out
