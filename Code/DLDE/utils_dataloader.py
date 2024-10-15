import numpy as np
import imageio
import torch
import os
import torch.utils.data as data
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import random
from scene import Scene
from DLDE.image_selection import image_selection
from PIL import Image

def get_patch(lr, hr, wide_lr, wide_ref0, tele_ref0, wide_ref1, tele_ref1, patch_size=108, shave=False):
    scale = 2
    _, L_h, L_w = lr.shape

    L_h, L_w = L_h//scale, L_w//scale
    L_p = patch_size//scale

    if shave:
        L_x = random.randrange(L_w//7, 6*L_w//7 - L_p +1)
        L_y = random.randrange(L_h//7, 6*L_h//7 - L_p +1)
    else:
        L_x = random.randrange(0,L_w - L_p + 1)
        L_y = random.randrange(0, L_h - L_p + 1)
        
    H_x, H_y = scale * L_x, scale * L_y
    H_p =  scale * L_p   
    patch_LR = lr[:, H_y:H_y + H_p, H_x:H_x + H_p]
    patch_HR = hr[:, H_y:H_y + H_p, H_x:H_x + H_p]
    patch_wide_lr = wide_lr[:, L_y:L_y + L_p, L_x:L_x + L_p]
    patch_wide_ref0 = wide_ref0[:, H_y:H_y + H_p, H_x:H_x + H_p]
    patch_tele_ref0 = tele_ref0[:, L_y:L_y + L_p, L_x:L_x + L_p]
    patch_wide_ref1 = wide_ref1[:, H_y:H_y + H_p, H_x:H_x + H_p]
    patch_tele_ref1 = tele_ref1[:, L_y:L_y + L_p, L_x:L_x + L_p]
    return patch_LR, patch_HR, patch_wide_lr, patch_wide_ref0, patch_tele_ref0, patch_wide_ref1, patch_tele_ref1

def get_patch2(lr, hr, wide_sr, wide_lr, wide_ref0, tele_ref0, wide_ref1, tele_ref1, patch_size=108, shave=False):
    scale = 2
    _, L_h, L_w = lr.shape

    L_h, L_w = L_h//scale, L_w//scale
    L_p = patch_size//scale

    if shave:
        L_x = random.randrange(L_w//7, 6*L_w//7 - L_p +1)
        L_y = random.randrange(L_h//7, 6*L_h//7 - L_p +1)
    else:
        L_x = random.randrange(0, L_w - L_p +1)
        L_y = random.randrange(0, L_h - L_p +1)
    H_x, H_y = scale * L_x, scale * L_y
    H_p =  scale * L_p   
    patch_LR = lr[:, H_y:H_y + H_p, H_x:H_x + H_p]
    patch_HR = hr[:, H_y:H_y + H_p, H_x:H_x + H_p]
    patch_wide_sr = wide_sr[:, H_y:H_y + H_p, H_x:H_x + H_p]
    patch_wide_lr = wide_lr[:, L_y:L_y + L_p, L_x:L_x + L_p]
    patch_wide_ref0 = wide_ref0[:, H_y:H_y + H_p, H_x:H_x + H_p]
    patch_tele_ref0 = tele_ref0[:, L_y:L_y + L_p, L_x:L_x + L_p]
    patch_wide_ref1 = wide_ref1[:, H_y:H_y + H_p, H_x:H_x + H_p]
    patch_tele_ref1 = tele_ref1[:, L_y:L_y + L_p, L_x:L_x + L_p]
    return patch_LR, patch_HR, patch_wide_sr, patch_wide_lr, patch_wide_ref0, patch_tele_ref0, patch_wide_ref1, patch_tele_ref1


def augment(*args):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    def _augment(img):
        if hflip: torch.flip(img, [2])
        if vflip: torch.flip(img, [1])        
        
        if rot90: img = torch.rot90(img, 1, [1, 2])
        
        return img

    return [_augment(a) for a in args]


def PILtoTorch(pil_image):
    image_PIL = pil_image
    image = torch.from_numpy(np.array(image_PIL)) / 255.0
    return image.permute(2, 0, 1)

class Train_Dataset(data.Dataset):
    def __init__(self, args, scene):
        super(Train_Dataset, self).__init__()
        self.args = args
        self.render_dir = args.train_render_dir
        self.sr_wide = args.sr_wide_dir
        self.lr_wide = args.lr_wide_dir
        self.tele_align_dir = args.tele_align_dir
        self.train_views = scene.getTrainCameras().copy()
        self.num_img = len(self.train_views)
        self.num_patch = args.num_patch
       

    def __getitem__(self, idx):
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = self._load_file(idx)
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = get_patch(render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1,
        #                                                            patch_size=self.args.patch_size, shave=True)
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = augment(render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1)
        # return render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = self._load_file(idx)
        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = get_patch2(render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1,
                                                                   patch_size=self.args.patch_size, shave=True)

        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = augment(render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1)

        return render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

    def __len__(self):
        return self.num_img * self.num_patch

    def center_crop_shift(self, img, shift=24):
        _, h, w= img.shape
        img = img[:, h//4:h//4*3, w//4+shift:w//4*3+shift]
        return img
    
    def center_crop(self, img):
        # crop image
        _, h, w= img.shape
        img = img[:, h//4:h//4*3, w//4:w//4*3]
        return img

    def _load_file(self, idx):
        idx = idx % self.num_img
        train_view = self.train_views[idx]
        other_views = self.train_views[:idx] + self.train_views[idx+1:]
        # print(len(self.train_views), len(other_views))

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img =  Image.open(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img = PILtoTorch(render_img)
        render_img = self.center_crop_shift(render_img)
        _, h, w = render_img.shape
        render_img = render_img[:, :h//4*4, :w//4*4]

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        wide_lr = Image.open(os.path.join(self.lr_wide, train_view.image_name.replace('train_sr', 'train')+'.jpg'))
        wide_lr = PILtoTorch(wide_lr)
        wide_lr = self.center_crop_shift(wide_lr)
        _, h, w = wide_lr.shape
        wide_lr = wide_lr[:, :h//4*4, :w//4*4]
        
        # print(os.path.join(self.tele_align_dir, train_view.image_name.replace('train_sr', 'trainTele')+'.jpg'))
        tele_align =  Image.open(os.path.join(self.tele_align_dir, train_view.image_name.replace('train_sr', 'trainTele')+'.jpg'))
        tele_align = PILtoTorch(tele_align)
        _, h, w = tele_align.shape
        tele_align = tele_align[:, :h//4*4, :w//4*4]

        sr_wide =  Image.open(os.path.join(self.sr_wide, train_view.image_name+'.jpg'))
        sr_wide = PILtoTorch(sr_wide)
        sr_wide = self.center_crop_shift(sr_wide)
        _, h, w = sr_wide.shape
        sr_wide = sr_wide[:, :h//4*4, :w//4*4]

        W_ref_0, T_ref_0, W_ref_1, T_ref_1 = image_selection(train_view, other_views)
        
        W_ref_0, W_ref_1 = self.center_crop_shift(W_ref_0), self.center_crop_shift(W_ref_1)
        _, h, w = W_ref_0.shape
        W_ref_0, W_ref_1 = W_ref_0[:, :h//4*4, :w//4*4], W_ref_1[:, :h//4*4, :w//4*4]

        _, h, w = T_ref_0.shape
        T_ref_0, T_ref_1 = T_ref_0[:, :h//4*4, :w//4*4], T_ref_1[:, :h//4*4, :w//4*4]
        T_ref_0, T_ref_1 = self.center_crop(T_ref_0), self.center_crop(T_ref_1)

        # print(render_img.shape, tele_align.shape, W_ref_0.shape, T_ref_0.shape, W_ref_1.shape, T_ref_1.shape)

        return render_img, tele_align, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

class Train_Real_Dataset(data.Dataset):
    def __init__(self, args, scene):
        super(Train_Real_Dataset, self).__init__()
        self.args = args
        self.render_dir = args.train_render_dir
        self.sr_wide = args.sr_wide_dir
        self.lr_wide = args.lr_wide_dir
        self.tele_align_dir = args.tele_align_dir
        self.train_views = scene.getTrainCameras().copy()
        self.num_img = len(self.train_views)
        self.num_patch = args.num_patch
       

    def __getitem__(self, idx):
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = self._load_file(idx)
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = get_patch(render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1,
        #                                                            patch_size=self.args.patch_size, shave=True)
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = augment(render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1)
        # return render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = self._load_file(idx)
        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = get_patch2(render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1,
                                                                   patch_size=self.args.patch_size, shave=True)

        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = augment(render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1)

        return render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

    def __len__(self):
        return self.num_img * self.num_patch


    def center_crop_shift(self, img, shift):
        _, h, w= img.shape
        img = img[:, (h // 10 * 3)-shift: (h // 10 * 8)-shift, (w // 16 * 4): (w // 16 * 12)]
        return img
    def center_crop(self, img):
        # crop image
        _, h, w= img.shape
        img = img[:, (h // 10 * 3): (h // 10 * 8), (w // 16 * 4): (w // 16 * 12)]
        return img

    def _load_file(self, idx):

        if "360_04" in self.tele_align_dir:
            shift = 30
        if "360_01" in self.tele_align_dir:
            shift = 90
        elif "360_0" in self.tele_align_dir:
            shift = 80
        elif "FF" in self.tele_align_dir:
            shift = 5
               
        idx = idx % self.num_img
        train_view = self.train_views[idx]
        other_views = self.train_views[:idx] + self.train_views[idx+1:]
        # print(len(self.train_views), len(other_views))

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img =  Image.open(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img = PILtoTorch(render_img)
        render_img = self.center_crop_shift(render_img, shift)
        _, h, w = render_img.shape
        render_img = render_img[:, :h//4*4, :w//4*4]

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        wide_lr = Image.open(os.path.join(self.lr_wide, train_view.image_name.replace('sr_train', 'train')+'.jpg'))
        wide_lr = PILtoTorch(wide_lr)
        wide_lr = self.center_crop_shift(wide_lr, shift)
        _, h, w = wide_lr.shape
        wide_lr = wide_lr[:, :h//4*4, :w//4*4]
        
        # print(os.path.join(self.tele_align_dir, train_view.image_name.replace('train_sr', 'trainTele')+'.jpg'))
        tele_align =  Image.open(os.path.join(self.tele_align_dir, train_view.image_name.replace('train', 'trainTele')+'.jpg'))
        tele_align = PILtoTorch(tele_align)
        _, h, w = tele_align.shape
        tele_align = tele_align[:, :h//4*4, :w//4*4]

        sr_wide =  Image.open(os.path.join(self.sr_wide, train_view.image_name+'.jpg'))
        sr_wide = PILtoTorch(sr_wide)
        sr_wide = self.center_crop_shift(sr_wide, shift)
        _, h, w = sr_wide.shape
        sr_wide = sr_wide[:, :h//4*4, :w//4*4]

        W_ref_0, T_ref_0, W_ref_1, T_ref_1 = image_selection(train_view, other_views)
        
        W_ref_0, W_ref_1 = self.center_crop_shift(W_ref_0, shift), self.center_crop_shift(W_ref_1, shift)
        _, h, w = W_ref_0.shape
        W_ref_0, W_ref_1 = W_ref_0[:, :h//4*4, :w//4*4], W_ref_1[:, :h//4*4, :w//4*4]

        _, h, w = T_ref_0.shape
        T_ref_0, T_ref_1 = T_ref_0[:, :h//4*4, :w//4*4], T_ref_1[:, :h//4*4, :w//4*4]
        T_ref_0, T_ref_1 = self.center_crop(T_ref_0), self.center_crop(T_ref_1)

        # print(render_img.shape, tele_align.shape, W_ref_0.shape, T_ref_0.shape, W_ref_1.shape, T_ref_1.shape)

        return render_img, tele_align, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1



class Train_Dataset_stage1(data.Dataset):
    def __init__(self, args, scene):
        super(Train_Dataset_stage1, self).__init__()
        self.args = args
        self.render_dir = args.train_render_dir
        self.lr_wide = args.lr_wide_dir
        self.sr_wide = args.sr_wide_dir
        # self.tele_align_dir = args.tele_align_dir
        self.train_views = scene.getTrainCameras().copy()
        self.num_img = len(self.train_views)
        self.num_patch = args.num_patch_stage1


    def __getitem__(self, idx):

        render_img, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = self._load_file(idx)
        render_img, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = get_patch(render_img, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1,
                                                                   patch_size=self.args.patch_size, shave=True)

        render_img, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = augment(render_img, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1)

        return render_img, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

    def __len__(self):
        return self.num_img * self.num_patch

    # def center_crop_shift(self, img, shift=24):
    #     _, h, w= img.shape
    #     img = img[:, h//4:h//4*3, w//4+shift:w//4*3+shift]
    #     return img
    
    # def center_crop(self, img):
    #     _, h, w= img.shape
    #     img = img[:, h//4:h//4*3, w//4:w//4*3]
    #     return img

    def _load_file(self, idx):
        idx = idx % self.num_img
        train_view = self.train_views[idx]
        other_views = self.train_views[:idx] + self.train_views[idx+1:]
        # print(len(self.train_views), len(other_views))

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img =  Image.open(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img = PILtoTorch(render_img)
        _, h, w = render_img.shape
        render_img = render_img[:, :h//4*4, :w//4*4]

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))sr_train
        # wide_lr = Image.open(os.path.join(self.lr_wide, train_view.image_name.replace('train_sr', 'train')+'.jpg'))
        wide_lr = Image.open(os.path.join(self.lr_wide, train_view.image_name.replace('sr_train', 'train')+'.jpg'))
        wide_lr = PILtoTorch(wide_lr)
        _, h, w = wide_lr.shape
        wide_lr = wide_lr[:, :h//4*4, :w//4*4]
        
        # print(os.path.join(self.sr_wide, train_view.image_name + '.jpg'))
        sr_wide =  Image.open(os.path.join(self.sr_wide, train_view.image_name+'.jpg'))
        sr_wide = PILtoTorch(sr_wide)
        _, h, w = sr_wide.shape
        sr_wide = sr_wide[:, :h//4*4, :w//4*4]


        W_ref_0, T_ref_0, W_ref_1, T_ref_1 = image_selection(train_view, other_views)
        
        _, h, w = W_ref_0.shape
        W_ref_0, W_ref_1 = W_ref_0[:, :h//4*4, :w//4*4], W_ref_1[:, :h//4*4, :w//4*4]

        _, h, w = T_ref_0.shape
        T_ref_0, T_ref_1 = T_ref_0[:, :h//4*4, :w//4*4], T_ref_1[:, :h//4*4, :w//4*4]
        # print("shape")
        # print(render_img.shape, sr_wide.shape, W_ref_0.shape, T_ref_0.shape, W_ref_1.shape, T_ref_1.shape)

        return render_img, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1
   

class Train_Dataset_stage2(data.Dataset):
    def __init__(self, args, scene):
        super(Train_Dataset_stage2, self).__init__()
        self.args = args
        self.render_dir = args.train_render_dir
        self.sr_wide = args.sr_wide_dir
        self.lr_wide = args.lr_wide_dir
        self.tele_align_dir = args.tele_align_dir
        self.train_views = scene.getTrainCameras().copy()
        self.num_img = len(self.train_views)
        self.num_patch = args.num_patch_stage2
       

    def __getitem__(self, idx):
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = self._load_file(idx)
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = get_patch(render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1,
        #                                                            patch_size=self.args.patch_size, shave=True)
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = augment(render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1)
        # return render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = self._load_file(idx)
        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = get_patch2(render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1,
                                                                   patch_size=self.args.patch_size, shave=True)

        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = augment(render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1)

        return render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

    def __len__(self):
        return self.num_img * self.num_patch

    def center_crop_shift(self, img, shift=24):
        _, h, w= img.shape
        img = img[:, h//4:h//4*3, w//4+shift:w//4*3+shift]
        return img
    
    def center_crop(self, img):
        _, h, w= img.shape
        img = img[:, h//4:h//4*3, w//4:w//4*3]
        return img

    def _load_file(self, idx):
        idx = idx % self.num_img
        train_view = self.train_views[idx]
        other_views = self.train_views[:idx] + self.train_views[idx+1:]
        # print(len(self.train_views), len(other_views))

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img =  Image.open(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img = PILtoTorch(render_img)
        render_img = self.center_crop_shift(render_img)
        _, h, w = render_img.shape
        render_img = render_img[:, :h//4*4, :w//4*4]

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        wide_lr = Image.open(os.path.join(self.lr_wide, train_view.image_name.replace('train_sr', 'train')+'.jpg'))

        wide_lr = PILtoTorch(wide_lr)
        wide_lr = self.center_crop_shift(wide_lr)
        _, h, w = wide_lr.shape
        wide_lr = wide_lr[:, :h//4*4, :w//4*4]
        
        # print(os.path.join(self.tele_align_dir, train_view.image_name.replace('train_sr', 'trainTele')+'.jpg'))
        tele_align =  Image.open(os.path.join(self.tele_align_dir, train_view.image_name.replace('train_sr', 'trainTele')+'.jpg'))
        tele_align = PILtoTorch(tele_align)
        _, h, w = tele_align.shape
        tele_align = tele_align[:, :h//4*4, :w//4*4]

        sr_wide =  Image.open(os.path.join(self.sr_wide, train_view.image_name+'.jpg'))
        sr_wide = PILtoTorch(sr_wide)
        sr_wide = self.center_crop_shift(sr_wide)
        _, h, w = sr_wide.shape
        sr_wide = sr_wide[:, :h//4*4, :w//4*4]

        W_ref_0, T_ref_0, W_ref_1, T_ref_1 = image_selection(train_view, other_views)
        
        W_ref_0, W_ref_1 = self.center_crop_shift(W_ref_0), self.center_crop_shift(W_ref_1)
        _, h, w = W_ref_0.shape
        W_ref_0, W_ref_1 = W_ref_0[:, :h//4*4, :w//4*4], W_ref_1[:, :h//4*4, :w//4*4]

        _, h, w = T_ref_0.shape
        T_ref_0, T_ref_1 = T_ref_0[:, :h//4*4, :w//4*4], T_ref_1[:, :h//4*4, :w//4*4]
        T_ref_0, T_ref_1 = self.center_crop(T_ref_0), self.center_crop(T_ref_1)

        # print(render_img.shape, tele_align.shape, W_ref_0.shape, T_ref_0.shape, W_ref_1.shape, T_ref_1.shape)

        return render_img, tele_align, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

class Train_Dataset_stage2_real(data.Dataset):
    def __init__(self, args, scene):
        super(Train_Dataset_stage2_real, self).__init__()
        self.args = args
        self.render_dir = args.train_render_dir
        self.sr_wide = args.sr_wide_dir
        self.lr_wide = args.lr_wide_dir
        self.tele_align_dir = args.tele_align_dir
        self.train_views = scene.getTrainCameras().copy()
        self.num_img = len(self.train_views)
        self.num_patch = args.num_patch_stage2
       

    def __getitem__(self, idx):
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = self._load_file(idx)
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = get_patch(render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1,
        #                                                            patch_size=self.args.patch_size, shave=True)
        # render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = augment(render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1)
        # return render_img, tele_align, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = self._load_file(idx)
        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = get_patch2(render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1,
                                                                   patch_size=self.args.patch_size, shave=True)

        render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1 = augment(render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1)

        return render_img, tele_align, wide_sr, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1

    def __len__(self):
        return self.num_img * self.num_patch

    def center_crop_shift(self, img, shift):
        _, h, w= img.shape
        img = img[:, (h // 10 * 3)-shift: (h // 10 * 8)-shift, (w // 16 * 4): (w // 16 * 12)]
        return img
    def center_crop(self, img):
        # crop image
        _, h, w= img.shape
        img = img[:, (h // 10 * 3): (h // 10 * 8), (w // 16 * 4): (w // 16 * 12)]
        return img
    
    def _load_file(self, idx):
        if "360_04" in self.tele_align_dir:
            shift = 30
        if "360_01" in self.tele_align_dir:
            shift = 90
        elif "360_0" in self.tele_align_dir:
            shift = 80
        elif "FF" in self.tele_align_dir:
            shift = 5

        idx = idx % self.num_img
        train_view = self.train_views[idx]
        other_views = self.train_views[:idx] + self.train_views[idx+1:]
        # print(len(self.train_views), len(other_views))

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img =  Image.open(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        render_img = PILtoTorch(render_img)
        render_img = self.center_crop_shift(render_img, shift)
        _, h, w = render_img.shape
        render_img = render_img[:, :h//4*4, :w//4*4]

        # print(os.path.join(self.render_dir, train_view.image_name+'.jpg'))
        wide_lr = Image.open(os.path.join(self.lr_wide, train_view.image_name.replace('sr_train', 'train')+'.jpg'))
        wide_lr = PILtoTorch(wide_lr)
        wide_lr = self.center_crop_shift(wide_lr, shift)
        _, h, w = wide_lr.shape
        wide_lr = wide_lr[:, :h//4*4, :w//4*4]
        
        # print(os.path.join(self.tele_align_dir, train_view.image_name.replace('train_sr', 'trainTele')+'.jpg'))
        tele_align =  Image.open(os.path.join(self.tele_align_dir, train_view.image_name.replace('train', 'trainTele')+'.jpg'))
        tele_align = PILtoTorch(tele_align)
        _, h, w = tele_align.shape
        tele_align = tele_align[:, :h//4*4, :w//4*4]

        sr_wide =  Image.open(os.path.join(self.sr_wide, train_view.image_name+'.jpg'))
        sr_wide = PILtoTorch(sr_wide)
        sr_wide = self.center_crop_shift(sr_wide, shift)
        _, h, w = sr_wide.shape
        sr_wide = sr_wide[:, :h//4*4, :w//4*4]

        W_ref_0, T_ref_0, W_ref_1, T_ref_1 = image_selection(train_view, other_views)
        
        W_ref_0, W_ref_1 = self.center_crop_shift(W_ref_0, shift), self.center_crop_shift(W_ref_1, shift)
        _, h, w = W_ref_0.shape
        W_ref_0, W_ref_1 = W_ref_0[:, :h//4*4, :w//4*4], W_ref_1[:, :h//4*4, :w//4*4]

        _, h, w = T_ref_0.shape
        T_ref_0, T_ref_1 = T_ref_0[:, :h//4*4, :w//4*4], T_ref_1[:, :h//4*4, :w//4*4]
        T_ref_0, T_ref_1 = self.center_crop(T_ref_0), self.center_crop(T_ref_1)

        # print(render_img.shape, tele_align.shape, W_ref_0.shape, T_ref_0.shape, W_ref_1.shape, T_ref_1.shape)

        return render_img, tele_align, sr_wide, wide_lr, W_ref_0, T_ref_0, W_ref_1, T_ref_1



class Test_Dataset(data.Dataset):
    def __init__(self, args, scene):
        super(Test_Dataset, self).__init__()
        self.args = args
        # self.do_eval = True
        self.render_dir = args.test_render_dir
        self.test_views = scene.getTestCameras().copy()
        self.train_views = scene.getTrainCameras().copy()
        self.gt_dir = args.gt_dir
        # self.wide_images = [x for x in self.train_views.original_image]
        # self.tele_images = [x for x in self.train_views.tele_image]
        self.num_img = len(self.test_views)

       

    def __getitem__(self, idx):
        render_img, gt, W_ref_0, T_ref_0, W_ref_1, T_ref_1, name = self._load_file(idx)


        return render_img, gt, W_ref_0, T_ref_0, W_ref_1, T_ref_1, name

    def __len__(self):
        return self.num_img 


    def _load_file(self, idx):
        test_view = self.test_views[idx]
        train_views = self.train_views
        # print(len(self.train_views), len(other_views))

        # print(os.path.join(self.render_dir, test_view.image_name+'.jpg'))
        render_img =  Image.open(os.path.join(self.render_dir, test_view.image_name+'.jpg'))
        render_img = PILtoTorch(render_img)
        _, h, w = render_img.shape
        render_img = render_img[:, :h//4*4, :w//4*4]
        
        # print(os.path.join(self.gt_dir, test_view.image_name.replace('test_sr', 'test')+'.jpg'))
        gt =  Image.open(os.path.join(self.gt_dir, test_view.image_name.replace('test_sr', 'test')+'.jpg'))
        gt = PILtoTorch(gt)
        # gt = test_view.original_image[0:3, :, :]
        _, h, w = gt.shape
        gt = gt[:, :h//4*4, :w//4*4]

        W_ref_0, T_ref_0, W_ref_1, T_ref_1 = image_selection(test_view, train_views)
        
        # W_ref_0, W_ref_1 = self.center_crop(W_ref_0), self.center_crop(W_ref_1)
        W_ref_0, W_ref_1 = W_ref_0[:, :h//4*4, :w//4*4], W_ref_1[:, :h//4*4, :w//4*4]

        _, h, w = T_ref_0.shape
        T_ref_0, T_ref_1 = T_ref_0[:, :h//4*4, :w//4*4], T_ref_1[:, :h//4*4, :w//4*4]

        # print(render_img.shape, tele_align.shape, W_ref_0.shape, T_ref_0.shape, W_ref_1.shape, T_ref_1.shape)

        return render_img, gt, W_ref_0, T_ref_0, W_ref_1, T_ref_1, test_view.image_name
