import os
import cv2
import numpy as np
from PIL import Image
from skimage.io import imread, imsave
import json

def downsample(img, scale=2):
    h, w, _ = img.shape
    img = np.array(Image.fromarray(img).resize((w//4*4 // scale, h//4*4 // scale), Image.BICUBIC))
    return img

def downsample_disp(img, scale=2):
    h, w = img.shape
    img = np.array(Image.fromarray(img).resize((w//4*4 // scale, h//4*4 // scale), Image.BICUBIC))
    return img

def downsample_AO(img, scale=2):
    h, w = img.shape
    img = np.array(Image.fromarray(img).resize((w//4*4 // scale, h//4*4 // scale), Image.BICUBIC))
    return img

def centerCrop(img):
    h, w, _ = img.shape
    img = img[h//4:h//4*3, w//4:w//4*3, :]
    return img

def disparity_to_depth(disparity_map, focal_length, baseline=0.1):
    # 避免除零错误
    disparity_map[disparity_map == 0] = 0.1
    
    # 使用公式计算深度图
    depth_map = (focal_length * baseline) / disparity_map
    
    return depth_map

if __name__ == '__main__':
    target_dir = 'E:\DLNeRF\stereoNeRF\DualCameraSynthetic'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # focalLength_dic = json.load(open('E:/DLNeRF/stereoNeRF/focalLength.json'))
    source_path = 'E:\DLNeRF\stereoNeRF\SelectedScenes'
    for filename in os.listdir(source_path):
        # filename = '0036'
        wide_source = os.path.join(source_path, filename, 'left')
        tele_source = os.path.join(source_path, filename, 'right')
        # disparity_source = os.path.join(source_path, filename, 'disparity')
        # AO_source = os.path.join(source_path, filename, 'AO')

        GT_dir_target = os.path.join(target_dir, filename, 'GT')
        wide_dir_target = os.path.join(target_dir, filename, 'wide')
        tele_dir_target = os.path.join(target_dir, filename, 'tele')
        # disp_dir_target = os.path.join(target_dir, filename, 'disparity')
        # depth_dir_target = os.path.join(target_dir, filename, 'depth')
        # AO_dir_target = os.path.join(target_dir, filename, 'AO')

        if not os.path.exists(GT_dir_target):
            os.makedirs(GT_dir_target)
        if not os.path.exists(wide_dir_target):
            os.makedirs(wide_dir_target)
        if not os.path.exists(tele_dir_target):
            os.makedirs(tele_dir_target)
        # if not os.path.exists(disp_dir_target):
        #     os.makedirs(disp_dir_target)
        # if not os.path.exists(depth_dir_target):
        #     os.makedirs(depth_dir_target)
        # if not os.path.exists(AO_dir_target):
        #     os.makedirs(AO_dir_target)
        print('================' + filename + '================')

        # print(wide_source, tele_source, disparity_source, AO_source)
        # print(GT_dir_target, wide_dir_target, tele_dir_target, disp_dir_target, depth_dir_target, AO_dir_target)
        print(GT_dir_target, wide_dir_target, tele_dir_target)

        print('================' + 'GT' + '================')
        print_flat = True
        for img_name in os.listdir(wide_source):
            img_path = os.path.join(wide_source, img_name)
            img = imread(img_path)
            h, w, _ = img.shape
            img = img[:h//4*4, :w//4*4,:]
            if print_flat: 
                print(img_path, img.shape)
            save_path = os.path.join(GT_dir_target, img_name)
            imsave(save_path, img)
            if print_flat: 
                print(save_path, img.shape)
            print_flat = False
        

        print('================' + 'Wide' + '================')
        print_flat = True
        for img_name in os.listdir(wide_source):
            img_path = os.path.join(wide_source, img_name)
            img = imread(img_path)
            if print_flat: 
                print(img_path, img.shape)
            img = downsample(img)
            save_path = os.path.join(wide_dir_target, img_name)
            imsave(save_path, img)
            if print_flat: 
                print(save_path, img.shape)
            print_flat = False

        print('================' + 'Tele' + '================')
        print_flat = 1

        for img_name in os.listdir(tele_source):
            img_path = os.path.join(tele_source, img_name)
            img = imread(img_path)
            if print_flat: print(img_path, img.shape)
            img = centerCrop(img)
            save_path = os.path.join(tele_dir_target, img_name)
            imsave(save_path, img)
            if print_flat: print(save_path, img.shape)
            print_flat = False


        # print('================' + 'Disp+Depth' + '================')
        # print_flat = 1
        # for img_name in os.listdir(disparity_source):
        #     img_path = os.path.join(disparity_source, img_name)
        #     disp = cv2.imread(img_path, -1).astype(np.float32)
        #     if print_flat: print(img_path, disp.shape)

        #     disp = (disp/64.).astype(np.float32)
        #     disp = downsample_disp(disp)
        #     disp = disp * 64.
        #     save_path = os.path.join(disp_dir_target, img_name[:-4]+'.npy')
        #     np.save(save_path, disp)
        #     if print_flat: print(save_path, disp.shape)

        #     focalLength = focalLength_dic[filename]
        #     depth = disparity_to_depth(disp, focalLength)
        #     save_path = os.path.join(depth_dir_target, img_name[:-4]+'.npy')
        #     np.save(save_path, depth)
        #     if print_flat: print(save_path, focalLength, depth.shape)

        #     print_flat = False

        # print('================' + 'AO Map' + '================')
        # print_flat = 1
        # for img_name in os.listdir(AO_source):
        #     img_path = os.path.join(AO_source, img_name)
        #     img = (cv2.imread(img_path, -1) / 65536.).astype(np.float32)
        #     if print_flat: print(img_path, img.shape)
        #     img = downsample_AO(img)
        #     img = img * 65536.
        #     save_path = os.path.join(AO_dir_target, img_name[:-4]+'.npy')
        #     np.save(save_path, img)
        #     if print_flat: print(save_path, img.shape)
        #     print_flat = False

        
        print('================' + 'Done!' + '================')
        
