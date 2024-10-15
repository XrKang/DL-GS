import os
import shutil

if __name__ == '__main__':
    root_path = 'E:\DLNeRF\stereoNeRF\DualCameraSynthetic'
    for dirname in os.listdir(root_path):
        dirname = os.path.join(root_path, dirname)
        target_dir = os.path.join(dirname, 'wideTele')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        wide_dir = os.path.join(dirname, 'wide')
        tele_dir = os.path.join(dirname, 'tele')
        for filename in os.listdir(wide_dir):
            wide_path = os.path.join(wide_dir, filename)
            tele_path = os.path.join(tele_dir, filename)
            target_path_W = os.path.join(target_dir, filename[:-4] + '_W.jpg')
            target_path_T = os.path.join(target_dir, filename[:-4] + '_T.jpg')
            shutil.copy(wide_path, target_path_W)
            shutil.copy(tele_path, target_path_T)
            print(wide_path, target_path_W)
            print(tele_path, target_path_T)