import os
import shutil
from PIL import Image
import numpy as np
import re

def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else '' for c in sub_strings]
    return sub_strings

if __name__ == '__main__':
    root_path = r'/home/ustc/data/DLNeRF/NeRFStereo/DualCameraSynthetic'
    n_views = 10
    # n_views = 20
    # n_views = 90

    dirnames = sorted(os.listdir(root_path))
    print(dirnames)

    for dirname in dirnames:
        dirname_full = os.path.join(root_path, dirname)
        print("Processing:", dirname_full)
        # ------------------------------
        shutil.rmtree(os.path.join(dirname_full, 'wide_trainTest_' + str(n_views)))
        # ------------------------------

        gt_dir = os.path.join(dirname_full, 'GT')
        wide_dir = os.path.join(dirname_full, 'wide')
        tele_dir = os.path.join(dirname_full, 'tele') # tele放在另一个文件夹
        target_wide_HR_GT = os.path.join(dirname_full, 'wide_trainTest_' + str(n_views) , 'HR_GT')
        target_wide_HR_GT_test = os.path.join(dirname_full, 'wide_trainTest_' + str(n_views) , 'HR_GT_test')
        target_wide_TrainTest = os.path.join(dirname_full, 'wide_trainTest_' + str(n_views) , 'images')
        target_wide_Train = os.path.join(dirname_full, 'wide_trainTest_' + str(n_views) , 'wide_trainForPC')
        target_tele = os.path.join(dirname_full, 'wide_trainTest_' + str(n_views) , 'tele')
        

        if not os.path.exists(os.path.join(dirname_full, 'wide_trainTest_' + str(n_views))):
            os.makedirs(os.path.join(dirname_full, 'wide_trainTest_' + str(n_views)))
        if not os.path.exists(target_wide_HR_GT):
            os.makedirs(target_wide_HR_GT)
        if not os.path.exists(target_wide_HR_GT_test):
            os.makedirs(target_wide_HR_GT_test)
        if not os.path.exists(target_wide_TrainTest):
            os.makedirs(target_wide_TrainTest)
        if not os.path.exists(target_wide_Train):
            os.makedirs(target_wide_Train)
        if not os.path.exists(target_tele):
            os.makedirs(target_tele)
        test_names = []
        train_names = []
        train_names_temp = []

        image_names = os.listdir(wide_dir)
        image_names = sorted(image_names, key=natural_sort_key)
        for idx, image_name in enumerate(image_names):
            if idx % 10 ==0:
                test_names.append(image_name)
            else:
                train_names_temp.append(image_name)
        train_names_temp = sorted(train_names_temp, key=natural_sort_key)
        if n_views > 0:
            idx_sub = np.linspace(0, len(train_names_temp)-1, n_views)
            idx_sub = [round(i) for i in idx_sub]
            train_names = [c for idx, c in enumerate(train_names_temp) if idx in idx_sub]
            assert len(train_names) == n_views
        else:
            print("Given n_views is not valid!")
            break
        print("Test Images", len(test_names), test_names)
        print("Train Images", len(train_names), train_names)
        for image_name in test_names:
            shutil.copy(os.path.join(wide_dir, image_name), 
                            os.path.join(target_wide_TrainTest, image_name.split('.')[0] + '_test'+ '.jpg'))
            shutil.copy(os.path.join(gt_dir, image_name), 
                            os.path.join(target_wide_HR_GT, image_name.split('.')[0] + '_test'+ '.jpg'))
            shutil.copy(os.path.join(gt_dir, image_name), 
                            os.path.join(target_wide_HR_GT_test, image_name.split('.')[0] + '_test'+ '.jpg'))
        for image_name in train_names:
            shutil.copy(os.path.join(wide_dir, image_name), 
                    os.path.join(target_wide_TrainTest, image_name.split('.')[0] + '_train' + '.jpg'))
            shutil.copy(os.path.join(wide_dir, image_name), 
                    os.path.join(target_wide_Train, image_name.split('.')[0] + '_train' + '.jpg'))
            shutil.copy(os.path.join(gt_dir, image_name), 
                            os.path.join(target_wide_HR_GT, image_name.split('.')[0] + '_train'+ '.jpg'))
        for image_name in train_names:
            shutil.copy(os.path.join(tele_dir, image_name), 
                            os.path.join(target_tele, image_name.split('.')[0] + '_trainTele' + '.jpg'))


        print()