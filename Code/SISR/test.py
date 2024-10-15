# flake8: noqa
import os.path as osp

import archs
import data
import models
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    print(root_path)
    test_pipeline(root_path)

    import datetime
    oldtime = datetime.datetime.now()
    for i in range(10):
        test_pipeline(root_path)
    newtime = datetime.datetime.now()
    print('Time consuming: ', (newtime - oldtime)/10)
    # Time consuming for 10 images: 1.8s
    # Time consuming for 1 images: 0.18s
