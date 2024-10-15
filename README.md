# DL-GS

> [High-Resolution and Few-shot View Synthesis from Asymmetric Dual-lens Inputs](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00368.pdf), In *ECCV* 2024.
> Ruikang Xu, Mingde Yao, Yue Li, Yueyi Zhang, Zhiwei Xiong.


****

## Dependencies
* Our Environment is Bulid on [the Docker Image from the INRIA lab](https://hub.docker.com/r/gaetanlandreau/3d-gaussian-splatting).
* Other Dependencies: BasicSR 1.3.4.9, OpenCV 4.7.0, Scikit-image, CuPy, Open3d, Pillow, Imageio, COLMAP. 
* Compile CUDA:
  ```
  cd ./Code
  pip install submodules/diff-gaussian-rasterization-confidence
  pip install submodules/simple-knn
  ```
****


## Data Preparation
* The StereoNeRF dataset can be downloaded from this [link](https://github.com/Haechan21/StereoNeRF).
* Simulated Dual-lens Scenes:
  ```
  cd ./SimulatedData && python dualLensSyn.py && python combinWideTele.py
  ```
* Split Training and Test views:
  ```
  cd ./SimulatedData && python split_TrainTest.py
  ```

## Quick Start
#### 1. Consistency-aware Training
* Pre-upsample (please download the pretrained [HAT](https://github.com/XPixelGroup/HAT/tree/main) for 2x SR):
  ```
  cd ./Code/SISR && python test.py -opt HAT-S_SRx2_SISR.yml
  ```
* Run COLMAP for Camera Pose Estimation with Sparse Views and Stereo-fusion-based Initialization:
  ```
  cd ./Code/colmap_sh && sh colmap.sh
  ```
* Training with Two Designed Loss Functions (please download the pretrained [MiDas](https://github.com/isl-org/MiDaS)):
  ```
  cd ./Code && sh train_gs.sh
  ```
* Rendering Gaussians:
  ```
  cd ./Code && sh render_gs.sh
  ```
#### 2. Multi-reference-guided Refinement
* Pre-alignment Telephoto Images to Wide-angle Images:
  ```
  cd ./Code/alignTele && sh align.sh
  ```
* Training with Self-learning Loss Functions:
  ```
  cd ./Code && sh train_dlde.sh
  ```
#### 3. Rendering Full Pipeline:
  ```
  cd ./Code && sh render_full.sh
  ```





****
## TODO List:
We will release our code for the real-captured dataset in the future.

****

## Contact
Any question regarding this work can be addressed to xurk@mail.ustc.edu.cn.

****


## Citation
If you find our work helpful, please cite the following paper.
```
@inproceedings{Xu_2024_ECCV,
  title={High-Resolution and Few-shot View Synthesis from Asymmetric Dual-lens Inputs},
  author={Xu, Ruikang and Yao, Mingde and Yue, Li and Yueyi, Zhang and Xiong, Zhiwei},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}
```