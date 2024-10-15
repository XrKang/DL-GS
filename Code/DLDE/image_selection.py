from scene import Scene
import numpy as np
import torch
import  glob 
import os
import math

def image_selection(render_veiw, train_veiws):
    n0  = np.matmul(render_veiw.R,np.array([[0,0,1]]).T).T
    n0 = n0/np.linalg.norm(n0)
    t0 = render_veiw.T

    c2c_dist = []
    normal_dot = []
    train_veiws = sorted(train_veiws, key = lambda x : x.image_name)
    for train_veiw in train_veiws:
        n1 = np.matmul(train_veiw.R,np.array([[0,0,1]]).T).T
        n1 = n1/np.linalg.norm(n1)
        t1 = train_veiw.T
        normal_dot.append( 1.0 - (n0 * n1).sum())
        c2c_dist.append( math.sqrt( ((t0 - t1)**2).sum() ))
    c2c_dist = np.array(c2c_dist)
    normal_dot = np.array(normal_dot)
        
    weights = c2c_dist + 50*normal_dot
    score = torch.from_numpy(weights)
    score = 1.0/(score+1e-6)
    # print('score',score.shape)
    _, ind_topK = torch.topk(score, k=2,dim=0) # [B, H] score.size()[0]
    ind_topK = ind_topK.numpy().astype(np.int64)
    W_ref_0, T_ref_0 = train_veiws[ind_topK[0]].original_image[0:3, :, :], train_veiws[ind_topK[0]].tele_image[0:3, :, :]
    W_ref_1, T_ref_1 = train_veiws[ind_topK[1]].original_image[0:3, :, :], train_veiws[ind_topK[1]].tele_image[0:3, :, :]

    return W_ref_0, T_ref_0, W_ref_1, T_ref_1