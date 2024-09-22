import sys
sys.path.append('model')

import os
import imageio
from tqdm import tqdm
import numpy as np
import glob
import torch
import time

from model.STFlow import STFlow
from datasets.DSECEITestdataloader import make_data_loader


@torch.no_grad()
def upload_STFlow_DSEC(args):

    dset = make_data_loader()
    model = STFlow()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters:", total_params)
    # ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_ed + '.pth')
    ckpt = torch.load(args.ckpt_path)
    print('Processing ', args.ckpt_path)
    model.load_state_dict(ckpt)
    model.cuda()
    model.eval()

    # voxels = glob.glob(os.path.join(args.test_path, 'test','*','*.npz'))
    # voxels.sort()
    time_list = []
    bar = tqdm(dset, total=len(dset), ncols=80)
    for voxel1, voxel2, img1, img2, city, ind in bar:
        
        
        print(voxel1.shape,voxel2.shape,img1.shape,img2.shape,city,ind)
        
        start = time.time()
        flow_up = model(voxel1.cuda(), voxel2.cuda(), img1.cuda(), img2.cuda())
        end = time.time()
        time_list.append((end-start)*1000)
        flo = flow_up[0].permute(1, 2, 0).cpu().numpy()

        uv = flo * 128.0 + 2**15
        valid = np.ones([uv.shape[0], uv.shape[1], 1])
        uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)


        test_save = args.test_save
        city = os.path.join(test_save, city[0])
        if not os.path.exists(city):
            os.makedirs(city)
        path_to_file = os.path.join(city, ind[0]+'.png')
        imageio.imwrite(path_to_file, uv, format='PNG-FI')
    avg_time = sum(time_list)/len(time_list)
    print(f'Time: {avg_time} ms.')  
    print('Done!')

      
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='upload')

    parser.add_argument('--ckpt_path', type=str, default='ckpts/')
    #save setting
    parser.add_argument('--test_path', default='')
    parser.add_argument('--test_save', default='upload/')
    args = parser.parse_args()

    upload_STFlow_DSEC(args)
