import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob

import imageio as iio


class DSECEITestdataset(data.Dataset):
    def __init__(self, augument=True):
        super(DSECEITestdataset, self).__init__()
        self.init_seed = False
        
        self.events_files = []
        self.flow_files = []
        self.image1_files = []
        self.image2_files = []

        self.events_root = '/data/DSEC-Flow/DSEC_Event_v0_15bins/test'
        self.images_root = '/data/DSEC-Flow/DSEC_Image/test/'
        
        # load all scenes
        scenes = [d for d in os.listdir(self.events_root) if os.path.isdir(os.path.join(self.events_root, d))]

        for scene in scenes:
            flow_ts = np.loadtxt(os.path.join(self.images_root,scene,'flow/forward_timestamps.txt'),delimiter=',', skiprows=1)
            images_ts = np.loadtxt(os.path.join(self.images_root,scene,'images/timestamps.txt'))
    
            for i, flowt in enumerate(flow_ts):

                idx = int(flow_ts[i][2])
        
                events_file = os.path.join(self.events_root,scene,f'{idx:06d}.npz')
                assert os.path.exists(events_file), f"The file {events_file} not exist."
        
                idx1 = np.where(images_ts == flow_ts[i][0])
                idx2 = np.where(images_ts == flow_ts[i][1])
                assert idx2[0].tolist()[0] - idx1[0].tolist()[0] == 2
        
                image1_file = os.path.join(self.images_root,scene,f'images/left/ev_inf/{idx1[0][0]:06d}.png')
                assert os.path.exists(image1_file), f"The file {image1_file} not exist."
                image2_file = os.path.join(self.images_root,scene,f'images/left/ev_inf/{idx2[0][0]:06d}.png')
                assert os.path.exists(image2_file), f"The file {image2_file} not exist."
        
                self.events_files.append(events_file)
                self.image1_files.append(image1_file)
                self.image2_files.append(image2_file)
        
        print('There has (', len(self.events_files),len(self.flow_files),len(self.image1_files),len(self.image2_files),1361*6,') samples in training')

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        eventfile = self.events_files[index]
        city = eventfile.split('/')[-2]
        ind = eventfile.split('/')[-1].split('.')[0].split('_')[-1]
        
        voxel_file = np.load(self.events_files[index])
        voxel1 = voxel_file['voxel_prev']#.transpose([1,2,0])
        voxel2 = voxel_file['voxel_curr']#.transpose([1,2,0])
        
        img1 = np.asarray(iio.imread(self.image1_files[index], format='PNG-FI'))
        img2 = np.asarray(iio.imread(self.image2_files[index], format='PNG-FI'))
        
        voxel1 = torch.from_numpy(voxel1).permute(2, 0, 1).float()
        voxel2 = torch.from_numpy(voxel2).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        return voxel1, voxel2, img1, img2, city, ind
    
    def __len__(self):
        return len(self.events_files)


def make_data_loader(batch_size=1):
    dset = DSECEITestdataset()
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=4)
    return loader
