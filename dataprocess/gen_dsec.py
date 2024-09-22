from event_slicer import EventSlicer
import h5py
import hdf5plugin
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import imageio
from taskmanager import TaskManager

import torch
import torch.utils.data as data

from representation import VoxelGrid

myrepresentation = VoxelGrid((15, 480, 640), normalize=True)

def events_to_voxel_grid(x, y, p, t):
    
    t = (t - t[0]).astype('float32')
    # print(len(t),"******",t[0],t[-1])
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(pol),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
        }
    return myrepresentation.convert(event_data_torch)

def rectify_and_write(rect_map, events_curr, events_prev, output_dir, idx):
    #rectify events
    p = events_curr['p']
    t = events_curr['t']
    x = events_curr['x']
    y = events_curr['y']
    xy_rect = rect_map[y, x]
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]
    # events1 = np.stack([x_rect, y_rect, t, p], axis=1)
    voxel2 = events_to_voxel_grid(x_rect, y_rect, p, t).permute(1, 2, 0).numpy()

    p = events_prev['p']
    t = events_prev['t']
    x = events_prev['x']
    y = events_prev['y']
    xy_rect = rect_map[y, x]
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]
    # events0 = np.stack([x_rect, y_rect, t, p], axis=1)
    voxel1 = events_to_voxel_grid(x_rect, y_rect, p, t).permute(1, 2, 0).numpy()
    
    #write events
    output_name = os.path.join(output_dir, '{:06d}'.format(idx))
    if os.path.exists(output_name):
        return
    np.savez(output_name, voxel_prev=voxel1, voxel_curr=voxel2)

if __name__ == '__main__':
    
    event_path = '/data/DSEC/rawdata_events/train_events'
    flow_path = '/data/DSEC/rawdata_events/train_optical_flow'
    dst_path = '/data/DSEC-Flow/DSEC_EventVoxel/train'

    sequences = os.listdir(flow_path)
    for seq in sequences:
        event_h5 = os.path.join(event_path, seq, 'events/left', 'events.h5')
        rectify_h5 = os.path.join(event_path, seq, 'events/left', 'rectify_map.h5')

        ts_file = os.path.join(flow_path, seq, 'flow', 'forward_timestamps.txt')
        flow_dir = os.path.join(flow_path, seq, 'flow', 'forward') 
        flow_list = sorted(glob(os.path.join(flow_dir, '*.png')))
        timestamps = np.genfromtxt(ts_file, delimiter=',')
        assert timestamps.shape[0]== len(flow_list)
        
        h5f = h5py.File(event_h5, 'r')
        slicer = EventSlicer(h5f)
        with h5py.File(rectify_h5, 'r') as h5_rect:
            rectify_map = h5_rect['rectify_map'][()]



        output_dir = os.path.join(dst_path, seq)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with TaskManager(total=len(timestamps), processes=8, queue_size=4, use_pbar=True, pbar_desc=f"Processing {seq}") as tm:

            for i in tqdm(range(len(timestamps)), ncols=60):
                #current
                t_curr = timestamps[i,0]
                t_next = timestamps[i,1]#+100*1000
                events_curr = slicer.get_events(t_curr, t_next)
                if events_curr == None:
                    print(f'None data can be converted to voxel in {seq} at {i}th timestamps for current condition!')
                    continue

                # previous 
                dt = 100 * 1000#us
                t_prev = t_curr - dt
                # t_prev = timestamps[i-1,0]
                events_prev = slicer.get_events(t_prev, t_curr)
                if events_prev == None:
                    print(f'None data can be converted to voxel in {seq} at {i}th timestamps for previous condition!')
                    continue
                rectify_and_write(rectify_map, events_curr, events_prev, output_dir, i)

                flow_16bit = imageio.imread(flow_list[i], format='PNG-FI')
                np.save(os.path.join(output_dir, 'flow_{:06d}.npy'.format(i)), flow_16bit)
        h5f.close()