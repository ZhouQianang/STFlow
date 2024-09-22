import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import ExtractorF, ExtractorC, MixFusion
from corr import CorrBlock
from aggregate import MotionFeatureEncoder, CMA
from update import UpdateBlock
from util import coords_grid


class STFlow(nn.Module):
    def __init__(self, input_bins=15):
        super(STFlow, self).__init__()

        f_channel = 128
        self.split = 5
        self.corr_level = 1
        self.corr_radius = 3

        self.fnet_ev = ExtractorF(input_channel=input_bins//self.split, outchannel=f_channel, norm='IN')
        self.fnet_img = ExtractorF(input_channel=3+input_bins//self.split, outchannel=f_channel, norm='IN')
        self.cnet_ev = ExtractorC(input_channel=input_bins//self.split + input_bins, outchannel=256, norm='BN')
        self.cnet_img = ExtractorC(input_channel=3, outchannel=256, norm='BN')
        self.cmerge = MixFusion(input_channels=256)

        self.mfe = MotionFeatureEncoder(corr_level=self.corr_level, corr_radius=self.corr_radius)
        self.cma = CMA(d_model=128)

        self.update = UpdateBlock(hidden_dim=128, split=(self.split+1))

    def upsample_flow(self, flow, mask, scale=8):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale*H, scale*W)


    def forward(self, x1, x2, m1, m2, iters=6):
        b,_,h,w = x2.shape
        
        m1 = 2 * (m1.float().contiguous() / 255) - 1
        m2 = 2 * (m2.float().contiguous() / 255) - 1

        #Feature maps [f_0 :: f_i :: f_g]
        voxels2 = x2.chunk(self.split, dim=1)
        voxelref = x1.chunk(self.split, dim=1)[-1]
        voxels = (voxelref,) + voxels2 #[group+1] elements
        fmaps_ev = self.fnet_ev(voxels)#Tuple(f0, f1, ..., f_g)
        
        ev1 = voxelref / (torch.amax(torch.abs(voxelref),dim=(1,2,3),keepdim=True) + 0.1)
        ev2 = voxels2[-1]
        ev2 = ev2 / (torch.amax(torch.abs(ev2),dim=(1,2,3),keepdim=True) + 0.1)
        
        img1 = torch.cat([ev1, m1], dim=1)
        img2 = torch.cat([ev2, m2], dim=1)
        fmaps_img = self.fnet_img((img1,img2))

        # Context map [net, inp]
        # print(m1.shape,len(voxels2),voxels2[0].shape,m2.shape)
        cmap_ev = self.cnet_ev(torch.cat(voxels, dim=1))
        cmap_img = self.cnet_img(m1)
        cmap = self.cmerge(cmap_ev, cmap_img)
        net, inp = torch.split(cmap, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)


        coords0 = coords_grid(b, h//8, w//8, device=cmap.device)
        coords1 = coords_grid(b, h//8, w//8, device=cmap.device)

        #MidCorr
        corr_fn_list = []
        for i in range(self.split):
            corr_fn = CorrBlock(fmaps_ev[0], fmaps_ev[i+1], num_levels=self.corr_level, radius=self.corr_radius) #[c01,c02,...,c05]
            corr_fn_list.append(corr_fn)
        corr_img = CorrBlock(fmaps_img[0], fmaps_img[1], num_levels=self.corr_level, radius=self.corr_radius)

        flow_predictions = []
        for iter in range(iters):

            coords1 = coords1.detach()
            flow = coords1 - coords0

            corr_map_list = []
            du = flow/self.split 
            for i in range(self.split):
                coords = (coords0 + du*(i+1)).detach()
                corr_map = corr_fn_list[i](coords)
                corr_map_list.append(corr_map)
            corr_map = corr_img(coords)
            corr_map_list.append(corr_map)

            corr_maps = torch.cat(corr_map_list, dim=0) 

            mfs = self.mfe(torch.cat([flow]*(self.split+1), dim=0), corr_maps)
            mfs = mfs.chunk((self.split+1), dim=0)
            mfs = self.cma(mfs)
            mf = torch.cat(mfs, dim=1)
            net, dflow, upmask = self.update(net, inp, mf)
            coords1 = coords1 + dflow
            
            if self.training:
                flow_up = self.upsample_flow(coords1 - coords0, upmask)
                flow_predictions.append(flow_up)

        if self.training:
            return flow_predictions
        else:
            return self.upsample_flow(coords1 - coords0, upmask)
