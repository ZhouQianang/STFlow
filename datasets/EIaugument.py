import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import skimage
from skimage import img_as_ubyte
import torch
from typing import Tuple
from torchvision.transforms import ColorJitter
from PIL import Image

#======================================images augumentor=================================
def torch_img_to_numpy(torch_img: torch.Tensor):
    ch, ht, wd = torch_img.shape
    assert ch == 3
    numpy_img = torch_img.numpy()
    numpy_img = np.moveaxis(numpy_img, 0, -1)
    return numpy_img

def numpy_img_to_torch(numpy_img: np.ndarray):
    ht, wd, ch = numpy_img.shape
    assert ch == 3
    numpy_img = np.moveaxis(numpy_img, -1, 0)
    torch_img = torch.from_numpy(numpy_img)
    return torch_img

class PhotoAugmentor:
    def __init__(self,
                 brightness: float,
                 contrast: float,
                 saturation: float,
                 hue: float,
                 probability_color: float,
                 noise_variance_range: Tuple[float, float],
                 probability_noise: float):
        assert 0 <= probability_color <= 1
        assert 0 <= probability_noise <= 1
        assert len(noise_variance_range) == 2
        self.photo_augm = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.probability_color = probability_color
        self.probability_noise = probability_noise
        self.var_min = noise_variance_range[0]
        self.var_max = noise_variance_range[1]
        assert self.var_max > self.var_min

        self.seed = torch.randint(low=0, high=2**32, size=(1,))[0].item()

    @staticmethod
    def sample_uniform(min_value: float=0, max_value: float=1) -> float:
        assert max_value > min_value
        uni_sample = torch.rand(1)[0].item()
        return (max_value - min_value)*uni_sample + min_value

    def _apply_jitter(self, images):
        assert isinstance(images, list)

        for idx, entry in enumerate(images):
            images[idx] = self.photo_augm(entry)

        return images

    def _apply_noise(self, images):
        assert isinstance(images, list)
        variance = self.sample_uniform(min_value=0.001, max_value=0.01)

        for idx, entry in enumerate(images):
            assert isinstance(entry, torch.Tensor)
            numpy_img = torch_img_to_numpy(entry)
            noisy_img = skimage.util.random_noise(numpy_img, mode='speckle', var=variance, clip=True, seed=self.seed) # return float64 in [0, 1]
            noisy_img = img_as_ubyte(noisy_img)
            torch_img = numpy_img_to_torch(noisy_img)
            images[idx] = torch_img

        return images

    def __call__(self, images):
        if self.probability_color > torch.rand(1).item():
            images = self._apply_jitter(images)
        if self.probability_noise > torch.rand(1).item():
            images = self._apply_noise(images)
        return images
#=============================================================================================


class Augumentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.4, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        
        self.photo_augmentor = PhotoAugmentor(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.5/3.14,
            probability_color=0.2,
            noise_variance_range=(0.001, 0.01),
            probability_noise=0.2)
    
    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=0.5]
        flow0 = flow[valid>=0.5]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img
    def spatial_transform(self, voxel1, voxel2, img1, img2, flow, valid):
        ht, wd = voxel2.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)
        
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            voxel1 = cv2.resize(voxel1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            voxel2 = cv2.resize(voxel2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
            # print('Resized:', voxel1.shape, voxel2.shape, flow.shape, valid.shape)
        

        margin_y = int(round(65 * scale_y))#downside
        margin_x = int(round(35 * scale_x))#leftside

        y0 = np.random.randint(0, voxel2.shape[0] - self.crop_size[0] - margin_y)
        x0 = np.random.randint(margin_x, voxel2.shape[1] - self.crop_size[1])

        y0 = np.clip(y0, 0, voxel2.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, voxel2.shape[1] - self.crop_size[1])
        
        voxel1 = voxel1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        voxel2 = voxel2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                voxel1 = voxel1[:, ::-1]
                voxel2 = voxel2[:, ::-1]
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]
        
            if np.random.rand() < self.v_flip_prob: # v-flip
                voxel1 = voxel1[::-1, :]
                voxel2 = voxel2[::-1, :]
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                valid = valid[::-1, :]
        return voxel1, voxel2, img1, img2, flow, valid
    
    def __call__(self, voxel1, voxel2, img1, img2, flow, valid):
        voxel1, voxel2, img1, img2, flow, valid = self.spatial_transform(voxel1, voxel2, img1, img2, flow, valid)
        #==============image augumentor=============================
        # print(img1.shape,img2.shape)
        img1 = torch.from_numpy(np.moveaxis(img1.copy(), -1, 0))
        img2 = torch.from_numpy(np.moveaxis(img2.copy(), -1, 0))
        imgs = self.photo_augmentor([img1,img2])
        img1 = imgs[0]
        img2 = imgs[1]
        # print(type(img1),type(img2),img1.shape,img2.shape,"&&&&&&&&")
        # print(imgs[0].max(),"*******")
        #===========================================================
        
        voxel1 = np.ascontiguousarray(voxel1)
        voxel2 = np.ascontiguousarray(voxel2)
        # img1 = np.ascontiguousarray(imgs[0])
        # img2 = np.ascontiguousarray(imgs[1])
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)  
        
        return voxel1, voxel2, img1, img2, flow, valid      