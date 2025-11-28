import torch
import math
import numpy as np
import cv2
import random
import os

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))
    #rounding = 1e8
    #y = torch.round(y*rounding)/rounding
    return y.float()

# for synthetic blender data
def event_loss_call_synthetic(all_rgb, event_data, rgb2grey, bin_num):
    loss = []
    for its in range(bin_num):
        start = its
        end = its + 1
        thres = (lin_log(torch.mv(all_rgb[end], rgb2grey) * 255) - lin_log(torch.mv(all_rgb[start], rgb2grey) * 255)) / 0.25
        event_cur = event_data[start]
        loss.append(torch.mean((thres - event_cur) ** 2))
    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss

# for real-world ellff data
def event_loss_call_real(all_rgb, event_data, rgb2grey, bin_num):
    loss = []
    for its in range(bin_num):
        start = its
        end = its + 1
        thres = (torch.log(torch.mv(all_rgb[end], rgb2grey) * 255) - torch.log(torch.mv(all_rgb[start], rgb2grey) * 255)) / 0.3
        event_cur = event_data[start]
        loss.append(torch.mean((thres - event_cur) ** 2))
    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss



class bin_num_eval(object):
    '''

    '''

    def __init__(self, views_num, dir, parent=None):
        self.path = dir
        if os.path.exists(dir + "/bin_num_np.npy"):
            self.bin_num_np = np.load(dir + "/bin_num_np.npy")
            self.bin_num_np_counter = np.load(dir + "/bin_num_np_counter.npy")
            self.bin_num_np_ave = np.load(dir + "/bin_num_np_ave.npy")
        else:
            self.bin_num_np = np.zeros((views_num))
            self.bin_num_np_counter = np.zeros((views_num))
            self.bin_num_np_ave = np.zeros((views_num))

    def update(self, img_i, offset, s):
        self.bin_num_np[img_i] += offset
        self.bin_num_np_counter[img_i] += s
        self.bin_num_np_ave[img_i] = self.bin_num_np[img_i] / self.bin_num_np_counter[img_i]
        #print(img_i, offset / s)

    def get_bin_flag(self, img_i):
        if self.bin_num_np_ave[img_i] <= 10:
            return 0
        elif self.bin_num_np_ave[img_i] <= 15:
            return 1
        else:
            return 2

    def get_bin_flag_ellff(self, img_i):
        if self.bin_num_np_ave[img_i] <= 5:
            return 0
        elif self.bin_num_np_ave[img_i] <= 10:
            return 1
        else:
            return 2

    def save(self):
        np.save(self.path + "/bin_num_np.npy", self.bin_num_np)
        np.save(self.path + "/bin_num_np_counter.npy", self.bin_num_np_counter)
        np.save(self.path + "/bin_num_np_ave.npy", self.bin_num_np_ave)

    def mean_pixels_offset_cal(self, pose_w2c, K, depthes, select_blur_coords, rays_os, rays_ds, img_i, near, far):

        flag = (depthes[0] >= near) & (depthes[0] <= far)
        for i in range(1, len(depthes) - 1):
            flag = (depthes[i] >= near) & (depthes[i] <= far) & flag
        s = flag.sum().item()
        flag = flag.view(-1)

        offsets = 0
        x0, y0 = select_blur_coords[:, 0], select_blur_coords[:, 1]
        for i in range(len(depthes) - 1):
            x, y = self.pixels_offset_estimate(
                rays_os[i][select_blur_coords[:, 0], select_blur_coords[:, 1]],
                rays_ds[i][select_blur_coords[:, 0], select_blur_coords[:, 1]],
                rays_os[i + 1][select_blur_coords[:, 0], select_blur_coords[:, 1]],
                torch.Tensor(pose_w2c[i + 1]), torch.Tensor(K), depthes[i])
            #print(x[0] - x0[0], y[0] - y0[0])
            offset = torch.sum(torch.sqrt((x - x0) ** 2 + (y - y0) ** 2) * flag)
            offsets += offset.item()
        self.update(img_i, offsets / len(depthes), s)
        return

    # 两帧间像素偏移值计算
    def pixels_offset_estimate(self, o0, d0, o1, w2c, K, depth):
        '''
        输入
        给定基准相机位姿： o0, d0
        给定偏移相机位姿： o1
        给定偏移相机w2c： w2c
        给定相机内参： K
        给定深度值： depth

        输出
        下一帧 x 坐标
        下一帧 y 坐标
        '''
        depth = depth.view(-1)
        p = o0 + torch.stack([d0[:, 0] * depth, d0[:, 1] * depth, d0[:, 2] * depth], dim=-1)
        xxx = torch.norm((p - o1), p=2, dim=1)
        zzz = torch.t(xxx.repeat(3, 1))
        #zz = (p - o1)[:,2]
        #zzz = torch.t(zz.repeat(3,1))
        new_d = (p - o1).div(zzz)
        c_d = torch.matmul(new_d, torch.t(w2c))
        #c_d_new = torch.t(torch.matmul(w2c, torch.t(new_d)))
        yyy = torch.t(c_d[:,2].expand(3,-1))
        c_d = -c_d / yyy
        y, x = c_d[:, 0] * K[0][0] + K[0][2], -(c_d[:,1] * K[1][1]) + K[1][2]
        return x, y