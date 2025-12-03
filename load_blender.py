import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_blender_data(basedir, bin_num):

    with open(os.path.join(basedir, 'transforms_train_blurry.json'), 'r') as fp:
        meta = json.load(fp)

    # split time/pose points
    poses_num = np.loadtxt(os.path.join(basedir, 'poses_num.npy'))

    train_imgs = []
    train_poses = []
    test_poses = []
    test_poses_novel = []

    num = 0
    for frame in meta['frames'][0:200:2]:
        fname = os.path.join(basedir, "./train/r_{}.png".format(num))
        train_imgs.append(imageio.imread(fname))

        train_pose = []
        for i in range(bin_num + 1):
            train_pose.append(frame['transform_matrix'][int(poses_num[int(num / 2)][i])])
        train_poses.append(np.array(train_pose))

        test_poses.append(np.array(frame['transform_matrix'][0]))
        num = num + 2

    for frame in meta['frames'][1:200:2]:
        test_poses_novel.append(np.array(frame['transform_matrix'][0]))

    train_imgs = (np.array(train_imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
    train_poses = np.array(train_poses).astype(np.float32)
    test_poses = np.array(test_poses).astype(np.float32)
    test_poses_novel = np.array(test_poses_novel).astype(np.float32)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 180 + 1)[:-1]], 0).cpu().numpy()

    H, W = train_imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    return train_imgs, train_poses, render_poses, test_poses, test_poses_novel, [H, W, focal]