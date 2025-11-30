from utils.poses.pose_utils import gen_poses
import shutil

if __name__=='__main__':
    datadir = './Real-World-Blur/lego/'
    gen_poses(datadir + "images_pose/", "exhaustive_matcher")
    shutil.copy(datadir + "images_pose/poses_bounds.npy", datadir + "poses_bounds.npy")