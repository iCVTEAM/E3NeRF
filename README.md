# Code for E<sup>3</sup>NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images
This is an official PyTorch implementation of the E<sup>3</sup>NeRF. Click [here](https://icvteam.github.io/E3NeRF.html) to see the video and supplementary materials in our project website.

## Method Overview



## Code



## Dataset

### Real-World-Challenge Dataset
We provide both original data and processed data for E<sup>3</sup>NEeRF training. Please down load them [here](https://drive.google.com/drive/folders/1iy0266P29K3O2PexX1obhtEPo4fqBKDY?usp=sharing).

**Original Data:** 
The original data consist of three folders in each scene's folder. "images"-The blurry images for training; "events"-The events corresponding to the blurry images; "gt"-The ground truth sharp images for testing.

**Processed Data:**
Compared to Original Data, Processed Data includes events.pt, frames_weights.npy, event_mask.npy, and pose_bounds.npy, which facilitates the event loss calculation, provides temporal and spatial blur prior, provides poses for training and testing, respectively.

**Generate Processd Data with Original Data:**
Please first run data/data_preprocess_real.py to generate the events.pt, frames_weights.npy, event_mask.npy, and images for pose estimation. Then use colmap to estimate the poses of images in the "images_pose" folder. Finally run data/data_preprocess_real_imgs2pose.py to generate the pose_bounds.npy for training.

## Citation

If you find this useful, please consider citing our paper:

```bibtex
@inproceedings{qi2023e2nerf,
  title={E2NeRF: Event enhanced neural radiance fields from blurry images},
  author={Qi, Yunshan and Zhu, Lin and Zhang, Yu and Li, Jia},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13254--13264},
  year={2023}
}
@article{qi2024e3nerf,
  title={E3NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images},
  author={Qi, Yunshan and Li, Jia and Zhao, Yifan and Zhang, Yu and Zhu, Lin},
  journal={arXiv preprint arXiv:2408.01840},
  year={2024}
}
```



## Acknowledgment

The overall framework are derived from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/). We appreciate the effort of the contributors to these repositories.
