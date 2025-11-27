# E<sup>3</sup>NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images
## Code
The new code for E<sup>3</sup>NeRF will be released soon.
## Dataset

### Real-World-Challenge Dataset
We provide both original data and processed data for E3NEeRF training. Please down load them [here](https://drive.google.com/file/d/1lDxf6mAvgNfWm0XIXT7BCji9tSVjyBDS/view?usp=sharing).

**Original Data:** 
The original data consist of three folders in each scene's folder. "images"-The blurry images for training; "events"-The events corresponding to the blurry images; "gt"-The ground truth sharp images for testing.

**Processed Data:**
Compared to Original Data, Processed Data includes events.pt, frames_weights.npy, event_mask.npy, and pose_bounds.npy, which facilitates the event loss calculation, provides temporal and spatial blur prior, poses for training and testing, respectively.

**Generate Processd Data with Original Data:**
Please first run data/data_preprocess_real.py to generate the events.pt, frames_weights.npy, event_mask.npy, and images for pose estimation. Then use colmap to estimate the poses of images in the "images_pose" folder. Finally run data/data_preprocess_real_imgs2pose.py to generate the pose_bounds.npy for training.


