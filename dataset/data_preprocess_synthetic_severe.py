import os
import numpy as np
import cv2
import subprocess
import torch
import shutil

def rgba2rgb(img):
    r = img[:, :, 0] * 0.9
    g = img[:, :, 1] * 0.9
    b = img[:, :, 2] * 0.9
    a = img[:, :, 3] / 255
    img[:, :, 0] = 240 * (1 - a) + r * a
    img[:, :, 1] = 240 * (1 - a) + g * a
    img[:, :, 2] = 240 * (1 - a) + b * a
    return img

def cal_event_sum(event_data_path):
    event_sum_all = []
    for i in range(0, 200, 2):
        file = os.path.join(event_data_path, "r_{}/v2e-dvs-events.txt".format(i))
        fp = open(file, "r")

        event_sum = 0
        for j in range(6):
            fp.readline()

        while True:
            line = fp.readline()
            if not line:
                break

            info = line.split()
            t = float(info[0])
            x = int(info[1])
            y = int(info[2])
            p = int(info[3])

            if t >= 0.33:
                break
            event_sum += 1

        event_sum_all.append(event_sum)
        print("View {0} contains {1} events".format(i, event_sum))
    return event_sum_all

def load_event_data(data_path, event_sum_all, bin_num):
    event_map = np.zeros((100, bin_num, 800, 800), dtype=np.int8)
    poses_index_all = []              #全部视角位姿索引
    frames_weights_all = []      #全部视角帧权重

    for i in range(0, 200, 2):
        # processing each view
        file = os.path.join(data_path, "r_{}/v2e-dvs-events.txt".format(i))
        fp = open(file, "r")

        poses_index = []                    # the splitting time/pose point index array
        frame_counter = 0                   # current frame index
        poses_index.append(frame_counter)   # record the first splitting point

        event_counter = 0                   # the number of event of current event bin
        bin_counter = 0                     # current event bin index

        for j in range(6):
            fp.readline()

        while True:
            line = fp.readline()
            if not line:
                break

            info = line.split()
            t = float(info[0])
            x = int(info[1])
            y = int(info[2])
            p = int(info[3])

            if t >= 0.33:
                frame_counter += 1
                poses_index.append(frame_counter)
                break

            # with "noisy" model of v2e, we need + 0.01
            if t > (frame_counter + 1) * 0.01 + 0.01:
                frame_counter += 1

                # if the event number of current bin reach the threshold
                if event_counter >= event_sum_all[int(i/2)] / (bin_num):
                    if bin_counter < bin_num - 1:
                        poses_index.append(frame_counter)
                        bin_counter += 1
                        event_counter = event_counter - event_sum_all[int(i/2)] / (bin_num)

            if p == 0:
                event_counter += 1
                event_map[int(i / 2)][bin_counter][y][x] -= 1
            else:
                event_counter += 1
                event_map[int(i / 2)][bin_counter][y][x] += 1

        poses_index_all.append(poses_index)
        print("views {}:".format(i) + "{} events; pose_index".format(event_sum_all[int(i/2)]), poses_index)

        #frame weight version 2
        frames_weight = []
        frames_velocities = []
        for j in range(bin_num):
            frames_velocities.append(poses_index[j+1] - poses_index[j])
        frames_velocities.insert(0, 0)
        frames_velocities.append(0)
        for j in range(bin_num + 1):
            frames_weight.append((frames_velocities[j] + frames_velocities[j+1]) / 2 / 32)
        print("frames weight: ", frames_weight, sum(frames_weight))
        frames_weights_all.append(frames_weight)

    return event_map, poses_index_all, frames_weights_all

def torch2mask(events, bin_num):
    events = events.numpy()
    masks = []
    for i in range(100):
        mask_0 = False
        for j in range(bin_num):
            img = events[i][j]!= 0
            mask_0 = mask_0 | img
        masks.append(mask_0.reshape(800, 800))
    masks = np.stack(masks)
    return masks



basedir = "./Synthetic-Severe/lego/"
bin_num = 6

print("Stage 1: Start generating images for event and blurry image synthesis and gt images")

os.makedirs(basedir + "original_rgb/", exist_ok=True)
os.makedirs(basedir + "gt/", exist_ok=True)
os.makedirs(basedir + "gt_16/", exist_ok=True)
os.makedirs(basedir + "gt_novel/", exist_ok=True)
shutil.copy(basedir + "original/transforms.json", basedir + "transforms_train_blurry.json")
for i in range(0, 200):
    print("Processing view {0:03d}".format(i))
    input_path = basedir + 'original/r_{}/'.format(i)

    if i % 2 == 0:
        output_path = basedir + 'original_rgb/r_{0}/'.format(i)
        os.makedirs(output_path, exist_ok=True)
        for j in range(10, 45):
            img = cv2.imread(input_path + str(j) + ".png", cv2.IMREAD_UNCHANGED)
            img = rgba2rgb(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            cv2.imwrite(output_path + "{0:03d}.png".format(j), img)
            if j == 10:
                cv2.imwrite(basedir + "gt/{0:03d}.png".format(int(i / 2)), img)
            if j == 26:
                cv2.imwrite(basedir + "gt_16/{0:03d}.png".format(int(i / 2)), img)

    else:
        img = cv2.imread(input_path + "10.png", cv2.IMREAD_UNCHANGED)
        img = rgba2rgb(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(basedir + "gt_novel/{0:03d}.png".format(int(i / 2)), img)



print("Stage 2: Start generating blurry images for training")

os.makedirs(basedir + "train/", exist_ok=True)
for i in range(0, 200, 2):
    print("Processing view {0:03d}".format(i))
    command = "python  ./utils/blur_synthesize/main.py --input_dir {0}/original_rgb/r_{1}/".format(basedir, i) + \
              " --output_name {0}/train/r_{1}.png".format(basedir, i) + \
              " --scale_factor 33 --input_exposure 10 --input_iso 50 --output_iso 50"
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print("blurry image synthesis error: view {0:03d}".format(i))
        print(err.output)



print("Stage 3: Start generating events for training")

os.makedirs(basedir + "/original_event/", exist_ok=True)
for i in range(0, 200, 2):
    print("Processing view {0:03d}".format(i))

    input_path = basedir + "/original_rgb/r_{0}".format(i)
    output_path = basedir + "/original_event/r_{0}".format(i)
    os.makedirs(input_path, exist_ok=True)

    line = "python ./utils/v2e/v2e.py --ignore-gooey --output_folder=" + output_path \
            +" --unique_output_folder=False --overwrite --disable_slomo --output_height=800 --output_width=800 --input=" \
            + input_path + " --input_frame_rate=100 --no_preview --dvs_aedat2=None --auto_timestamp_resolution=False --dvs_params=noisy"

    os.system(line)



print("Stage 4: Start generating events.pt for training")

event_data_path = basedir + "/original_event/"

# calculate the number of events in each view
event_sum_all = cal_event_sum(event_data_path)
np.savetxt(basedir + "/event_sum_all.npy", np.array(event_sum_all))
event_sum_all = np.loadtxt(basedir + "/event_sum_all.npy")

events, poses_index_all, frames_weights_all = load_event_data(event_data_path, event_sum_all, bin_num)
events = torch.tensor(events).view(100, bin_num, 640000)
torch.save(events, basedir + "/events.pt")
np.savetxt(basedir + "/poses_index_all.npy", poses_index_all)
np.savetxt(basedir + "/frames_weights.npy", frames_weights_all)



print("Stage 5: Start generating event_mask.npy for spatial blur prior")

events = torch.load(basedir + "/events.pt")
masks = torch2mask(events, bin_num)
np.save(basedir + "/event_mask.npy", masks)