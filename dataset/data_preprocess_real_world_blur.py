import os.path
import cv2
import torch
import numpy as np
import shutil
from dv import AedatFile

def load_events_txt(data_dir, views_num, inter_num, height, width):

    event_map = np.zeros((views_num, inter_num - 1, height, width), dtype=np.int8) # output event_map
    frames_weights = []  # output weights of each potential sharp image

    # processing each view
    for view_counter in range(views_num):
        print("Processing View {0}".format(view_counter))

        file = data_dir + '/{0:03d}.txt'.format((view_counter))
        fp = open(file, "r")
        lines = fp.readlines()

        event_num = len(lines)                          # number of events
        exposure_start = float(lines[0].split()[0])     # exposure start time
        exposure_end = float(lines[-1].split()[0])      # exposure end time
        exposure_time = exposure_end - exposure_start   # exposure time

        event_num_counter = -1  # event number counter
        frames_time = []        # the spliting time points of event bins for the calculation of frames_weights
        bin_num_counter_pre = 0

        frames_time.append(exposure_start)
        frames_time.append(exposure_start)

        for line in lines:
            info = line.split()
            t = float(info[0])
            x = int(info[1])
            y = int(info[2])
            p = int(info[3])

            event_num_counter = event_num_counter + 1
            bin_num_counter = int(event_num_counter * (inter_num - 1) / event_num)

            # record the splitting time points
            if bin_num_counter != bin_num_counter_pre:
                #print(bin_num_counter)
                frames_time.append(t)
                bin_num_counter_pre = bin_num_counter

            # record events in event_map
            if p==1:
                event_map[view_counter][bin_num_counter][y][x] += 1
            else:
                event_map[view_counter][bin_num_counter][y][x] -= 1

        frames_time.append(exposure_end)
        frames_time.append(exposure_end)

        # calculate the frames weights of current view
        weights = []
        for i in range(inter_num):
            weights.append((frames_time[i + 2] - frames_time[i]) / 2 / exposure_time)
        frames_weights.append(weights)

    return event_map, frames_weights

def EDI_Weighted(basedir, view_counter, bin_num, events, weights, height, width):
    event_sum = torch.zeros(height, width)

    EDI = torch.ones(height, width) * weights[0]
    for i in range(bin_num):
        event_sum = event_sum + events[view_counter][i]
        EDI = EDI + torch.exp(0.3 * event_sum) * weights[i + 1]

    EDI = torch.stack([EDI, EDI, EDI], axis=-1)
    img = blurry_image / EDI
    img = torch.clamp(img, max=255)
    cv2.imwrite(basedir + "images_pose_{0}/{1:03d}.jpg".format(bin_num, view_counter * (bin_num + 1)), img.numpy())

    offset = torch.zeros(height, width)
    for i in range(bin_num):
        offset = offset + events[view_counter][i]
        imgs = img * torch.exp(0.3 * torch.stack([offset, offset, offset], axis=-1))
        cv2.imwrite(basedir + "images_pose_{0}/{1:03d}.jpg".format(bin_num, view_counter * (bin_num + 1) + 1 + i), imgs.numpy())

def torch2mask(basedir, view_num, bin_num, height, width):
    events = torch.load(basedir + "events_{0}.pt".format(bin_num)).numpy()
    masks = []
    # calculate event masks for all views
    for i in range(view_num):
        # calculate event mask for each view
        mask = False
        for j in range(bin_num):
            img = events[i][j] != 0
            mask = mask | img
        masks.append(mask.reshape(height, width))
    masks = np.stack(masks)
    np.save(basedir + "event_mask_{0}.npy".format(bin_num), masks)




# change the scene name here
scene = "toys"

basedir = "./Real-World-Blur/{0}/".format(scene)
height = 260    # resolution h
width = 346     # resolution w
views_num = 30  # number of training views



print("Stage 1: Start generating event_map and frames_weights")
for bin_num in [4, 8, 12]:
    print("Spliting event to {0} bins".format(bin_num))
    inter_num = bin_num + 1
    event_map, frames_weights = load_events_txt(basedir + "events", views_num, inter_num, height, width)
    event_map = torch.tensor(event_map).view(-1, bin_num, height * width)
    torch.save(event_map, basedir + "events_{0}.pt".format(bin_num))
    np.savetxt(basedir + "frames_weights_{0}.npy".format(bin_num), frames_weights)



print("Stage 2: Start generating images for pose estimation with colmap")
# generate the images_pose for b=4, b=8, b=12 (E2NeRF with different b)
for bin_num in [4, 8, 12]:
    os.makedirs(basedir + "images_pose_{0}".format(bin_num), exist_ok=True)
    events = torch.load(basedir + "events_{0}.pt".format(bin_num)).view(views_num, bin_num, height, width)
    weights = np.loadtxt(basedir + "frames_weights_{0}.npy".format(bin_num))
    for i in range(views_num):
        # using edi to generate the images_pose for each view
        blurry_image = torch.tensor(cv2.imread(basedir + "images/{0:03d}.jpg".format(i)), dtype=torch.float)
        EDI_Weighted(basedir, i, bin_num, events, weights[i], height, width)

# generate the images_pose for adaptive b (synthesize b=4, b=8, b=12, for Eq.(17) in the paper of E3NeRF)
os.makedirs(basedir + "images_pose", exist_ok=True)
for i in range(views_num):
    flag = 0
    for j in range(17):
        if j in [2, 6, 10, 14]:
            if scene == "corridor":
                # for the low-light corridor scene, increase the image brightness to facilitate the pose estimation
                img = cv2.imread(basedir + "images_pose_8/{0:03d}.jpg".format(int(j/2) + i * 9))
                img = img * 1.5
                img = np.clip(img, 0, 254)
                cv2.imwrite(basedir + "images_pose/{0:03d}.jpg".format(j + i * 17), img)
            else:
                shutil.copy(basedir + "images_pose_8/{0:03d}.jpg".format(int(j/2) + i * 9),
                            basedir + "images_pose/{0:03d}.jpg".format(j + i * 17))
        else:
            if scene == "corridor":
                # for the low-light corridor scene, increase the image brightness to facilitate the pose estimation
                img = cv2.imread(basedir + "images_pose_12/{0:03d}.jpg".format(flag + i * 13))
                img = img * 1.5
                img = np.clip(img, 0, 254)
                cv2.imwrite(basedir + "images_pose/{0:03d}.jpg".format(j + i * 17), img)
            else:
                shutil.copy(basedir + "images_pose_12/{0:03d}.jpg".format(flag + i * 13),
                            basedir + "images_pose/{0:03d}.jpg".format(j + i * 17))
            flag = flag + 1


print("Stage 3: Start generating event_mask for the spatial blur prior")
# generate the event_mask for different b.
for bin_num in [4, 8, 12]:
    torch2mask(basedir, views_num, bin_num, height, width)
# use b=12 event_mask for E3NeRF.
shutil.copy(basedir + "event_mask_12.npy", basedir + "event_mask.npy")

print("Finshed")