import os
import time
import cv2
def command(i):
    
    inname = input + str(i) + "/"
    outname = output + str(i) + "/"
    if os.path.isdir(outname) is False:
        try:
            os.makedirs(outname)
        except:
            print("CREATE: " + outname + "FAILED")

    line = "python v2e.py --ignore-gooey --output_folder=" + outname \
            +" --unique_output_folder=False --overwrite --disable_slomo --output_height=800 --output_width=800 --input=" \
            + inname + " --input_frame_rate=100 --no_preview --dvs_aedat2=None --auto_timestamp_resolution=False --dvs_params=clean"
    #--auto_timestamp_resolution=True --dvs_params=clean
    print(line)
    os.system(line)
    return

input = "G:/NeRF/3-ERGB-NeRF/0-NeRF_Data/my_data_new/lego_shake_4D/r_"
output = "G:/NeRF/3-ERGB-NeRF/0-NeRF_Data/my_data_new/lego_shake_4D_event_clean/r_"

for i in range(0, 200, 2):
    print("Started-" + str(i))
    print(str(time.asctime(time.localtime(time.time()))))
    command(i)

