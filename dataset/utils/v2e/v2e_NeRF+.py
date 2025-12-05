import os
import time

'''
使用v2e默认参数模拟事件产生
延用E2NeRF设计
'''

def command(inname, outname):

    if os.path.isdir(outname) is False:
        try:
            os.makedirs(outname)
        except:
            print("CREATE: " + outname + "FAILED")

    line = "python v2e.py --ignore-gooey --output_folder=" + outname \
            +" --unique_output_folder=False --overwrite --disable_slomo --output_height=800 --output_width=800 --input=" \
            + inname + " --input_frame_rate=100 --no_preview --dvs_aedat2=None --auto_timestamp_resolution=False --dvs_params=noisy"
    #--auto_timestamp_resolution=True --dvs_params=clean
    print(line)
    os.system(line)
    return

input = "G:/NeRF/3-ERGB-NeRF/23-dataset-new/blurry/materials_5D/r_"
output = "G:/NeRF/3-ERGB-NeRF/23-dataset-new/event/materials_5D/r_"

for i in range(0, 200, 2):
    print("Started-{}".format(i))
    print(str(time.asctime(time.localtime(time.time()))))

    inname = input + str(i) + "/"
    outname = output + str(i) + "/"

    command(inname, outname)

