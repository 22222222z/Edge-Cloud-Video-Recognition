import os

log_path = "/root/Projects/CENet/tools/vis_cam_2.log"
path_prefix = "/root/autodl-tmp/visualizations/cam_outputs/last_conv_svd"

with open(log_path, "r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    # print(lines[i])
    # breakpoint()
    if lines[i] != "inference res: False\n":
        continue

    path_ele = lines[i-2].split(" ")
    gif_path = os.path.join(path_prefix, path_ele[0], path_ele[1], path_ele[2].split("\n")[0])
    print(gif_path)

