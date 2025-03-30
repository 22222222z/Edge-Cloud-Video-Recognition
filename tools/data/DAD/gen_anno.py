
import os
root="/root/autodl-tmp/dataset/DAD/DAD"

train_classes_map = { 
    'normal_driving_1': 0, 
    'normal_driving_2': 0, 
    'normal_driving_3': 0, 
    'normal_driving_4': 0, 
    'normal_driving_5': 0, 
    'normal_driving_6': 0, 
    'adjusting_radio': 1, 
    'drinking': 1, 
    'messaging_left': 1, 
    'messaging_right': 1, 
    'reaching_behind': 1, 
    'talking_with_passenger': 1, 
    'talking_with_phone_left': 1, 
    'talking_with_phone_right': 1, 
}

# train data
train_anno_file = os.path.join(root, "train_anno_file.txt")
tester_num = 15
with open(train_anno_file, 'w') as f:
    for i in range(1, tester_num+1):
        tester_dir = os.path.join(root, f"Tester{i}")

        # if train_classes_map == {}:
        #     for cls_idx, cls_name in enumerate(os.listdir(tester_dir)):
        #         train_classes_map[cls_name] = cls_idx

        #     print(train_classes_map)

        for cls_name, cls_idx in train_classes_map.items():
            for img_mod in os.listdir(f"{tester_dir}/{cls_name}"):
                img_dir = f"{tester_dir}/{cls_name}/{img_mod}"
                imgs_cnt = len(os.listdir(img_dir))
                line = f"{img_dir} {imgs_cnt-1} {cls_idx}\n"
                f.write(line)
            # img_dir = f"{tester_dir}/{cls_name}/front_IR"
            # imgs_cnt = len(os.listdir(img_dir))
            # line = f"{img_dir} {imgs_cnt-1} {cls_idx}\n"
            # f.write(line)


# val data
val_anno_file = os.path.join(root, "val_anno_file.txt")
tester_num = 6
with open(val_anno_file, 'w') as f:
    for i in range(1, tester_num+1):
        tester_dir = os.path.join(root, f"val0{i}")

        # if train_classes_map == {}:
        #     for cls_idx, cls_name in enumerate(os.listdir(tester_dir)):
        #         train_classes_map[cls_name] = cls_idx

        #     print(train_classes_map)

        for iidx in range(1, 6):

            for img_mod in os.listdir(f"{tester_dir}/rec{iidx}"):
                img_dir = f"{tester_dir}/rec{iidx}/{img_mod}"
                imgs_cnt = len(os.listdir(img_dir))
                line = f"{img_dir} {imgs_cnt-1} {cls_idx}\n"
                f.write(line)
            
            # img_dir = f"{tester_dir}/rec{iidx}/front_IR"
            # imgs_cnt = len(os.listdir(img_dir))
            # line = f"{img_dir} {imgs_cnt-1} {0}\n"
            # f.write(line)
