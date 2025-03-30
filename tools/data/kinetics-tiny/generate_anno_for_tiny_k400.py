import os
import os.path as osp

dataset_root = '/root/autodl-tmp/dataset/K400/tiny-Kinetics-400'

class_names = [f for f in os.listdir(dataset_root) if not f.startswith('.')]
annos = []
for idx, class_name in enumerate(class_names):
    dir_name = osp.join(dataset_root, class_name)
    if not osp.isdir(dir_name):
        continue
    
    videos = [f for f in os.listdir(dir_name) if not f.startswith('.')]
    for video in videos:

        if '.mp4' not in video:
            continue

        video_path = osp.join(dataset_root, class_name, video)
        annos.append(f'{video_path} {idx}\n')

with open(osp.join(dataset_root, 'train_anno.txt'), 'w') as f:
    for line in annos:
        f.write(line)

'''
    label,youtube_id,time_start,time_end,split,is_cc
'''
# class_names = [f for f in os.listdir(dataset_root) if not f.startswith('.')]
# annos = []
# for class_name in class_names:
#     dir_name = osp.join(dataset_root, class_name)
#     videos = [f for f in os.listdir(dir_name) if not f.startswith('.')]
#     for video in videos:

#         if '.mp4' not in video:
#             continue

#         # print(class_name, video, video.split('.')[0].split('_'))
#         time_start, time_end = video.split('.')[0].split('_')[-2], video.split('.')[0].split('_')[-1]
#         youtube_id = video.split('_'+time_start)[0]
#         annos.append([class_name, youtube_id, int(time_start), int(time_end)])

# with open(osp.join(dataset_root, 'train_anno.txt'), 'w') as f:
#     f.write('label,youtube_id,time_start,time_end,split,is_cc\n')
#     split = 'train'
#     is_cc = 0
#     for label, youtube_id, time_start, time_end in annos:
#         f.write(f'{label},{youtube_id},{time_start},{time_end},{split},{is_cc}\n')