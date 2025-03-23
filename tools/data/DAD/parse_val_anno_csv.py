
import csv
import os

csv_path = '/root/autodl-tmp/dataset/DAD/DAD/LABEL.csv'
root_path = '/root/autodl-tmp/dataset/DAD/DAD'
views = ['front_depth', 'front_IR', 'top_depth', 'top_IR']

with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[-1] == '':
            continue
        if row[0] != '':
            which_val_path = os.path.join(root_path, row[0].strip())
        if row[1] != '':
            video_path_prefix = os.path.join(which_val_path, row[1])

        video_begin = int(row[2])
        video_end = int(row[3])

        for view in views:
            video_path = os.path.join(video_path_prefix, view)
            if row[4] == 'N':
                label = 1
            elif row[4] == 'A':
                label = 0
            print(video_path, video_begin, video_end, label)
            continue
            clips = get_clips(video_path, video_begin, video_end, label, view, sample_duration)
            dataset = dataset + clips