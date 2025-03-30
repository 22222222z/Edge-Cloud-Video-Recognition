# source~/.bashrc
# conda activate CENet

root_dir=/root/Projects/Edge-Cloud-Video-Recognition
config=$root_dir/configs/recognition/tsm/tsm_attnlast.py
checkpoint=$root_dir/work_dirs/tsm_attnlast_DAD/epoch_50.pth

# cpu test
# CUDA_VISIBLE_DEVICES=-1 python $root_dir/tools/test.py $config $checkpoint > $root_dir/scripts/run_test.log

# gpu test
CUDA_VISIBLE_DEVICES=0 python $root_dir/tools/test.py $config $checkpoint > $root_dir/scripts/run_test.log

# dist test
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh $config $checkpoint 2 > $root_dir/scripts/run_test.log