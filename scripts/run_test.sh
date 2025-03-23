# source~/.bashrc
# conda activate CENet

root_dir=/root/Projects/CENet
config=/root/Projects/CENet/configs/recognition/tsm/tsm_attnlast.py
# config=/root/Projects/CENet/configs/recognition/uniformerv2-cloud_model/raw_model_stage_2_feat.py
# config=/root/Projects/CENet/configs/recognition/uniformerv2-cloud_model/cloud_model_base.py
config=/root/Projects/CENet/configs/recognition/tsm/tsm_attnlast_DAD.py
# config=/root/Projects/CENet/configs/recognition/tsm/tsm_DAD.py

checkpoint=/root/Projects/CENet/work_dirs/tsm_attnlast_DAD/epoch_50.pth

# cpu test
# CUDA_VISIBLE_DEVICES=-1 python $root_dir/tools/test.py $config $checkpoint > $root_dir/scripts/run_test.log

# gpu test
CUDA_VISIBLE_DEVICES=0 python $root_dir/tools/test.py $config $checkpoint > $root_dir/scripts/run_test.log

# dist test
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh $config $checkpoint 2 > $root_dir/scripts/run_test.log