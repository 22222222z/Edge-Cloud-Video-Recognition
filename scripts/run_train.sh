# source~/.bashrc
# conda activate CENet

root_dir=/root/Projects/CENet
config=/root/Projects/CENet/configs/recognition/tsm/tsm_attnlast.py
# config=/root/Projects/CENet/configs/recognition/uniformerv2-cloud_model/raw_model_stage_2_feat.py
# config=/root/Projects/CENet/configs/recognition/uniformerv2-cloud_model/cloud_model_base.py
# config=/root/Projects/CENet/configs/recognition/tsm/tsm_attnlast_DAD.py
# config=/root/Projects/CENet/configs/recognition/tsm/tsm_DAD.py

config=/root/Projects/CENet/configs/recognition/uniformerv2-cloud_model/uniformerv2_CENet_DAD.py

# cpu train
# CUDA_VISIBLE_DEVICES=-1 python $root_dir/tools/train.py $config > $root_dir/scripts/run_train.log

# gpu train
CUDA_VISIBLE_DEVICES=0 python $root_dir/tools/train.py $config > $root_dir/scripts/run_train.log

# dist train
# CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh $config 2 > $root_dir/scripts/run_train.log