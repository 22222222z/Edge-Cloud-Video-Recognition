# source~/.bashrc
# conda activate CENet

root_dir=/root/Projects/Edge-Cloud-Video-Recognition

export PYTHONPATH=$root_dir:$root_dir/mmaction:$PYTHONPATH

# knetics-400
config=$root_dir/configs/recognition/tsm/edge_model_k400.py
config=$root_dir/configs/recognition/tsm/edge_model_k400_distill.py
config=$root_dir/configs/recognition/uniformerv2-cloud_model/uniformerv2_ECNet_K400.py

# DAD
config=$root_dir/configs/recognition/tsm/edge_model_DAD.py
config=$root_dir/configs/recognition/tsm/edge_model_DAD_distill.py
config=$root_dir/configs/recognition/uniformerv2-cloud_model/uniformerv2_ECNet_DAD.py

# cpu train
# CUDA_VISIBLE_DEVICES=-1 python $root_dir/tools/train.py $config 2>&1 | tee $root_dir/scripts/run_train.log
# CUDA_VISIBLE_DEVICES=-1 python $root_dir/tools/train.py $config > $root_dir/scripts/run_train.log

# gpu train
CUDA_VISIBLE_DEVICES=0 python $root_dir/tools/train.py $config > $root_dir/scripts/run_train.log

# dist train
# CUDA_VISIBLE_DEVICES=0,1 bash $root_dir/tools/dist_train.sh $config 2 > $root_dir/scripts/run_train.log