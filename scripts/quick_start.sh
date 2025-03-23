# conda create --name CENet python=3.8 -y
# conda activate CENet
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
# mim install mmcv
pip install mmcv==2.1.0
mim install pytorchvideo
# mim install mmdet  # optional
# mim install mmpose  # optional
# git clone https://github.com/open-mmlab/mmaction2.git
# cd mmaction2
pip install -v -e .