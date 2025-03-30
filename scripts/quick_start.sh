conda create --name ECNet python=3.8 -y
conda activate ECNet
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0
mim install pytorchvideo
pip install -v -e .