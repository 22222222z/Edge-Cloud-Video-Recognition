from mmaction.apis import inference_recognizer, init_recognizer

config_path = '/root/Projects/CENet/configs/recognition/tsm/tsm_attnlast_DAD.py'
checkpoint_path = '/root/Projects/CENet/work_dirs/tsm_attnlast_DAD/epoch_50.pth' # 可以是本地路径
img_path = 'demo/demo.mp4'   # 您可以指定自己的图片路径
img_path = '/root/autodl-tmp/dataset/DAD/DAD/Tester10/normal_driving_1/front_IR'

# 从配置文件和权重文件中构建模型
# model = init_recognizer(config_path, checkpoint_path, device="cpu")  # device 可以是 'cuda:0'
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device 可以是 'cuda:0'
# 对单个视频进行测试
result = inference_recognizer(model, img_path)