# coding=utf-8
import argparse

import os
from PIL import Image, ImageSequence

def parse_inference_log():

    log_path = "/root/Projects/CENet/tools/vis_cam_2.log"
    path_prefix = "/root/autodl-tmp/visualizations/cam_outputs/last_conv_svd"

    with open(log_path, "r") as f:
        lines = f.readlines()

    gifs = []
    for i in range(len(lines)):
        # print(lines[i])
        # breakpoint()
        if lines[i] != "inference res: False\n":
            continue

        path_ele = lines[i-2].split(" ")
        gif_path = os.path.join(path_prefix, path_ele[0], path_ele[1], path_ele[2].split("\n")[0])
        # print(gif_path)
        gifs.append(gif_path)
    
    return gifs

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert gif to imgs')
    parser.add_argument('--gif_path', type=str, help='')
    parser.add_argument('--tgt_path', type=str, default='', help='')

    args = parser.parse_args()

    return args


def parseGIF(args):
    gifname = args.gif_path
    # gifname = '/root/Projects/CENet/tools/visualizations/cam_outputs/last_conv_svd/Tester1/reaching_behind/front_IR/res.gif'
    save_dir = os.path.dirname(gifname)
    # tgt_path = args.tgt_path
    # 将gif解析为图片
    # 读取GIF
    im = Image.open(gifname)
    # GIF图片流的迭代器
    iter = ImageSequence.Iterator(im)
    # 获取文件名
    file_name = gifname.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, file_name)
    index = 1
    # 判断目录是否存在
    # save_prefix = gifname.split(".")[-3]
    # pic_dirct = save_prefix
    # pic_dirct = save_prefix + "/{0}".format(file_name)
    mkdirlambda = lambda x: os.makedirs(
        x) if not os.path.exists(x) else True  # 目录是否存在,不存在则创建
    mkdirlambda(save_path)
    # 遍历图片流的每一帧,保存为
    for frame in iter:
        print("image %d: mode %s, size %s" % (index, frame.mode, frame.size))
        frame.save("{}/frame%d.png".format(save_path) % (index))  # 保存图片
        index += 1

    # frame0 = frames[0]
    # frame0.show()

    # 把GIF拆分为图片流
    imgs = [frame.copy() for frame in ImageSequence.Iterator(im)]
    # 把图片流重新成成GIF动图
    imgs[0].save('000out.gif', save_all=True, append_images=imgs[1:])

    # 图片流反序
    imgs.reverse()
    # 将反序后的所有帧图像保存下来
    imgs[0].save('000reverse_out.gif', save_all=True, append_images=imgs[1:])


if __name__ == "__main__":
    args = parse_args()

    # failure_gifs = parse_inference_log()
    # for gif in failure_gifs:
    #     args.gif_path = gif + "/res.gif"
    #     parseGIF(args)

    args.gif_path = "/root/autodl-tmp/visualizations/cam_outputs/last_conv_svd/Tester24/adjusting_radio/front_IR_2/res.gif"
    parseGIF(args)


    # vis_dir = '/root/Projects/CENet/tools/visualizations/cam_outputs/last_conv_svd'
    # tester_dirs = os.listdir(vis_dir)
    # for tester in tester_dirs:
    #     gif_path = os.path.join(vis_dir, tester, 'reaching_behind/front_IR/res.gif')
    #     args.gif_path = gif_path
    #     parseGIF(args)
    # parseGIF(args)
