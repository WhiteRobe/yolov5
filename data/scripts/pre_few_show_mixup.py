"""
预叠底小样本回归框标注
"""

import numpy as np
import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

np.random.seed(825)


def read_img_label(img_url, label_url, pre_label_url):
    _img = Image.open(img_url)
    _label = pd.read_csv(header=None, names=['cls', 'x', 'y', 'w', 'h'], sep=' ', filepath_or_buffer=label_url)
    _pre_label = pd.read_csv(header=None, names=['cls', 'x', 'y', 'w', 'h'], sep=' ', filepath_or_buffer=pre_label_url)
    return _img, _label, _pre_label


def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)


def give_box(_x, _y, _w, _h, _img_size):
    _weight, _height = _img_size
    _cx, _cy = _x * _weight, _y * _height
    return _cx - _w * _weight // 2, _cx + _w * _weight // 2, _cy - _h * _height // 2, _cy + _h * _height // 2  # xl, xr, yt, yb


colors = ['red', 'blue', 'green', 'pink', 'yellow']

if __name__ == '__main__':
    # --变量修改区-- #
    root = Path('/data2/datasets/fs_obd_ds/mafia')  # 目标域数据集存放地址
    fs_root = root / 'fs'
    img_root = fs_root / 'images'
    lb_root = fs_root / 'labels'  # fs真实框的地址
    pre_box_root = Path(fs_root / 'labels')  # 预测框的label地址
    output_img_root = fs_root / 'pre_images'

    # --- #
    if os.path.exists(output_img_root):
        shutil.rmtree(output_img_root)
    mkdir(output_img_root)

    for img_name in tqdm(os.listdir(img_root)):
        name_stem = Path(img_name).stem
        image, label, pre_label = read_img_label(img_root / img_name, lb_root / f"{name_stem}.txt",
                                                 pre_box_root / f"{name_stem}.txt")
        image_out = image.copy()
        # draw = ImageDraw.Draw(image)
        cls_list = list(label['cls'].unique())  # 类别总列表

        for _, row in pre_label.iterrows():
            xl, xr, yt, yb = give_box(row['x'], row['y'], row['w'], row['h'], image.size)
            # draw.line([(xl, yt), (xr, yt), (xr, yb), (xl, yb), (xl, yt)], width=3, fill=colors[int(row['cls'])])
            patch = image.crop((xl, yt, xr, yb))  # 预测框的图

            # 选中某类中的一个真实对象的框
            try:
                blend_item = label.loc[label['cls'] == int(row['cls'])].sample(1)  # cls
                txl, txr, tyt, tyb = give_box(blend_item['x'], blend_item['y'], blend_item['w'], blend_item['h'],
                                              image.size)
                target_patch = image.crop((txl, tyt, txr, tyb))  # 剪出来真实标注对象
                target_patch = target_patch.resize(patch.size, Image.ANTIALIAS)  # 对齐尺寸

                patch = Image.blend(patch, target_patch, np.random.random() * 0.5)  # 预测对象叠加真实对象
                image_out.paste(patch, (int(xl), int(yt)))
            except:
                print(f'损失一副图像拼接:[{img_name}]')  # 本图中不包含所需的cls

        # image_out.show()
        image_out.save(output_img_root / img_name)
