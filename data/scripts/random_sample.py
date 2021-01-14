"""
随机采样，构建few-shot数据集
"""

import numpy as np
import os
import shutil
from pathlib import Path
import pandas as pd

pick_img_num = 1  # 每类x个图
class_per_img = None  # 每图每类x个框
train_txt_path = Path('/data2/datasets/fs_obd_ds/cityscapes/yolo_train.txt')
# classes = ['sidewalk', 'sky', 'road', 'building', 'traffic light', 'pole',
#            'static', 'cargroup', 'vegetation', 'traffic sign', 'car', 'bicycle',
#            'train', 'dynamic', 'terrain', 'fence', 'ground', 'motorcycle',
#            'bicyclegroup', 'license plate', 'parking', 'person', 'persongroup',
#            'bridge', 'polegroup', 'tunnel', 'bus', 'rider', 'guard rail', 'wall',
#            'truck', 'trailer', 'caravan', 'rail track', 'rectification border', 'motorcyclegroup',
#            'ridergroup', 'truckgroup']
classes = ['person', 'car', 'truck']  # udacity & cityscapes
# classes = ['person', 'car', 'motorcycle']  # sim10k
# classes = ['person', 'car', 'motorbike']  # mafia


def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)


def read_label(file_path):
    return pd.read_csv(header=None, names=['cls', 'x', 'y', 'w', 'h'], sep=' ',
                       filepath_or_buffer=train_txt_path.parent / file_path.replace('images', 'labels')
                       .replace('\n', '')
                       .replace('png', 'txt')
                       .replace('jpg', 'txt'))


def sample():
    np.random.seed(825)
    with open(train_txt_path, 'r') as txt_f:
        imgs = txt_f.readlines()
    np.random.shuffle(imgs)
    if os.path.exists(train_txt_path.parent / 'fs'):
        shutil.rmtree(train_txt_path.parent / 'fs')

    mkdir(train_txt_path.parent / 'fs' / 'images'), mkdir(train_txt_path.parent / 'fs' / 'labels')
    index = 0
    need_to_match = {label: 0 for label in classes}  # 待匹配的列表
    matched = 0
    fs_train_txt = []
    while index < len(imgs) and matched < len(classes):
        img = imgs[index].replace('\n', '')  # 某个图的地址
        labels = read_label(img)
        for cls_index in labels['cls'].unique():  # 类列表
            label = classes[cls_index]  # 类的名字
            if label in need_to_match and need_to_match[label] < pick_img_num:  # 选中一定数量的图
                # print(label, img, '\n', labels.loc[labels['cls'] == cls_index].iloc[0])
                fs_train_txt.append(img)
                # 复制图像
                shutil.copy(train_txt_path.parent / imgs[index].replace('\n', ''),
                            train_txt_path.parent / 'fs' / 'images' / Path(imgs[index]).name.replace('\n', ''))
                # 约束label
                hit_labels = labels.loc[labels['cls'] == cls_index]
                filter_labels = hit_labels.iloc[0: min(hit_labels.shape[0], class_per_img)]
                filter_labels.to_csv(sep=' ', header=None, index=None,
                                     path_or_buf=train_txt_path.parent / 'fs' / 'labels' /
                                                 Path(imgs[index]).name.replace('\n', '')
                                     .replace('png', 'txt').replace('jpg', 'txt'))
                need_to_match[label] += 1
                matched += int(need_to_match[label] == pick_img_num)
                # print(label)
                break  # 每张图只匹配一个类
        index += 1
    with open(train_txt_path.parent / 'train_yolo_fs.txt', 'w') as f:
        for im in fs_train_txt:
            f.write(f'./fs/images/{Path(im).name}\n')
            if class_per_img == float('inf'):  # 如果不限制每类的数量 则全部复制
                shutil.copy(train_txt_path.parent / im.replace('images', 'labels')
                            .replace('\n', '').replace('png', 'txt').replace('jpg', 'txt'),
                            train_txt_path.parent / 'fs' / 'labels' / Path(im).name
                            .replace('\n', '').replace('png', 'txt').replace('jpg', 'txt'))


if __name__ == '__main__':
    class_per_img = float('inf') if class_per_img is None else class_per_img
    sample()
