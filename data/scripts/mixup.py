"""
本文件用于做图像混合
"""

import numpy as np
import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

np.random.seed(825)


def read_img_label(base, url):
    img = base / url
    label = base / url.replace('images', 'labels').replace('png', 'txt').replace('jpg', 'txt')
    # print(img, label)
    img = Image.open(img)
    label = pd.read_csv(header=None, names=['cls', 'x', 'y', 'w', 'h'], sep=' ', filepath_or_buffer=label)
    return img, label


def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)


if __name__ == '__main__':
    # --变量修改区-- #
    ori_root = Path('/data2/datasets/fs_obd_ds/sim10k')  # 源域数据集存放地址
    root = Path('/data2/datasets/fs_obd_ds/mafia')  # 目标域数据集存放地址

    train_txt = 'train.txt'
    fs_train_txt = 'train_yolo_fs.txt'  # 小样本数据集标签 由random_sample采样得到

    fs_pair_num, fs_label_num = 2, 10  # 每张源图配对x个小样本图，小样本图中的框至多选择x个
    # --变量修改区-- #

    # pre-paired
    if os.path.exists(root / 'mixup'):
        shutil.rmtree(root / 'mixup')

    mkdir(root / 'mixup' / 'labels'), mkdir(root / 'mixup' / 'images')

    mixup_dict = []
    miss = [0, 0]
    with open(ori_root / train_txt, 'r') as f:
        for img_url in tqdm(f.readlines()):
            try:
                oimg, olabel = read_img_label(ori_root, img_url.replace('\n', ''))
            except:
                miss[0] += 1
                continue

            with open(root / fs_train_txt, 'r') as fs:
                _fs = list(fs.readlines())
                np.random.shuffle(_fs)
                for img_url2 in _fs[:min(fs_pair_num, len(_fs))]:
                    try:
                        fimg, flabel = read_img_label(root, img_url2.replace('\n', ''))
                    except:
                        miss[1] += 1
                        continue
                    fimg = fimg.resize(oimg.size, Image.ANTIALIAS)

                    blend_rate = round(np.random.random() * 0.5, 2)  # <0.5
                    mimg = Image.blend(oimg, fimg, blend_rate)  # 混合好的图像

                    mimg_url = Path(img_url2.replace('fs', 'mixup').replace('\n', ''))
                    file_name = f'{Path(img_url).stem}__{Path(img_url2).stem}__{str(blend_rate)}{mimg_url.suffix}'
                    print(file_name)
                    mimg.save(root / 'mixup' / 'images' / file_name)
                    mixup_dict.append(f"./{Path('mixup/images') / file_name}\n")

                    # 处理标签
                    label_file_name = file_name.replace(mimg_url.suffix, '.txt')
                    mlabel = pd.concat([olabel, flabel.sample(min(fs_label_num, flabel.shape[0]))])
                    # print(olabel.shape, flabel.shape)
                    # print(mlabel.shape)
                    mlabel.to_csv(sep=' ', header=False, index=False,
                                  path_or_buf=root / 'mixup' / 'labels' / label_file_name)

    # 写出配置文件
    with open(root / 'mixup_train_yolo.txt', 'w') as fo:
        fo.writelines(mixup_dict)

    print(f'miss = {miss}')
