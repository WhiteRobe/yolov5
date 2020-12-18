import pandas as pd
import os
from pathlib import Path
import shutil
import numpy as np
from collections import defaultdict


def mkdir(url):
    if not os.path.exists(url):
        os.mkdir(url)


if __name__ == '__main__':
    # datasets download : https://github.com/udacity/self-driving-car/tree/master/annotations
    np.random.seed(825)
    root_url = Path(__file__).parent / 'udacity-annoations-crowdai'
    img_url = root_url / 'images'
    labels_url = root_url / 'labels'

    img_w, img_h = 1920., 1200.  # 图像尺寸

    # change dir name
    shutil.move(root_url / 'object-detection-crowdai', img_url)
    mkdir(labels_url)

    # 读入标记
    annotations = pd.read_csv('labels_crowdai.csv')  # xmin,ymin,xmax,ymax,Frame,Label,Preview URL

    label_map = {
        'Pedestrian': 0,
        'Car': 1,
        'Truck': 2
    }
    label_count = defaultdict(int)
    for _, row in annotations.iterrows():
        # if row.Label != 'Car':  # 这里只准备提取car这一种类
        #     continue

        if row.Label not in label_map.keys():
            label_map[row.Label] = len(label_map)
        label_count[row.Label] += 1

        # 归一化
        w, h = float(row.xmax - row.xmin), float(row.ymax - row.ymin)
        x, y = (row.xmax + row.xmin) / 2.0 - 1, (row.ymax + row.ymin) / 2.0 - 1

        with open(labels_url / f'{Path(row.Frame).stem}.txt', 'a') as f:  # 改为yolo格式
            f.write(f'{label_map[row.Label]} {x / img_w} {y / img_h} {w / img_w} {h / img_h}\n')

    with open(root_url / 'classes.txt', 'w') as f:  # 写出类标签
        f.writelines([f'{k.lower()}\n' for k in label_map.keys()])
    shutil.copy(root_url / 'classes.txt', labels_url / 'classes.txt')  # 用于labelImg

    # 划分数据集(可选)
    split_rate = [0.6, 0.2, 0.2]  # 训练集　验证集　测试集
    image_list = list(os.listdir(img_url))
    np.random.shuffle(image_list)  # 原地洗乱
    total_num = len(image_list)
    train, val, test = image_list[:int(split_rate[0] * total_num)], \
                       image_list[int(split_rate[0] * total_num): int(sum(split_rate[0:2]) * total_num)], \
                       image_list[int(sum(split_rate[0:2]) * total_num):]
    for _t_, _d_ in zip(['train', 'val', 'test'], [train, val, test]):
        with open(root_url / f'{_t_}.txt', 'w') as f:
            f.writelines([f'./images/{img}\n' for img in _d_])
    print(list(label_map.keys()))
    print(label_count)
