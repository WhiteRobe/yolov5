import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict

# 请自行做好原数据集的备份
# 本质是一个VOC
if __name__ == '__main__':
    np.random.seed(825)
    root_url = Path(__file__).parent
    img_url = root_url / 'images'
    label_url = root_url / 'labels'
    try:  # 转化图像文件
        shutil.move(root_url / 'JPEGImages', img_url)
    except AttributeError as e:
        pass

    if not os.path.exists(label_url):
        os.mkdir(label_url)

    # 划分数据集
    images_list = list(os.listdir(img_url))
    np.random.shuffle(images_list)
    total_num = len(images_list)
    split_rate = [0.6, 0.2, 0.2]
    train, val, test = images_list[:int(split_rate[0] * total_num)], \
                       images_list[int(split_rate[0] * total_num):int(sum(split_rate[0:2]) * total_num)], \
                       images_list[int(sum(split_rate[0:2]) * total_num):]
    for _t_, _d_ in zip(['train', 'val', 'test'], [train, val, test]):
        with open(root_url / f'{_t_}.txt', 'w') as f:
            f.writelines([f'./images/{img}\n' for img in _d_])

    label_map = {
        'person': 0,
        'car': 1,
        'motorbike': 2
    }
    label_count = defaultdict(int)
    for label in os.listdir(root_url / 'Annotations'):
        label_xml = ET.parse(root_url / 'Annotations' / label).getroot()
        # 原图的长和宽
        img_w, img_h = [float(label_xml.find('size').find(_v).text) for _v in ['width', 'height']]
        obj_bboxes = []
        for obj in label_xml.iter('object'):  # 对图中的每个目标
            name = obj.find('name').text.lower()  # 目标的类别
            truncated = int(obj.find('truncated').text)  # 是否被裁剪
            difficult = int(obj.find('difficult').text)  # 难度等级

            if difficult != 0 or truncated != 0:  # Opt. 丢弃困难目标 + 丢弃裁剪目标
                continue
            # elif name in ['motorbike', 'person']:  # Opt. 共三个类，丢弃这两种只留car
            #     continue

            if name not in label_map.keys():  # 添加未被识别的类
                label_map[name] = len(label_map)
            label_count[name] += 1  # 类别计数

            bbox = [float(obj.find('bndbox').find(_k).text) for _k in ['xmin', 'xmax', 'ymin', 'ymax']]

            # 中心归一化
            x, y = (bbox[1] + bbox[0]) / 2. - 1., (bbox[3] + bbox[2]) / 2. - 1.
            w, h = bbox[1] - bbox[0], bbox[3] - bbox[2]
            obj_bboxes.append(f'{label_map[name]} {x / img_w} {y / img_h} {w / img_w} {h / img_h}\n')

        # 写出标签文件
        with open(label_url / f'{Path(label).stem}.txt', 'w') as f:
            f.writelines(obj_bboxes)

    # 写出标签总览文件
    with open(root_url / 'classes.txt', 'w') as f:
        f.writelines([f'{k}\n' for k in label_map.keys()])
    shutil.copy(root_url / 'classes.txt', label_url / 'classes.txt')  # 用于labelImg
    print(list(label_map.keys()))
    print(label_count)
    # shutil.rmtree(root_url / 'Annotations')  # 删除原本的标签集
