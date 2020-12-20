import argparse
from pathlib import Path
import os


def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)


# 转化的标签仅可用于预测
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fr', type=str, default='cityscapes_foggy')
    parser.add_argument('--to', type=str, default='udacity')
    parser.add_argument('--ds_url', type=str, default='/data2/datasets/fs_obd_ds')
    args = parser.parse_args()

    maps = {
        'sim10k': ['person', 'car', 'motorcycle'],
        'cityscapes': ['sidewalk', 'sky', 'road', 'building', 'traffic light', 'pole',
                       'static', 'cargroup', 'vegetation', 'traffic sign', 'car', 'bicycle',
                       'train', 'dynamic', 'terrain', 'fence', 'ground', 'motorcycle',
                       'bicyclegroup', 'license plate', 'parking', 'person', 'persongroup',
                       'bridge', 'polegroup', 'tunnel', 'bus', 'rider', 'guard rail', 'wall',
                       'truck', 'trailer', 'caravan', 'rail track', 'rectification border', 'motorcyclegroup',
                       'ridergroup', 'truckgroup'],
        'udacity': ['person', 'car', 'truck'],
        'mafia': ['person', 'car', 'motorcycle'],
    }
    maps['cityscapes_foggy'] = maps['cityscapes']

    # --- beg --- #
    ds_root = Path(args.ds_url)
    fr_ds_url = ds_root / args.fr
    to_ds_url = ds_root / args.to


    def cg(label_url, fr_dict, to_dict, save_url):
        for file in os.listdir(label_url):  # 对每个标签文件
            if 'classes.txt' == str(file):
                continue
            res = []
            with open(label_url / file, 'r') as f:
                for lines in f.readlines():
                    line = lines.split(' ')
                    label_text = fr_dict[int(line[0])]
                    label_index = None
                    try:
                        label_index = to_dict.index(label_text)
                    except:
                        pass
                    if label_index is not None:
                        res.append(f'{label_index} {line[1]} {line[2]} {line[3]} {line[4]}')
            with open(save_url / file, 'w') as f:
                f.writelines(res)


    if 'cityscapes' in str(fr_ds_url):
        for _t_ in ['train', 'val', 'test']:
            new_label_save_url = fr_ds_url / 'labels_new' / _t_
            mkdir(new_label_save_url)
            cg(fr_ds_url / 'labels' / _t_, maps[args.fr], maps[args.to], new_label_save_url)
    else:
        new_label_save_url = fr_ds_url / 'labels_new'
        mkdir(new_label_save_url)
        cg(fr_ds_url / 'labels', maps[args.fr], maps[args.to], new_label_save_url)

    with open(fr_ds_url / 'labels_new' / 'classes.txt', 'w') as of:
        of.writelines([f'{k}\n' for k in maps[args.to]])
