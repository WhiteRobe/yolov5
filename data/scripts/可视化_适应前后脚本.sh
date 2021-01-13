# ft_1a
python detect.py --img-size 1024 --conf-thres 0.6 --iou-thres 0.6 --augment --project adpt/detect/C2F --name ft_1a --exist-ok --line_thickness 3 --weights adpt/train/C2F/ft_1a/weights/best.pt --source /data2/datasets/fs_obd_ds/cityscapes_foggy/images/detect --classes 10 21 --fixed-colors
# ft_1a_mixup
python detect.py --img-size 1024 --conf-thres 0.6 --iou-thres 0.6 --augment --project adpt/detect/cityscapes_foggy --name source --exist-ok --line_thickness 3 --weights adpt/train/cityscapes/pre-train/weights/best.pt --source /data2/datasets/fs_obd_ds/cityscapes_foggy/images/detect --classes 10 21 --fixed-colors
# source
python detect.py --img-size 1024 --conf-thres 0.6 --iou-thres 0.6 --augment --project adpt/detect/C2F --name ft_f1a_mixup --exist-ok --line_thickness 3 --weights adpt/train/C2F/ft_1a_mixup/weights/best.pt --source /data2/datasets/fs_obd_ds/cityscapes_foggy/images/detect --classes 10 21 --fixed-colors
# pre-train
python detect.py --img-size 1024 --conf-thres 0.6 --iou-thres 0.6 --augment --project adpt/detect/cityscapes_foggy --name pre-train --exist-ok --line_thickness 3 --weights adpt/train/cityscapes_foggy/pre-train/weights/best.pt --source /data2/datasets/fs_obd_ds/cityscapes_foggy/images/detect --classes 10 21 --fixed-colors
# cityscapes
python detect.py --img-size 1024 --conf-thres 0.6 --iou-thres 0.6 --augment --project adpt/detect/cityscapes --name pre-train --exist-ok --line_thickness 3 --weights adpt/train/cityscapes/pre-train/weights/best.pt --source /data2/datasets/fs_obd_ds/cityscapes/images/detect --classes 10 21 --fixed-colors
