import torch
from utils.torch_utils import select_device, intersect_dicts
from utils.general import make_divisible, check_file, set_logging
from models.yolo import Model

model_path = 'adpt/train/mafia/pre-train-1024/weights/best.pt'
bs = 1
device = select_device('', batch_size=bs)
ckpt = torch.load(model_path, map_location=device)

print(ckpt.keys())
print(ckpt['model'].yaml)

model = Model(ckpt['model'].yaml, ch=3, nc=3).to(device)  # create
state_dict = ckpt['model'].float().state_dict()  # to FP32
state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=['anchor'])  # intersect
model.load_state_dict(state_dict, strict=False)  # load
