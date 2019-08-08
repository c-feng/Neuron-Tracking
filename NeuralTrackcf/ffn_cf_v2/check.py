import torch
import torch.utils.data as data

from utils import prepare_data, batch_subvol, mask_subvol, get_data, set_data, get_new_locs
from data import *
from modules import FocalLoss


cfg = neural_train
dataset =  NeuralData(list_file=cfg['list_file'], data_root=cfg['data_root'])

def test1():
    fw, fh, fd, fc = cfg['fov_shape']

    data_loader = data.DataLoader(dataset, batch_size=1,
                                    num_workers=4,
                                    shuffle=True, collate_fn=detection_collate,
                                    pin_memory=True)

    batch_iterator = iter(data_loader)
    images, targets = next(batch_iterator)

    locations = prepare_data(labels=targets, patch_shape=cfg['subvox_shape'])

    print(len(locations), locations.shape)
    print(locations)

crit = FocalLoss()
a = 0.5*torch.ones(1,1,3,3,3)
target = torch.ones(1,1,3,3,3)
print(target,a)
loss = crit(a, target.long())
print(loss)
