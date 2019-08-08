import os
import argparse
import numpy as np
from queue import Queue
from skimage.external import tifffile
from scipy.special import logit, expit

import torch
import torch.nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from data import *
# from model import FloodFillingNetwork as ffn
from models.voxResnet import VoxResnet as ffn
from inference_utils import prepare_data, is_valid_seed, in_min_boundary_dist, mask_subvol, get_data, set_data, get_new_locs

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='./weights/lr1e-3/FFN_Neural_180.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def inference_v1(model, images_, targets, gts, cfg):
    fw, fh, fd, fc = cfg['fov_shape']
    images = images_

    locations = prepare_data(labels=targets, patch_shape=cfg['sample_shape'], nums=1)

    segmentation = np.zeros([1, 1, 300, 300, 300])
    label = 1
    for _, location in enumerate(locations, start=1):
        # whether the location is valid
        if not (is_valid_seed(location, segmentation)):
            continue
        if in_min_boundary_dist(location, segmentation):
            continue
        
        # 抽取patch, 同时生成与中心点相对应的监督标签
        w, h, d, c = cfg['subvol_shape']
        subvol_data = np.zeros(shape=[1, c, w, h, d], dtype=np.float32)
        subvol_data = images.cpu()
        # 与patch相对应的soft二值mask
        subvol_mask = mask_subvol(cfg['subvol_shape'])  # [300,300,300]

        # Create FOV dicts, and center locations
        cx, cy, cz = location + np.array([fw, fh, fd], np.int32) // 2
        start_pos = (cx, cy, cz)
        done = {(cx, cy, cz)}  # set()
        queue = Queue()
        queue.put([cx, cy, cz])
        
        # initilize the mask center with value logit(0.95)
        subvol_mask[0, 0, cx, cy, cz] = logit(0.95)
        min_pos = np.array(start_pos)
        max_pos = np.array(start_pos)
        
        # Compute upper and lower bounds
        upper = [w - fw // 2, h - fh // 2, d - fd // 2]
        lower = [fw // 2, fh // 2, fd // 2]
        
        while not queue.empty():
            # Get new list of FOV locations
            current_loc = np.array(queue.get(), np.int32)
            # Center around FOV
            fov_data = get_data(subvol_data, current_loc, cfg['fov_shape'])
            fov_mask = get_data(subvol_mask, current_loc, cfg['fov_shape'])
            # Add merging of old and new mask
            d_m = np.concatenate([fov_data, fov_mask], axis=1)
            if args.cuda:
                d_m = torch.Tensor(d_m).to('cuda')
            else:
                d_m = torch.Tensor(d_m)
            
            pred = model(d_m)
            logit_seed = torch.add(torch.from_numpy(fov_mask).to('cuda'), other=pred)
            prob_seed = expit(logit_seed.detach().cpu().numpy())
            print(np.max(prob_seed), np.min(prob_seed), np.sum(prob_seed>0.9)/(33*33*33))

            # 更新patch对应的soft二值mask
            set_data(subvol_mask, current_loc, logit_seed.detach().cpu().numpy())

            # 记录fov遍历的范围
            min_pos = np.minimum(min_pos, current_loc)
            max_pos = np.maximum(max_pos, current_loc)

            # Compute new locations
            new_locations = get_new_locs(logit_seed.detach().cpu().numpy(), cfg['delta'], cfg['tmove'])
            for new in new_locations:
                new = np.array(new, np.int32) + current_loc
                bounds = [lower[j] <= new[j] < upper[j] for j in range(3)]
                stored_loc = tuple([new[i] for i in range(3)])
                if all(bounds) and stored_loc not in done:
                    done.add(stored_loc)
                    queue.put(new)
            print("One FOV ends")
        print("One instance ends. Label: {}".format(label))
        print("-"*10, '\n', "-"*10)

        # segmentation update
        if subvol_mask[0, 0, cx, cy, cz] < logit(0.9):
            if segmentation[0, 0, cx, cy, cz] == 0:
                segmentation[0, 0, cx, cy, cz] = -1
            print("Failed: weak seed")
            continue
        
        sel = [slice(max(s, 0), e + 1) for s, e in zip(
                    min_pos - np.array(cfg['fov_shape'])[:-1] // 2,
                    max_pos + np.array(cfg['fov_shape'])[:-1] // 2)]
        mask = subvol_mask[..., sel[0], sel[1], sel[2]] >= logit(0.6)

        mask &= segmentation[..., sel[0], sel[1], sel[2]] <= 0
        actual_segmented_voxels = np.sum(mask)
        
        # Segment too small?
        if actual_segmented_voxels < 20:
            if segmentation[0, 0, cx, cy, cz] == 0:
                segmentation[0, 0, cx, cy, cz] = -1
            print('Failed: too small: %d'%(actual_segmented_voxels))
            continue
        
        segmentation[..., sel[0], sel[1], sel[2]][mask] = label
        label += 1

    return segmentation

def build_model(args):

    # model = ffn(in_planes=2, module_nums=8)
    model = ffn()

    if args.cuda:
        model = torch.nn.DataParallel(model)  # 多卡存在问题, priors加倍了
        cudnn.benchmark = True

    if args.cuda:
        model = model.cuda()
    
    print("Initializing weights...")
    model.load_state_dict(torch.load(args.trained_model))
    model.eval()
    return model

def test_net(args):
    cfg = neural_test
    dataset = NeuralData(list_file=cfg['list_file'],
                        data_root=cfg['data_root'])
    model = build_model(args)
    for i in range(len(dataset)):
        img_gt, target, img_ = dataset[i]
        img = img_
        img_name = dataset.get_img_name(i)
        print("Processing {}....".format(img_name))
        imgs = torch.unsqueeze(img.cuda(), dim=0)
        targets = torch.unsqueeze(target.cuda(), dim=0)
        gts = torch.unsqueeze(img_gt.cuda(), dim=0)

        ins_mask = inference_v1(model=model, images_=imgs, targets=targets, gts=gts, cfg=cfg)
        tifffile.imsave(file=os.path.join(args.save_folder, img_name+'_pred.tif'), data=ins_mask.astype(np.int))
        print(np.unique(ins_mask))
        print("Finishing predict {}".format(img_name))


if __name__ == "__main__":
    # cfg = neural_test
    # dataset = NeuralData(list_file=cfg['list_file'],
    #                     data_root=cfg['data_root'])

    # for i in range(1):
    #     img, gt = dataset[i]
    #     img_name = dataset.get_img_name(i)
    #     print("Processing {}....".format(img_name))
    #     ins_mask = inference(image=img, target=gt, cfg=cfg)
    #     tifffile.imsave(file=os.path.join(args.save_folder, img_name+'_pred.tif'), data=ins_mask.astype(np.int))
    #     print(np.unique(ins_mask))
    #     print("Finishing predict {}".format(img_name))



    test_net(args)