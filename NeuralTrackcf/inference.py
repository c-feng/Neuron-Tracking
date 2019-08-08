import os
import numpy as np
import time
from itertools import product
from queue import Queue
from skimage.external import tifffile
from skimage.morphology import dilation, cube, ball
import pdb

import torch
import torch.nn as nn

from neuralTrack.utils.serialization import load_checkpoint, save_checkpoint, read_json
from utils.utils import fromSeed2Data, sample_point, findCenterCoord
from utils.inference_utils import set_data, get_new_locations, getfovLabel
from utils.transcheckpoint import trans_Distributed2Parallel
from utils.direct_field import batch_direct_field3D

class Inference():
    def __init__(self, model, model_path, patch_shape, fov_shape, logs_path, metric_func=None, overlap=10, device="cuda"):
        self.model = model
        self.model_path = model_path
        self.metric_func = metric_func
        self.patch_shape = np.array(patch_shape)
        self.fov_shape = np.array(fov_shape)
        self.logs_path = logs_path
        self.overlap = overlap
        self.device = device

        self.init_model(model_path)

    def init_model(self, path):
        check_point = load_checkpoint(path)
        # check_point = trans_Distributed2Parallel(path)
        self.model.load_state_dict(check_point['state_dict'])
        
        self.featureHead = nn.DataParallel(self.model.featureHead)
        self.segHead = nn.DataParallel(self.model.segHead)
        self.fusenet = nn.DataParallel(self.model.fuseNet)
        self.ffn = nn.DataParallel(self.model.ffn)

    def split_data(self, shape, patch_shape, overlap):
        s = np.array(shape)
        ps = np.array(patch_shape)
        ol = np.array(overlap)
        split_num = np.ceil((s - ps ) / (ps - ol)) + 1
        index = [[i*po for i in range(int(sn))] for sn, po in zip(split_num, (ps-ol))]
        
        # 左上角顶点
        if len(shape) == 2:
            coords = [[i,j] for i,j in product(*index)]
        else:
            coords = [[i,j,k] for i,j,k in product(*index)]
        return np.array(coords)

    def pad_data(self, data, patch_shape, overlap):
        s = data.shape[-3:]
        ps = np.array(patch_shape)
        ol = np.array(overlap)
        shape = np.ceil((s - ps) / (ps - ol)) * (ps - ol) + ps
        pad = np.zeros(shape.astype(int))
        pad[:s[0], :s[1], :s[2]] = data[..., :, :, :]
        return pad

    def _parse_data(self, seg):
        pass

    def forward(self, img):
        patch_seeds = self.split_data(img.shape, self.patch_shape, self.overlap) # 左上
        patch_seeds = patch_seeds + self.patch_shape // 2 - 1 + self.patch_shape % 2 # 中心点
        img_pad = self.pad_data(img, self.patch_shape, self.overlap)

        ins_mask = np.zeros(img_pad.shape)
        max_id = 0
        for i, seed in enumerate(patch_seeds):
            patch = fromSeed2Data(seed, img_pad, self.patch_shape)
            patch = patch[None, None, ...]

            imgs = torch.from_numpy(patch).float()

            rrb = torch.cat(self.featureHead(imgs), dim=1)

            full_seg_output = self.segHead(rrb)
            feature = self.fusenet(rrb)

            _, full_seg_pred = torch.max(full_seg_output, dim = 1)
            full_seg_pred_array = full_seg_pred.detach().cpu().numpy()

            overlap = ((ins_mask>0) & (fromSeed2fullData(seed, img_pad.shape, self.patch_shape)>0)).astype(int)
            overlap_coords = np.stack(np.where(overlap>0), axis=0).T
            overlap[overlap_coords[:, 0], overlap_coords[:, 1], overlap_coords[:, 2]] = ins_mask[overlap_coords[:, 0],
                                                overlap_coords[:, 1], overlap_coords[:, 2]]
            overlap_patch = fromSeed2Data(seed, overlap, self.patch_shape)
            
            # if np.sum(overlap) == 0:
            #     seed_points, labels = sample_point(full_seg_pred_array, num=1)
            #     seed_points = findCenterCoord(seed_points[None], None, self.patch_shape, self.fov_shape)
            # else:
            #     seed_points, labels = sample_point(overlap_patch, num=1)
            # patch_mask, max_id = self.ffn_inference(self.ffn, full_seg_pred_array, feature, overlap_patch, max_id, i)
            patch_mask, max_id = self.ffn_inference_fixedsplit(self.ffn, full_seg_pred_array,
                                                               feature.detach().cpu().numpy(),
                                                               overlap_patch, max_id, i, self.logs_path, False)

            # 更新ins_mask
            set_data(ins_mask, seed, patch_mask, False)

            # # vis
            # full_seg_path = os.path.join("/media/fcheng/NeuralTrackcf/eval", "patch_"+str(i)+'.tif')
            path = self.logs_path
            # tifffile.imsave(full_seg_path, full_seg_pred_array.astype(np.float32))
            # tifffile.imsave(os.path.join(path, "patch_tif_"+str(i)+'.tif'), patch.astype(np.float32)[0,0,...])
            # tifffile.imsave(os.path.join(path, "patchmask_tif_"+str(i)+'.tif'), patch_mask.astype(np.float32))
            # print(f"'patch_{str(i)}.tif have been saved.")

            tifffile.imsave(os.path.join(path, "ins_mask_"+str(i)+'.tif'), ins_mask.astype(np.float16))
        print("max_id:", max_id)
        return ins_mask

    def forward_gt(self, img, gt):
        patch_seeds = self.split_data(img.shape, self.patch_shape, self.overlap) # 左上
        patch_seeds = patch_seeds + self.patch_shape // 2 - 1 + self.patch_shape % 2 # 中心点
        img_pad = self.pad_data(img, self.patch_shape, self.overlap)
        full_gt_pad = self.pad_data((gt>0).astype(int), self.patch_shape, self.overlap)

        ins_mask = np.zeros(img_pad.shape)
        max_id = 0
        for i, seed in enumerate(patch_seeds):
            patch = fromSeed2Data(seed, img_pad, self.patch_shape)
            patch = patch[None, None, ...]

            patch_gt = fromSeed2Data(seed, full_gt_pad, self.patch_shape)
            patch_gt = patch_gt[None, ...]

            imgs = torch.from_numpy(patch).float()

            rrb = torch.cat(self.featureHead(imgs), dim=1)

            full_seg_output = self.segHead(rrb)
            feature = self.fusenet(rrb)

            _, full_seg_pred = torch.max(full_seg_output, dim = 1)
            full_seg_pred_array = full_seg_pred.detach().cpu().numpy()

            overlap = ((ins_mask>0) & (fromSeed2fullData(seed, img_pad.shape, self.patch_shape)>0)).astype(int)
            overlap_coords = np.stack(np.where(overlap>0), axis=0).T
            overlap[overlap_coords[:, 0], overlap_coords[:, 1], overlap_coords[:, 2]] = ins_mask[overlap_coords[:, 0],
                                                overlap_coords[:, 1], overlap_coords[:, 2]]
            overlap_patch = fromSeed2Data(seed, overlap, self.patch_shape)
            
            # patch_mask, max_id = self.ffn_inference(self.ffn, full_seg_pred_array, feature, overlap_patch, max_id, i)
            patch_mask, max_id = self.ffn_inference_fixedsplit(self.ffn, patch_gt,
                                                               feature.detach().cpu().numpy(),
                                                               overlap_patch, max_id, i, self.logs_path, False)

            # 更新ins_mask
            set_data(ins_mask, seed, patch_mask, True)

            # # vis
            # full_seg_path = os.path.join("/media/fcheng/NeuralTrackcf/eval", "patch_"+str(i)+'.tif')
            path = self.logs_path
            # tifffile.imsave(full_seg_path, full_seg_pred_array.astype(np.float32))
            # tifffile.imsave(os.path.join(path, "patch_tif_"+str(i)+'.tif'), patch.astype(np.float32)[0,0,...])
            # tifffile.imsave(os.path.join(path, "patchmask_tif_"+str(i)+'.tif'), patch_mask.astype(np.float32))
            # print(f"'patch_{str(i)}.tif have been saved.")

            tifffile.imsave(os.path.join(path, "ins_mask_"+str(i)+'.tif'), ins_mask.astype(np.float16))
        print("max_id:", max_id)
        return ins_mask

    def ffn_inference(self, ffn_model, full_seg, feature, overlap_p, max_id, idx, logs_path, vis=False):
        """ffn inference过程
        """
        # 初始种子点的产生
        if np.sum(overlap_p) == 0:
            label_coords, labels = sample_point(full_seg[0,...], num=1)
            labels += max_id
            seed_points = findCenterCoord(label_coords, None, self.patch_shape, self.fov_shape)
            label_coords = label_coords.tolist()
        else:
            seed_points = []
            labels = []
            label_coords = []
            for i in np.unique(overlap_p)[1:]:
                op = (overlap_p == i).astype(int)
                op[op>0] = i
                label_coord, label = sample_point(op, num=1)
                seed_point = findCenterCoord(label_coord, None, self.patch_shape, self.fov_shape)
                
                label_coords += label_coord.tolist()
                seed_points += seed_point
                labels += label

        # patch分割的结果
        patch_mask = np.zeros(overlap_p.shape)
        if np.sum(overlap_p) != 0:
            patch_mask[overlap_p>0] = overlap_p[overlap_p>0]

        # 种子队列
        queue = Queue()
        for sp, lc, l in zip(seed_points, label_coords, labels):
            queue.put([sp, lc, l])  # [coords, label]
        
        done = set()
        curr_id = max_id
        count = 0
        while not queue.empty():
            coord, l_c, label = queue.get()
            # print("label ", label)
            done.add(tuple(l_c))
            curr_id = max(curr_id, label)

            fov_feature = fromSeed2Data(coord, feature, self.fov_shape)
            seg_map = fromSeed2Data(coord, full_seg[None], self.fov_shape)
            # seed_map = get_fovSeedData()
            related_coord = self.fov_shape // 2 - (np.array(coord) - np.array(l_c)) \
                                 - 1 + self.fov_shape%2
            seed_map = CenterGaussianHeatMap(self.fov_shape, related_coord)
            seed_map = seed_map[None, None]

            # ffn网络forward
            seg_map = torch.from_numpy(seg_map).to(self.device).float()
            seed_map = torch.from_numpy(seed_map).to(self.device).float()

            try:
                fov_data = torch.cat([fov_feature, seg_map, seed_map], dim=1)
            except RuntimeError:
                pdb.set_trace()
            fov_output = self.ffn(fov_data)

            fov_prob, fov_pred = torch.max(fov_output, dim=1)
            fov_pred_array = fov_pred.detach().cpu().numpy()
            fov_pred_array[fov_pred_array>0] = label
            # find new locations
            new_locations = get_new_locations(fov_prob.detach().cpu().numpy(), patch_mask, coord, fov_pred_array)

            # 更新patch_mask
            set_data(patch_mask, coord, np.squeeze(fov_pred_array))

            if new_locations != []:
                for new in new_locations:
                    label_coord = (np.array(new) + coord).astype(int)
                    # label = patch_mask[label_coord[0], label_coord[1], label_coord[2]]
                    # seed_point = findCenterCoord(label_coord, None, self.patch_shape, self.fov_shape)\
                    seed_point = findNextcoord(label_coord, self.patch_shape, self.fov_shape)
                    stored_loc = tuple([label_coord[i] for i in range(3)])
                    stored_locs = []
                    for i,j,k in product(range(-2, 3), range(-2, 3), range(-2, 3)):
                        stored_locs.append(tuple(label_coord + [i,j,k]))
                    if np.all([i not in done for i in stored_locs]):
                        queue.put([seed_point, stored_loc, label])
    
            elif queue.empty():
                remain = ((full_seg>0) & (patch_mask==0)).astype(int)
                if np.sum(remain) == 0: continue
                label_coord, _ = sample_point(remain[0], num=1)
                label = curr_id + 1
                seed_point = findCenterCoord(label_coord, None, self.patch_shape, self.fov_shape)
                queue.put([seed_point[0], label_coord[0].tolist(), label])
            
            if count % 10 == 0 and vis:
                # vis
                fov_path = os.path.join(logs_path, "fov_"+str(idx)+"_"+str(count)+'.tif')
                path = logs_path
                tifffile.imsave(fov_path, fov_pred_array.astype(np.float32)[0,...])
                tifffile.imsave(os.path.join(path, "fov_tif_"+str(idx)+'_'+str(count)+'.tif'), seg_map.detach().cpu().numpy().astype(np.float32)[0,0,...])
                print(f"'fov_{str(idx)}.tif have been saved.")
            count += 1
        return patch_mask, curr_id
    
    def ffn_inference_fixedsplit(self, ffn_model, full_seg, feature, overlap_p, max_id, idx, logs_path, vis=False):
        """ffn inference过程
        """
        fov_ol_delta = 5
        fov_coords = self.split_data(self.patch_shape, self.fov_shape, overlap=fov_ol_delta)
        fov_coords = fov_coords + self.fov_shape//2-1 + self.fov_shape%2
        
        seg_pad = self.pad_data(full_seg[0], self.fov_shape, fov_ol_delta)
        feature_pad = self.pad_data(feature, self.fov_shape, fov_ol_delta)
        overlap_pad = self.pad_data(overlap_p, self.fov_shape, fov_ol_delta)

        # 分割结果
        patch_mask = np.zeros(overlap_pad.shape)
        seed_mask = np.zeros(overlap_pad.shape)
        if np.sum(overlap_pad) != 0:
            # patch_mask[overlap_pad>0] = overlap_pad[overlap_pad>0]
            seed_mask[overlap_pad>0] = overlap_pad[overlap_pad>0]

        curr_id = max_id
        count = 0
        for coord in fov_coords:

            # fov_mask = np.zeros(self.fov_shape)
            fov_feature = fromSeed2Data(coord, feature_pad, self.fov_shape)
            seg_map = fromSeed2Data(coord, seg_pad, self.fov_shape)
            
            fov_feature = torch.from_numpy(fov_feature).to(self.device).float()[None, None]
            seg_map = torch.from_numpy(seg_map).to(self.device).float()[None, None]
            while condition(patch_mask, seg_pad, coord, self.fov_shape):
                label, l_c = getfovLabel(coord, patch_mask, seg_pad, seed_mask, self.fov_shape)
                label = curr_id + 1 if label == 0 else label
                print(label)
                curr_id = max(curr_id, label)

                seed_map = CenterGaussianHeatMap(self.fov_shape, l_c)
                seed_map = seed_map[None, None]

                # ffn网络forward
                seed_map = torch.from_numpy(seed_map).to(self.device).float()

                fov_data = torch.cat([fov_feature, seg_map, seed_map], dim=1)
                fov_output = self.ffn(fov_data)

                fov_prob, fov_pred = torch.max(fov_output, dim=1)
                fov_pred_array = fov_pred.detach().cpu().numpy()
                fov_pred_array[fov_pred_array>0] = label

                set_data(patch_mask, coord, np.squeeze(fov_pred_array), covered=False)
                set_data(seed_mask, coord, np.squeeze(fov_pred_array), covered=False)
                
                if count % 1 == 0 and vis:
                    # vis
                    # path = "/media/fcheng/NeuralTrackcf/eval"
                    path = logs_path
                    fov_path = os.path.join(path, "fov_"+str(idx)+"_"+str(count)+'.tif')
                    tifffile.imsave(fov_path, fov_pred_array.astype(np.float32)[0,...])
                    tifffile.imsave(os.path.join(path, "fov_seg_"+str(idx)+'_'+str(count)+'.tif'), seg_map.detach().cpu().numpy().astype(np.float32)[0,0,...])
                    print(f"'fov_{str(idx)}.tif have been saved.")
                count += 1
        return patch_mask[..., :self.patch_shape[0], :self.patch_shape[1], :self.patch_shape[2]], curr_id

class DF_Inference():
    def __init__(self, model, model_path, patch_shape, fov_shape, logs_path, metric_func=None, overlap=10, device="cuda"):
        self.model = model
        self.model_path = model_path
        self.metric_func = metric_func
        self.patch_shape = np.array(patch_shape)
        self.fov_shape = np.array(fov_shape)
        self.logs_path = logs_path
        self.overlap = overlap
        self.device = device

        self.init_model(model_path)

    def init_model(self, path):
        check_point = load_checkpoint(path)
        # check_point = trans_Distributed2Parallel(path)
        self.model.load_state_dict(check_point['state_dict'])
        
        self.featureHead = nn.DataParallel(self.model.featureHead)
        self.segHead = nn.DataParallel(self.model.segHead)
        # self.fusenet = nn.DataParallel(self.model.fuseNet)
        self.ffn = nn.DataParallel(self.model.ffn)

    def split_data(self, shape, patch_shape, overlap):
        s = np.array(shape)
        ps = np.array(patch_shape)
        ol = np.array(overlap)
        split_num = np.ceil((s - ps ) / (ps - ol)) + 1
        index = [[i*po for i in range(int(sn))] for sn, po in zip(split_num, (ps-ol))]
        
        # 左上角顶点
        if len(shape) == 2:
            coords = [[i,j] for i,j in product(*index)]
        else:
            coords = [[i,j,k] for i,j,k in product(*index)]
        return np.array(coords)

    def pad_data(self, data, patch_shape, overlap):
        s0 = data.shape[:-3]
        s = data.shape[-3:]
        ps = np.array(patch_shape)
        ol = np.array(overlap)
        shape = np.ceil((s - ps) / (ps - ol)) * (ps - ol) + ps
        shape = shape.astype(int)
        if len(s0) == 0:
            pad = np.zeros(shape)
        else:
            pad = np.zeros([*s0, *shape])
        pad[..., :s[0], :s[1], :s[2]] = data[..., :, :, :]
        return pad

    def _parse_data(self, seg):
        pass

    def forward(self, img, gt=None):
        patch_seeds = self.split_data(img.shape, self.patch_shape, self.overlap) # 左上
        patch_seeds = patch_seeds + self.patch_shape // 2 - 1 + self.patch_shape % 2 # 中心点
        img_pad = self.pad_data(img, self.patch_shape, self.overlap)
        if gt is not None:
            full_gt_pad = self.pad_data((gt>0).astype(int), self.patch_shape, self.overlap)

        ins_mask = np.zeros(img_pad.shape)
        max_id = 0
        for i, seed in enumerate(patch_seeds):
            patch = fromSeed2Data(seed, img_pad, self.patch_shape)
            patch = patch[None, None, ...]

            if gt is not None:
                patch_gt = fromSeed2Data(seed, full_gt_pad, self.patch_shape)
                patch_gt = patch_gt[None]

            imgs = torch.from_numpy(patch).float()

            rrb = torch.cat(self.featureHead(imgs), dim=1)

            full_seg_output = self.segHead(rrb)
            # feature = self.fusenet(rrb)

            _, full_seg_pred = torch.max(full_seg_output, dim = 1)
            full_seg_pred_array = full_seg_pred.detach().cpu().numpy()

            overlap = ((ins_mask>0) & (fromSeed2fullData(seed, img_pad.shape, self.patch_shape)>0)).astype(int)
            overlap_coords = np.stack(np.where(overlap>0), axis=0).T
            overlap[overlap_coords[:, 0], overlap_coords[:, 1], overlap_coords[:, 2]] = ins_mask[overlap_coords[:, 0],
                                                overlap_coords[:, 1], overlap_coords[:, 2]]
            overlap_patch = fromSeed2Data(seed, overlap, self.patch_shape)
            
            # if np.sum(overlap) == 0:
            #     seed_points, labels = sample_point(full_seg_pred_array, num=1)
            #     seed_points = findCenterCoord(seed_points[None], None, self.patch_shape, self.fov_shape)
            # else:
            #     seed_points, labels = sample_point(overlap_patch, num=1)
            # patch_mask, max_id = self.ffn_inference(self.ffn, full_seg_pred_array, feature, overlap_patch, max_id, i)
            if gt is not None:
                patch_mask, max_id = self.ffn_inference_fixedsplit(self.ffn, patch_gt,
                                                                # feature.detach().cpu().numpy(),
                                                                overlap_patch, max_id, i, self.logs_path, False)
            else:
                patch_mask, max_id = self.ffn_inference_fixedsplit(self.ffn, full_seg_pred_array,
                                                                # feature.detach().cpu().numpy(),
                                                                overlap_patch, max_id, i, self.logs_path, False)


            # 更新ins_mask
            set_data(ins_mask, seed, patch_mask, False)

            # # vis
            # full_seg_path = os.path.join("/media/fcheng/NeuralTrackcf/eval", "patch_"+str(i)+'.tif')
            path = self.logs_path
            # tifffile.imsave(full_seg_path, full_seg_pred_array.astype(np.float32))
            # tifffile.imsave(os.path.join(path, "patch_tif_"+str(i)+'.tif'), patch.astype(np.float32)[0,0,...])
            # tifffile.imsave(os.path.join(path, "patchmask_tif_"+str(i)+'.tif'), patch_mask.astype(np.float32))
            # print(f"'patch_{str(i)}.tif have been saved.")

            tifffile.imsave(os.path.join(path, "ins_mask_"+str(i)+'.tif'), ins_mask.astype(np.float16))
        print("max_id:", max_id)
        return ins_mask

    def ffn_inference_fixedsplit(self, model, full_seg, overlap_p, max_id, idx, logs_path, vis=False):
        fov_ol_delta = 5
        fov_coords = self.split_data(self.patch_shape, self.fov_shape, overlap=fov_ol_delta)
        fov_coords = fov_coords + self.fov_shape//2-1 + self.fov_shape%2
        
        # full_seg 方向场
        direct_field = batch_direct_field3D(full_seg)

        seg_pad = self.pad_data(full_seg[0], self.fov_shape, fov_ol_delta)
        # feature_pad = self.pad_data(feature, self.fov_shape, fov_ol_delta)
        overlap_pad = self.pad_data(overlap_p, self.fov_shape, fov_ol_delta)
        df_pad = self.pad_data(direct_field[0], self.fov_shape, fov_ol_delta)

        # 分割结果
        patch_mask = np.zeros(overlap_pad.shape)
        seed_mask = np.zeros(overlap_pad.shape)
        if np.sum(overlap_pad) != 0:
            # patch_mask[overlap_pad>0] = overlap_pad[overlap_pad>0]
            seed_mask[overlap_pad>0] = overlap_pad[overlap_pad>0]

        curr_id = max_id
        count = 0
        for coord in fov_coords:

            # fov_mask = np.zeros(self.fov_shape)
            # fov_feature = fromSeed2Data(coord, feature_pad, self.fov_shape)
            seg_map = fromSeed2Data(coord, seg_pad, self.fov_shape)
            df_map = fromSeed2Data(coord, df_pad, self.fov_shape)
            
            # fov_feature = torch.from_numpy(fov_feature).to(self.device).float()
            seg_map = torch.from_numpy(seg_map).to(self.device).float()[None, None]
            df_map = torch.from_numpy(df_map).to(self.device).float()[None]
            
            done = set()
            while condition(patch_mask, seg_pad, coord, self.fov_shape):
                label, l_c = getfovLabel(coord, patch_mask, seg_pad, seed_mask, self.fov_shape)
                
                # 判断当前种子点 是否被访问过
                if tuple(l_c) in done:
                    print(l_c, curr_id, label)
                    patch_mask[..., l_c[0], l_c[1], l_c[2]] = curr_id
                    seed_mask[..., l_c[0], l_c[1], l_c[2]] = curr_id
                    continue
                done.add(tuple(l_c))

                label = curr_id + 1 if label == 0 else label
                print(label)
                curr_id = max(curr_id, label)

                seed_map = CenterGaussianHeatMap(self.fov_shape, l_c)
                seed_map = seed_map[None, None]

                # ffn网络forward
                seed_map = torch.from_numpy(seed_map).to(self.device).float()

                fov_data = torch.cat([seg_map, df_map, seed_map], dim=1)
                fov_output = self.ffn(fov_data)

                fov_prob, fov_pred = torch.max(fov_output, dim=1)
                fov_pred_array = fov_pred.detach().cpu().numpy()
                fov_pred_array[fov_pred_array>0] = label

                set_data(patch_mask, coord, np.squeeze(fov_pred_array), covered=False)
                set_data(seed_mask, coord, np.squeeze(fov_pred_array), covered=False)
                
                if count % 10 == 0 and vis:
                    # vis
                    # path = "/media/fcheng/NeuralTrackcf/eval"
                    path = logs_path
                    fov_path = os.path.join(path, "fov_"+str(idx)+"_"+str(count)+'.tif')
                    tifffile.imsave(fov_path, fov_pred_array.astype(np.float32)[0,...])
                    tifffile.imsave(os.path.join(path, "fov_seg_"+str(idx)+'_'+str(count)+'.tif'), seg_map.detach().cpu().numpy().astype(np.float32)[0,0,...])
                    print(f"'fov_{str(idx)}.tif have been saved.")
                count += 1
        return patch_mask[..., :self.patch_shape[0], :self.patch_shape[1], :self.patch_shape[2]], curr_id


## *********************** ##
#############################
def condition(patch_mask, seg_pad, coord, fov_shape):
    fov_mask = fromSeed2Data(coord, patch_mask, fov_shape)
    fov_seg = fromSeed2Data(coord, seg_pad, fov_shape)
    
    mask1 = np.zeros(fov_mask.shape)
    mask1[fov_seg>0] = fov_mask[fov_seg>0]
    # fov_mask[fov_seg==0] = 0
    # c = (not np.all((mask1>0) & (fov_seg>0))) and (np.any(fov_seg>0))
    mask1 = dilation(mask1, cube(3))
    c = ((mask1==0) & (fov_seg>0)).astype(int)
    return np.sum(c)

def fromSeed2fullData(seed_coord, shape, patch_shape):
    """ 以seed为中心, 在shape大小的mask中, 产生patch_shape的全1 mask
    """
    full = np.zeros(shape)
    
    sel = [slice(max(s, 0), e+1) for s, e in zip(
                np.array(seed_coord)+1-np.array(patch_shape)//2-np.array(patch_shape)%2,
                np.array(seed_coord)+np.array(patch_shape)//2)]

    if len(sel) == 2:
        full[..., sel[0], sel[1]] = 1
    elif len(sel) == 3:
        full[..., sel[0], sel[1], sel[2]] = 1
    else:
        print("the data have shape of {}".format(shape))
    return full

def CenterGaussianHeatMap(shape, center, sigma=1):
    dims = len(shape)
    lins = []
    for i in range(dims):
        lins += [np.linspace(0, shape[i]-1, shape[i]).tolist()]
    
    coords = np.stack(np.meshgrid(*lins), axis=-1)
    D = 0.5 * np.sum(np.power(coords - center, 2), axis=-1)
    E = 2.0 * sigma * sigma
    Exponent = D / E
    heatmap = np.exp(-Exponent)
    return heatmap.swapaxes(0, 1)

def findNextcoord(coord, full_shape, patch_shape):
    """ 使seed位置尽量在中心, 当seed靠近full_data的边缘时,
        使外表面重合
    """

    ranges = [(max(s, 0), min(e+1, full_shape[0])) for s, e in zip(
                np.array(coord)+1-np.array(patch_shape)//2-np.array(patch_shape)%2,
                np.array(coord)+np.array(patch_shape)//2)]
    center = []
    for i, r in enumerate(ranges):
        if r[1] - r[0] < patch_shape[i]:
            if r[0] == 0:
                # r[1] = r[1] + r[1] - coord[i] - (coord - r[0] + 1)
                center.append( r[0] + np.array(patch_shape[i])//2 -1 + np.array(patch_shape[i])%2 )
            elif r[1] == full_shape[0]:
                # r[0] = r[0] - (coord[i] - r[0] - (r[1] - coord[i]))
                center.append( r[1] - np.array(patch_shape[i])//2 - 1 )
        else:
            center.append(coord[i])
        # ranges_.append([r[0], r[1]])
    # center = [(r[0]+r[1])//2-1+(r[0]+r[1])%2 for r in ranges_]
    return center

if __name__ == "__main__":
    # test split_data
    shape = [300, 300]
    patch_shape = [96, 72]
    overlap = 10
    data = np.ones(shape)
    idx = split_data(shape, patch_shape, overlap)
    d = pad_data(data, patch_shape, overlap)




