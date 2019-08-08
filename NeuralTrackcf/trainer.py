import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralTrack.utils.meters import AverageMeter
from utils.utils import sample_batch, get_batchCropData, get_batchSeedMap, sample_seedfromfullseg
from utils.utils import trans3Dto2D
from utils.direct_field import batch_direct_field3D
from neuralTrack.utils.cluster_utils import emb_finch
import pdb

class BaseTrainer():
    def __init__(self, model, criterion, logs_path, metric_func=None, device="cuda"):
        self.device = torch.device(device)
        self.model = model
        self.criterion = criterion
        self.logs_path = logs_path
        self.metric_func = metric_func
    
    def train(self, epoch, data_loader, optimizer, current_lr=0., print_freq=1, tfLogger=None):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        for i, input in enumerate(data_loader):
            start = time.time()

            img, gts = self._parse_data(input)
            batch_size = img.size(0)

            data_time.update(time.time() - start)

            loss, metrics, vis = self._forward(img, gts)
            
            dices,precs,recalls = metrics
            losses.update(loss.item(),batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - start)

            start = time.time()
            if (i + 1) % print_freq == 0:
                print('epoch:[{}][{}/{}]  '
                      'batch_time:{:.3f}s({:.3f})  '
                      'data:{:.3f}s({:.3f})  '
                      'Loss:{:.3f}({:.3f})'
                    .format(epoch,i+1,len(data_loader),
                        batch_time.val,batch_time.avg,
                        data_time.val,data_time.avg,
                        losses.val,losses.avg,
                        ))
            
            if (i+1) % (print_freq*10) == 0:
                if tfLogger is not None:
                    step = epoch*len(data_loader) + (i+1)
                    infos = {"lr":current_lr,
                             "loss":loss.item()}
                    
                    for dice in dices:
                        infos[dice.infos] = dice.avg
                    for prec in precs:
                        infos[prec.infos] = prec.avg
                    for recall in recalls:
                        infos[recall.infos] = recall.avg

                    for tag,value in infos.items():
                        tfLogger.scalar_summary(tag, value, step)
            
            if (i+1) % (print_freq*20) == 0:
                if tfLogger is not None:
                    vis_info = {}
                    vis_info["full_seg"] = vis[0]
                    vis_info["full_seg_gt"] = vis[1]
                    vis_info["fov_seg"] = vis[2]
                    vis_info["fov_seg_gt"] = vis[3]

                    for tag, value in vis_info.items():
                        tfLogger.image_summary(tag, trans3Dto2D(value), step)

        print("Training Finished in Epoch {}".format(epoch))
        
        for dice, prec, recall in zip(dices, precs, recalls):
            print("{} {:.2%} {} {:.2%} {} {:.2%}".format(\
                    dice.infos, dice.avg,\
                    prec.infos, prec.avg,\
                    recall.infos, recall.avg))
        print("\n")
        
    def _parse_data(self, input):
        NotImplemented
    
    def _forward(self, img, gts):
        NotImplemented

class SegFFNNetTrainer(BaseTrainer):
    def __init__(self, model, criterion, logs_path, metric_func, fov_shape, device="cuda"):
        super(SegFFNNetTrainer, self).__init__(model, criterion, logs_path, metric_func, device)
        self.fov_shape = fov_shape

        self.dice_ins = AverageMeter("dice_ins")
        self.prec_ins = AverageMeter("prec_ins")
        self.recall_ins = AverageMeter("recall_ins")

        self.dice_patch = AverageMeter("dice_patch")
        self.prec_patch = AverageMeter("prec_patch")
        self.recall_patch = AverageMeter("recall_patch")

        self.featureHead = nn.DataParallel(self.model.featureHead)
        self.segHead = nn.DataParallel(self.model.segHead)
        self.fusenet = nn.DataParallel(self.model.fuseNet)
        self.ffn = nn.DataParallel(self.model.ffn)
        
        # self.featureHead = self.model.module.featureHead
        # self.segHead = self.model.module.segHead
        # self.fusenet = self.model.module.fuseNet
        # self.ffn = self.model.module.ffn

    def _parse_data(self, input):
        img, ins, info = input
        seed_coords, labels_pad = sample_batch(ins.detach().cpu().numpy(), self.fov_shape, 1)  # (max_label, N, 3)
        # print("*"*10)
        # print(seed_coords)
        # print(labels_pad)
        # pdb.set_trace()

        img = img.to(self.device)
        ins = ins.to(self.device)
        return img, [ins, seed_coords, labels_pad, info]
    
    def _forward(self, imgs, gts):
        """
            cur_img: B * 1 * h * w * d
            cur_ins: B * h * w * d
            labels_pad: max_ins_length * B
        """
        batch_size = imgs.size(0)
        ins, seeds, labels_pad, infos = gts
        
        full_seg_gt = (ins > 0).long()

        rrb = torch.cat(self.featureHead(imgs), dim=1)

        full_seg_output = self.segHead(rrb)
        feature = self.fusenet(rrb)

        _, full_seg_pred = torch.max(full_seg_output, dim = 1)

        full_seg_array = full_seg_gt.detach().cpu().numpy()
        full_seg_pred_array = full_seg_pred.detach().cpu().numpy()
        metric_full_seg = self.metric_func(full_seg_pred_array, full_seg_array)

        loss = self.criterion(full_seg_output, full_seg_gt)

        loss_ins = 0.
        num_seeds = seeds.shape[0]
        metrics_fov = []
        for seed, label in zip(seeds, labels_pad):
            fov_feature = get_batchCropData(seed, feature.detach().cpu().numpy(), self.fov_shape)
            seg_map = get_batchCropData(seed, np.expand_dims(full_seg_pred_array, 1), self.fov_shape)
            seed_map = np.expand_dims(get_batchSeedMap(seed, label, ins.detach().cpu().numpy(), self.fov_shape), 1)

            fov_ins = get_batchCropData(seed, ins.detach().cpu().numpy(), self.fov_shape)
            if np.any(label==0):
                label_ = np.array([l+1 if l==0 else l for l in label])
                try:
                    fov_gt = (fov_ins == label_[:, None, None, None]).astype(int)
                except AttributeError:
                    pdb.set_trace()
            else:
                fov_gt = (fov_ins == label[:, None, None, None]).astype(int)
            fov_gt = torch.from_numpy(fov_gt).to(self.device)

            fov_feature = torch.from_numpy(fov_feature).to(self.device).float()
            seg_map = torch.from_numpy(seg_map).to(self.device).float()
            seed_map = torch.from_numpy(seed_map).to(self.device).float()
            try:
                fov_data = torch.cat([fov_feature, seg_map, seed_map], dim=1)
            except RuntimeError:
                pdb.set_trace()
            fov_output = self.ffn(fov_data)
            
            curr_loss = self.criterion(fov_output, fov_gt)
            loss_ins += curr_loss

            _, fov_seg = torch.max(fov_output, dim=1)
            fov_seg_array = fov_seg.detach().cpu().numpy()
            fov_gt_array = fov_gt.detach().cpu().numpy()
            metric_seg = self.metric_func(fov_seg_array, fov_gt_array)
            metrics_fov.append(metric_seg)

            # 寻找异常loss
            if curr_loss > 3:
                # curr_loss.item()
                log = []
                for i_, (i0, i1, i2, i3) in enumerate(zip(infos[0], infos[1], infos[2], infos[3])):
                    log.append([i0, [i1.item(), i2.item(), i3.item()]] +\
                           [seed[i_].tolist()] + [label[i_]])  # ['tifname', [patch_seed], [fov_seed], label]
                print("*"*10)
                with open(os.path.join(self.logs_path, "odd_error.txt"), 'a') as f:
                    print(log, "loss:", curr_loss.item(), file=f)

        loss_ins /= num_seeds
        loss += loss_ins

        metric_fov = np.mean(metrics_fov, axis=0)

        self.dice_ins.update(metric_fov[0], batch_size)
        self.prec_ins.update(metric_fov[1], batch_size)
        self.recall_ins.update(metric_fov[2], batch_size)

        self.dice_patch.update(metric_full_seg[0], batch_size)
        self.prec_patch.update(metric_full_seg[1], batch_size)
        self.recall_patch.update(metric_full_seg[2], batch_size)

        dices = [self.dice_patch, self.dice_ins]
        precs = [self.prec_patch, self.prec_ins]
        recalls = [self.recall_patch, self.recall_ins]

        return loss, [dices, precs, recalls], [full_seg_pred_array, full_seg_array, fov_seg_array, fov_gt_array]

    def forward_seedfromfullseg(self, imgs, gts):
        """
            cur_img: B * 1 * h * w * d
            cur_ins: B * h * w * d
            labels_pad: max_ins_length * B
        """
        batch_size = imgs.size(0)
        ins, seeds, labels_pad = gts

        full_seg_gt = (ins > 0).long()

        rrb = torch.cat(self.featureHead(imgs), dim=1)

        full_seg_output = self.segHead(rrb)
        feature = self.fusenet(rrb)

        _, full_seg_pred = torch.max(full_seg_output, dim = 1)

        full_seg_array = full_seg_gt.detach().cpu().numpy()
        full_seg_pred_array = full_seg_pred.detach().cpu().numpy()
        metric_full_seg = self.metric_func(full_seg_pred_array, full_seg_array)

        loss = self.criterion(full_seg_output, full_seg_gt)

        loss_ins = 0.
        num_seeds = seeds.shape[0]
        metrics_fov = []
        seeds_center, seeds_coords, labels = sample_seedfromfullseg(full_seg_pred_array, ins.detach().cpu().numpy(), labels_pad, self.fov_shape)
        for seed, label, seed_coord in zip(seeds_center, labels, seeds_coords):
            fov_feature = get_batchCropData(seed, feature.detach().cpu().numpy(), self.fov_shape)
            seg_map = get_batchCropData(seed, np.expand_dims(full_seg_pred_array, 1), self.fov_shape)
            seed_map = np.expand_dims(get_batchSeedMap(seed, label, ins.detach().cpu().numpy(), self.fov_shape), 1)

            fov_ins = get_batchCropData(seed, ins.detach().cpu().numpy(), self.fov_shape)
            if np.any(label==0):
                label_ = np.array([l+1 if l==0 else l for l in label])
                try:
                    fov_gt = (fov_ins == label_[:, None, None, None]).astype(int)
                except AttributeError:
                    pdb.set_trace()
            else:
                fov_gt = (fov_ins == label[:, None, None, None]).astype(int)
            fov_gt = torch.from_numpy(fov_gt).to(self.device)

            fov_feature = torch.from_numpy(fov_feature).to(self.device).float()
            seg_map = torch.from_numpy(seg_map).to(self.device).float()
            seed_map = torch.from_numpy(seed_map).to(self.device).float()
            try:
                fov_data = torch.cat([fov_feature, seg_map, seed_map], dim=1)
            except RuntimeError:
                pdb.set_trace()
            fov_output = self.ffn(fov_data)

            loss_ins += self.criterion(fov_output, fov_gt)

            _, fov_seg = torch.max(fov_output, dim=1)
            fov_seg_array = fov_seg.detach().cpu().numpy()
            fov_gt_array = fov_gt.detach().cpu().numpy()
            metric_seg = self.metric_func(fov_seg_array, fov_gt_array)
            metrics_fov.append(metric_seg)

        loss_ins /= num_seeds
        loss += loss_ins

        metric_fov = np.mean(metrics_fov, axis=0)

        self.dice_ins.update(metric_fov[0], batch_size)
        self.prec_ins.update(metric_fov[1], batch_size)
        self.recall_ins.update(metric_fov[2], batch_size)

        self.dice_patch.update(metric_full_seg[0], batch_size)
        self.prec_patch.update(metric_full_seg[1], batch_size)
        self.recall_patch.update(metric_full_seg[2], batch_size)

        dices = [self.dice_patch, self.dice_ins]
        precs = [self.prec_patch, self.prec_ins]
        recalls = [self.recall_patch, self.recall_ins]

        return loss, [dices, precs, recalls], [full_seg_pred_array, full_seg_array, fov_seg_array, fov_gt_array]

class SegFFNNetEmbTrainer(BaseTrainer):
    def __init__(self, model, criterion, logs_path, metric_func, dsc_loss, fov_shape, device="cuda"):
        super(SegFFNNetTrainer, self).__init__(model, criterion, logs_path, metric_func, device)
        self.fov_shape = fov_shape

        self.dice_ins = AverageMeter("dice_ins")
        self.prec_ins = AverageMeter("prec_ins")
        self.recall_ins = AverageMeter("recall_ins")

        self.dice_patch = AverageMeter("dice_patch")
        self.prec_patch = AverageMeter("prec_patch")
        self.recall_patch = AverageMeter("recall_patch")

        self.dsc_loss = dsc_loss
        self.l2_loss = nn.MSELoss()

        self.featureHead = self.model.module.featureHead
        self.segHead = self.model.module.segHead
        self.embHead = self.model.module.embHead
        self.matchHead = self.model.module.matchHead
        self.fusenet = self.model.module.fuseNet
        self.ffn = self.model.module.ffn

    def anchor_sel(self, emb, ins, labels):
        """ emb: (N, channels, H, W, D)
            ins: (N, H, W, D)
            labels: (N, )
        """
        anchors_list = []
        for e, i, l in zip(emb, ins, labels):
            mask = ins == l
            if np.sum(mask) > 2:
                ins_list, labels_list = emb_finch(emb, mask)  # emb选出当前实例中有代表性的一些点
                ins_sel = ins_list[-1]
                labels_sel = labels_list[-1]
                anchor_label = np.argmax([np.sum(ins_sel==x) for x in labels_sel]) + 1
                anchor = (ins_sel == anchor_label).astype(int)  # 选出当前聚类结果中点数最多的一类, 作为anchor mask
            else:
                anchor = mask.astype(int)
            anchors_list.append(anchor)
        anchors_array = np.array(anchors_list)
        anchors = torch.from_numpy(anchors_array).to(self.device)
        return anchors

    def _forward(self, imgs, gts):
        
        batch_size = imgs.size(0)
        ins, seeds, labels_pad, _ = gts

        full_seg_gt = (ins > 0).long()

        rrb = torch.cat(self.featureHead(imgs), dim=1)
        full_seg_output = self.segHead(rrb)
        feature = self.fusenet(rrb)

        _, full_seg_pred = torch.max(full_seg_output, dim=1)

        full_seg_array = full_seg_gt.detach().cpu().numpy()
        full_seg_pred_array = full_seg_pred.detach().cpu().numpy()
        metric_full_seg = self.metric_func(full_seg_pred_array, full_seg_array)

        emb = self.embHead(rrb)

        loss = self.criterion(full_seg_output, full_seg_gt) +\
               self.dsc_loss(emb, ins)
        
        loss_ins = 0.
        num_seeds = seeds.shape[0]
        metrics_fov = []
        emb_array = emb.unsqueeze(-1).transpose(1, -1).squeeze(1).cpu().detach().numpy()
        ins_array = ins.cpu().detach().numpy()

        for seed, label in zip(seed, labels_pad):
            fov_feature = get_batchCropData(seed, feature.detach().cpu().numpy(), self.fov_shape)
            seg_map = get_batchCropData(seed, np.expand_dims(full_seg_pred_array, 1), self.fov_shape)
            # seed_map = np.expand_dims(get_batchSeedMap(seed, label, ins.detach().cpu().numpy(), self.fov_shape), 1)

            fov_gt = get_batchCropData(seed, ins.detach().cpu().numpy(), self.fov_shape)
            anchors = self.anchor_sel(emb_array, ins_array, label.cpu().numpy())

            match = self.matchHead(emb, emb, anchors)  # 根据anchor点, 找出一组相似点, 代表性点

            fs = torch.cat([match, anchors[:, None].float(),
                           seg_map[:, None].float()], dim=1)
            
            fov_output = self.ffn(fs)

            curr_loss = self.criterion(fov_output, fov_gt) +\
                        self.l2_loss(match, fov_gt[:, None].float())
            loss_ins += curr_loss

            _, fov_seg = torch.max(fov_output, dim=1)
            fov_seg_array = fov_seg.detach().cpu().numpy()
            fov_gt_array = fov_gt.detach().cpu().numpy()
            metric_seg = self.metric_func(fov_seg_array, fov_gt_array)
            metrics_fov.append(metric_seg)

        loss_ins /= num_seeds
        loss += loss_ins

        metric_fov = np.mean(metrics_fov, axis=0)

        self.dice_ins.update(metric_fov[0], batch_size)
        self.prec_ins.update(metric_fov[1], batch_size)
        self.recall_ins.update(metric_fov[2], batch_size)

        self.dice_patch.update(metric_full_seg[0], batch_size)
        self.prec_patch.update(metric_full_seg[1], batch_size)
        self.recall_patch.update(metric_full_seg[2], batch_size)

        dices = [self.dice_patch, self.dice_ins]
        precs = [self.prec_patch, self.prec_ins]
        recalls = [self.recall_patch, self.recall_ins]

        return loss, [dices, precs, recalls], [full_seg_pred_array, full_seg_array, fov_seg_array, fov_gt_array]

class DirectFieldTrainer(BaseTrainer):
    def __init__(self, model, criterion, logs_path, metric_func, fov_shape, device="cuda"):
        super(DirectFieldTrainer, self).__init__(model, criterion, logs_path, metric_func, device)
        self.fov_shape = fov_shape

        self.dice_ins = AverageMeter("dice_ins")
        self.prec_ins = AverageMeter("prec_ins")
        self.recall_ins = AverageMeter("recall_ins")

        self.dice_patch = AverageMeter("dice_patch")
        self.prec_patch = AverageMeter("prec_patch")
        self.recall_patch = AverageMeter("recall_patch")

        self.featureHead = nn.DataParallel(self.model.featureHead)
        self.segHead = nn.DataParallel(self.model.segHead)
        # self.fusenet = nn.DataParallel(self.model.fuseNet)
        self.ffn = nn.DataParallel(self.model.ffn)

    def _parse_data(self, input):
        img, ins, info = input
        seed_coords, labels_pad = sample_batch(ins.detach().cpu().numpy(), self.fov_shape, 1)  # (max_label, N, 3)
        # print("*"*10)
        # print(seed_coords)
        # print(labels_pad)
        # pdb.set_trace()

        img = img.to(self.device)
        ins = ins.to(self.device)
        return img, [ins, seed_coords, labels_pad, info]

    def _forward(self, imgs, gts):
        """
            cur_img: B * 1 * h * w * d
            cur_ins: B * h * w * d
            labels_pad: max_ins_length * B
        """
        batch_size = imgs.size(0)
        ins, seeds, labels_pad, infos = gts
        
        full_seg_gt = (ins > 0).long()

        rrb = torch.cat(self.featureHead(imgs), dim=1)

        full_seg_output = self.segHead(rrb)
        # feature = self.fusenet(rrb)

        _, full_seg_pred = torch.max(full_seg_output, dim = 1)

        full_seg_array = full_seg_gt.detach().cpu().numpy()
        full_seg_pred_array = full_seg_pred.detach().cpu().numpy()
        metric_full_seg = self.metric_func(full_seg_pred_array, full_seg_array)

        loss = self.criterion(full_seg_output.float(), full_seg_gt)

        # direction field
        direct_field = batch_direct_field3D(full_seg_array)

        loss_ins = 0.
        num_seeds = seeds.shape[0]
        metrics_fov = []
        for seed, label in zip(seeds, labels_pad):
            # fov_feature = get_batchCropData(seed, feature.detach().cpu().numpy(), self.fov_shape)
            seg_map = get_batchCropData(seed, np.expand_dims(full_seg_pred_array, 1), self.fov_shape)
            seed_map = np.expand_dims(get_batchSeedMap(seed, label, ins.detach().cpu().numpy(), self.fov_shape), 1)
            df = get_batchCropData(seed, direct_field, self.fov_shape)

            fov_ins = get_batchCropData(seed, ins.detach().cpu().numpy(), self.fov_shape)
            if np.any(label==0):
                label_ = np.array([l+1 if l==0 else l for l in label])
                try:
                    fov_gt = (fov_ins == label_[:, None, None, None]).astype(int)
                except AttributeError:
                    pdb.set_trace()
            else:
                fov_gt = (fov_ins == label[:, None, None, None]).astype(int)
            fov_gt = torch.from_numpy(fov_gt).to(self.device)

            # fov_feature = torch.from_numpy(fov_feature).to(device=self.device, dtype=torch.float)
            seg_map = torch.from_numpy(seg_map).to(device=self.device, dtype=torch.float)
            seed_map = torch.from_numpy(seed_map).to(device=self.device, dtype=torch.float)
            df = torch.from_numpy(df).to(device=self.device, dtype=torch.float)

            # fov_data = torch.cat([fov_feature, seg_map, df, seed_map], dim=1)
            fov_data = torch.cat([seg_map, df, seed_map], dim=1)

            fov_output = self.ffn(fov_data)
            
            curr_loss = self.criterion(fov_output, fov_gt)
            loss_ins += curr_loss

            _, fov_seg = torch.max(fov_output, dim=1)
            fov_seg_array = fov_seg.detach().cpu().numpy()
            fov_gt_array = fov_gt.detach().cpu().numpy()
            metric_seg = self.metric_func(fov_seg_array, fov_gt_array)
            metrics_fov.append(metric_seg)

            # 寻找异常loss
            # if curr_loss > 3:
            #     # curr_loss.item()
            #     log = []
            #     for i_, (i0, i1, i2, i3) in enumerate(zip(infos[0], infos[1], infos[2], infos[3])):
            #         log.append([i0, [i1.item(), i2.item(), i3.item()]] +\
            #                [seed[i_].tolist()] + [label[i_]])  # ['tifname', [patch_seed], [fov_seed], label]
            #     print("*"*10)
            #     with open(os.path.join(self.logs_path, "odd_error.txt"), 'a') as f:
            #         print(log, "loss:", curr_loss.item(), file=f)

        loss_ins /= num_seeds
        loss = 0.3*loss + 0.7*loss_ins

        metric_fov = np.mean(metrics_fov, axis=0)

        self.dice_ins.update(metric_fov[0], batch_size)
        self.prec_ins.update(metric_fov[1], batch_size)
        self.recall_ins.update(metric_fov[2], batch_size)

        self.dice_patch.update(metric_full_seg[0], batch_size)
        self.prec_patch.update(metric_full_seg[1], batch_size)
        self.recall_patch.update(metric_full_seg[2], batch_size)

        dices = [self.dice_patch, self.dice_ins]
        precs = [self.prec_patch, self.prec_ins]
        recalls = [self.recall_patch, self.recall_ins]

        return loss, [dices, precs, recalls], [full_seg_pred_array, full_seg_array, fov_seg_array, fov_gt_array]






        

        

