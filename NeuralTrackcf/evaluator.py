import time
import numpy as np

import torch
import torch.nn as nn

from neuralTrack.utils.meters import AverageMeter
from utils.utils import sample_batch, get_batchCropData, get_batchSeedMap
from utils.utils import trans3Dto2D
from utils.direct_field import batch_direct_field3D


class BaseEvaluator():
    def __init__(self, model, criterion, metric_func=None, device="cuda"):
        super(BaseEvaluator,self).__init__()
        self.device = torch.device(device)
        self.model = model
        self.criterion = criterion
        self.metric_func = metric_func
    
    def evaluate(self, data_loader, step, print_freq=1, tfLogger=None):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        for i, input in enumerate(data_loader):
            start = time.time()

            img, gts = self._parse_data(input)
            batch_size = img.size(0)

            data_time.update(time.time() - start)

            with torch.no_grad():
                loss, metrics, vis = self._forward(img, gts)

            losses.update(loss.item(),batch_size)

            dices, precs, recalls = metrics

            batch_time.update(time.time() - start)
            
            start = time.time()
            if (i + 1) % print_freq == 0:
                print('[{}/{}]  '
                      'batch_time:{:.3f}s({:.3f})  '
                      'data:{:.3f}s({:.3f})  '
                      'Loss:{:.3f}({:.3f})'
                    .format(i+1,len(data_loader),
                        batch_time.val,batch_time.avg,
                        data_time.val,data_time.avg,
                        losses.val,losses.avg,
                        ))
        if tfLogger is not None:
            infos = {"loss":loss.item()}
            
            for dice in dices:
                infos[dice.infos] = dice.avg
            for prec in precs:
                infos[prec.infos] = prec.avg
            for recall in recalls:
                infos[recall.infos] = recall.avg

            for tag,value in infos.items():
                tfLogger.scalar_summary(tag, value, step)


        if tfLogger is not None:
            vis_info = {}
            vis_info["full_seg"] = vis[0]
            vis_info["full_seg_gt"] = vis[1]
            vis_info["fov_seg"] = vis[2]
            vis_info["fov_seg_gt"] = vis[3]

            for tag, value in vis_info.items():
                tfLogger.image_summary(tag, trans3Dto2D(value), step)

        print("Evaluator finished ! the evaluate results is: ")
        for dice, prec, recall in zip(dices, precs, recalls):
            print("{} {:.2%} {} {:.2%} {} {:.2%}".format(\
                    dice.infos, dice.avg,\
                    prec.infos, prec.avg,\
                    recall.infos, recall.avg))
        print("\n")
        flag_value = np.mean([x.avg for x in dices]) 
        return flag_value
        
    def _parse_data(self, input):
        NotImplemented
    
    def _forward(self, img, gts):
        NotImplemented

class SegFFNNetEvaluator(BaseEvaluator):
    def __init__(self, model, criterion, metric_func, fov_shape, device="cuda"):
        super(SegFFNNetEvaluator, self).__init__(model, criterion, metric_func, device)
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
        img, ins, _ = input
        seed_coords, labels_pad = sample_batch(ins.detach().cpu().numpy(), self.fov_shape, 1)  # (max_label, N, 3)

        img = img.to(self.device)
        ins = ins.to(self.device)
        return img, [ins, seed_coords, labels_pad]

    def _forward(self, imgs, gts):
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
        for seed, label in zip(seeds, labels_pad):
            fov_feature = get_batchCropData(seed, feature.detach().cpu().numpy(), self.fov_shape)
            seg_map = get_batchCropData(seed, np.expand_dims(full_seg_pred_array, 1), self.fov_shape)
            seed_map = np.expand_dims(get_batchSeedMap(seed, label, ins.detach().cpu().numpy(), self.fov_shape), 1)

            fov_ins = get_batchCropData(seed, ins.detach().cpu().numpy(), self.fov_shape)
            fov_gt = (fov_ins == label[:, None, None, None]).astype(int)
            fov_gt = torch.from_numpy(fov_gt).to(self.device)

            fov_feature = torch.from_numpy(fov_feature).to(self.device).float()
            seg_map = torch.from_numpy(seg_map).to(self.device).float()
            seed_map = torch.from_numpy(seed_map).to(self.device).float()

            fov_data = torch.cat([fov_feature, seg_map, seed_map], dim=1)

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



class DirectFieldEvaluator(BaseEvaluator):
    def __init__(self, model, criterion, metric_func, fov_shape, device="cuda"):
        super(DirectFieldEvaluator, self).__init__(model, criterion, metric_func, device)
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
        # self.featureHead = self.model.module.featureHead
        # self.segHead = self.model.module.segHead
        # self.fusenet = self.model.module.fuseNet
        # self.ffn = self.model.module.ffn
    
    def _parse_data(self, input):
        img, ins, _ = input
        seed_coords, labels_pad = sample_batch(ins.detach().cpu().numpy(), self.fov_shape, 1)  # (max_label, N, 3)

        img = img.to(self.device)
        ins = ins.to(self.device)
        return img, [ins, seed_coords, labels_pad]

    def _forward(self, imgs, gts):
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
        # feature = self.fusenet(rrb)

        _, full_seg_pred = torch.max(full_seg_output, dim = 1)

        full_seg_array = full_seg_gt.detach().cpu().numpy()
        full_seg_pred_array = full_seg_pred.detach().cpu().numpy()
        metric_full_seg = self.metric_func(full_seg_pred_array, full_seg_array)

        loss = self.criterion(full_seg_output, full_seg_gt)

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
            fov_gt = (fov_ins == label[:, None, None, None]).astype(int)
            fov_gt = torch.from_numpy(fov_gt).to(self.device)

            # fov_feature = torch.from_numpy(fov_feature).to(self.device).float()
            seg_map = torch.from_numpy(seg_map).to(self.device).float()
            seed_map = torch.from_numpy(seed_map).to(self.device).float()
            df = torch.from_numpy(df).to(self.device).float()

            # fov_data = torch.cat([fov_feature, seg_map, df, seed_map], dim=1)
            fov_data = torch.cat([seg_map, df, seed_map], dim=1)

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
