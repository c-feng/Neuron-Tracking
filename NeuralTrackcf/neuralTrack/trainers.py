import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import MeanShift
from skimage.morphology import remove_small_objects,skeletonize_3d,dilation,cube,ball
from multiprocessing import Pool
from skimage.external import tifffile

from .utils.meters import AverageMeter
  
from .utils.direct_field import batch_direct_field_cal
from .utils.hungarian import match, softIoU
from .utils.canvas import Canvas,CanvasBatch
from .loss.loss_utils import VoxLoss, VoxEndsLoss
from .utils.fov_utils import batch_fov_update



class BaseTrainer(object):
    def __init__(self, model, criterion, metric_func):
        super(BaseTrainer,self).__init__()
        self.device = torch.device("cuda")
        self.model = model 
        self.criterion = criterion
        self.metric_func = metric_func

    def train(self, epoch, data_loader, optimizer,current_lr = 0.0, 
            print_freq=1, tfLogger=None):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        for i,input_ in enumerate(data_loader):
            #if i > 100:break
            end = time.time()

            img,gts = self._parse_data(input_)
            batch_size = img.size(0)

            data_time.update(time.time() - end)

            loss,metrics = self._forward(img,*gts)

            losses.update(loss.item(),batch_size)
            
            dices,precs,recalls = metrics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)

            end = time.time()
            if (i + 1)%print_freq ==0:
                print('epoch: [{}][{}/{}]\t'
                    'batch_time: {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    .format(epoch,i+1,len(data_loader),
                        batch_time.val,batch_time.avg,
                        data_time.val,data_time.avg,
                        losses.val,losses.avg,
                        ))

            if (i+1)%(print_freq*10) ==0:
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
 
        print("Training Finished in Epoch {}".format(epoch))

        for dice, prec, recall in zip(dices, precs, recalls):
            print("{} {:.2%} {} {:.2%} {} {:.2%}".format(\
                    dice.infos, dice.avg,\
                    prec.infos, prec.avg,\
                    recall.infos, recall.avg))
        print("\n")

    def _parse_data(self,input_):
        pass
    def _forward(self,img,gts):
        pass

class VoxTrainer(BaseTrainer):
    def __init__(self,model,criterion,metric_func):
        super(VoxTrainer,self).__init__(model, criterion, metric_func)
        self.dice_vox = AverageMeter("dice_vox")
        self.prec_vox = AverageMeter("prec_vox")
        self.recall_vox = AverageMeter("recall_vox")


        
    def _parse_data(self,input_):
        device = self.device 
        img,gts = input_

        img = img.to(device)
        gts = gts.to(device)

        img_gt = gts[:,0]

        return img,[img_gt]

    def _forward(self,img,img_gt):
        assert isinstance(self.criterion, VoxLoss)
        batch_size = img.size(0)
        num_classes_vox = 2
        bg = 0

        outputs = self.model(img)
    
        loss = 0.0
        #loss = self.criterion(vox_output, img_gt)
        for vox_output in outputs:
            loss += self.criterion(vox_output[:,:2], img_gt)

        vox_output = outputs[0][:,:2]
        _, vox_pred = torch.max(vox_output, 1)
        vox_pred = vox_pred.cpu().numpy()
        vox_label = img_gt.cpu().numpy()

        metric = self.metric_func(vox_pred, vox_label, num_classes = num_classes_vox, bg = bg)
        self.dice_vox.update(metric[0], batch_size)
        self.prec_vox.update(metric[1], batch_size)
        self.recall_vox.update(metric[2], batch_size)

        dices = [self.dice_vox]
        precs = [self.prec_vox]
        recalls = [self.recall_vox]
        return loss,[dices,precs,recalls]

class VoxEndsMultiTrainer(VoxTrainer):
    def ends_gt_dilation(self, batch_segs_gt_array, batch_ends_gt_array, thres = 3):
        batch_ends_gt_d = []

        for segs_gt, ends_gt in zip(batch_segs_gt_array, batch_ends_gt_array):
            ends_gt_d = dilation(ends_gt, ball(thres))
            ends_gt_d[segs_gt == 0] = 0

            batch_ends_gt_d.append(ends_gt_d[None])
        batch_ends_gt_d = np.concatenate(batch_ends_gt_d, axis = 0)
        return batch_ends_gt_d

    def _parse_data(self,input_):
        device = self.device 
        img,gts = input_

        img_gt = gts[:,0]
        img_gt_array = img_gt.numpy()
        ends_gt_array = gts[:,1].numpy()
        ends_gt_d_array = self.ends_gt_dilation(img_gt_array, ends_gt_array) 
        ends_gt_d = torch.from_numpy(ends_gt_d_array).long()

        img_gt[ends_gt_d > 0] = ends_gt_d[ends_gt_d > 0] + 1

        img = img.to(device)
        img_gt = img_gt.to(device)
        ends_gt_d = ends_gt_d.to(device)

        return img,[img_gt]

    def _forward(self,img,img_gt):
        assert isinstance(self.criterion, VoxLoss)
        batch_size = img.size(0)
        num_classes_vox = 4
        bg = 0

        outputs = self.model(img)
    
        loss = 0.0
        #loss = self.criterion(vox_output, img_gt)
        for vox_output in outputs:
            loss += self.criterion(vox_output[:,:4], img_gt)

        vox_output = outputs[0][:,:4]
        _, vox_pred = torch.max(vox_output, 1)
        vox_pred = vox_pred.cpu().numpy()
        vox_label = img_gt.cpu().numpy()

        metric = self.metric_func(vox_pred, vox_label, num_classes = num_classes_vox, bg = bg)
        self.dice_vox.update(metric[0], batch_size)
        self.prec_vox.update(metric[1], batch_size)
        self.recall_vox.update(metric[2], batch_size)

        dices = [self.dice_vox]
        precs = [self.prec_vox]
        recalls = [self.recall_vox]
        return loss,[dices,precs,recalls]

class EndsNetTrainer(VoxTrainer):
    def ends_gt_dilation(self, batch_segs_gt_array, batch_ends_gt_array, thres = 2):
        batch_ends_gt_d = []
        
        mask = np.zeros(batch_ends_gt_array.shape[1:], dtype = np.bool)
        mask[0:2] = 1
        mask[-3:-1] = 1
        mask[:,0:2] = 1
        mask[:,-3:-1] = 1
        mask[:,:,0:2] = 1
        mask[:,:,-3:-1] = 1
        for segs_gt, ends_gt in zip(batch_segs_gt_array, batch_ends_gt_array):
            ends_gt_d = dilation(ends_gt, ball(thres))
            ends_gt_d[segs_gt == 0] = 0
            ends_gt_d[mask] = segs_gt[mask]

            batch_ends_gt_d.append(ends_gt_d[None])
        batch_ends_gt_d = np.concatenate(batch_ends_gt_d, axis = 0)
        return batch_ends_gt_d

    def _parse_data(self,input_):
        device = self.device 
        _, gts = input_

        img_gt = gts[:,0]
        img_gt_array = img_gt.numpy()
        ends_gt_array = gts[:,1].numpy()
        ends_gt_d_array = self.ends_gt_dilation(img_gt_array, ends_gt_array) 
        ends_gt_d = torch.from_numpy(ends_gt_d_array).long()
        
        img = img_gt.unsqueeze(1).float().to(device)

        #img_gt[ends_gt_d > 0] = ends_gt_d[ends_gt_d > 0] + 1
        #img_gt = img_gt.to(device)
        ends_gt_d = ends_gt_d.to(device)
        #print(img.size(), img_gt.size())
        return img,[ends_gt_d]

    def _forward(self,img,img_gt):
        #assert isinstance(self.criterion, VoxLoss)
        batch_size = img.size(0)
        num_classes_vox = 3
        bg = 0

        outputs = self.model(img)
    
        loss = 0.0
        #loss = self.criterion(vox_output, img_gt)
        loss = self.criterion(outputs, img_gt)

        vox_output = outputs
        _, vox_pred = torch.max(vox_output, 1)
        vox_pred = vox_pred.cpu().numpy()
        vox_label = img_gt.cpu().numpy()

        metric = self.metric_func(vox_pred, vox_label, num_classes = num_classes_vox, bg = bg)
        self.dice_vox.update(metric[0], batch_size)
        self.prec_vox.update(metric[1], batch_size)
        self.recall_vox.update(metric[2], batch_size)

        dices = [self.dice_vox]
        precs = [self.prec_vox]
        recalls = [self.recall_vox]
        return loss,[dices,precs,recalls]

class VoxEndsTrainer(BaseTrainer):
    def __init__(self,model,criterion,metric_func):
        super(VoxEndsTrainer,self).__init__(model, criterion, metric_func)

        self.dice_vox = AverageMeter("dice_vox")
        self.prec_vox = AverageMeter("prec_vox")
        self.recall_vox = AverageMeter("recall_vox")

        self.dice_ends = AverageMeter("dice_ends")
        self.prec_ends = AverageMeter("prec_ends")
        self.recall_ends = AverageMeter("recall_ends")

    def ends_gt_dilation(self, batch_segs_gt_array, batch_ends_gt_array, thres = 3):
        batch_ends_gt_d = []

        for segs_gt, ends_gt in zip(batch_segs_gt_array, batch_ends_gt_array):
            ends_gt_d = dilation(ends_gt, ball(thres))
            ends_gt_d[segs_gt == 0] = 0

            batch_ends_gt_d.append(ends_gt_d[None])
        batch_ends_gt_d = np.concatenate(batch_ends_gt_d, axis = 0)
        return batch_ends_gt_d

    def _parse_data(self,input_):
        device = self.device 
        img,gts = input_

        img_gt = gts[:,0]
        img_gt_array = img_gt.numpy()
        ends_gt_array = gts[:,1].numpy()
        #ends_gt_array = (gts[:,1].numpy() > 0).astype(int)
        ends_gt_d_array = self.ends_gt_dilation(img_gt_array, ends_gt_array) 
        ends_gt_d = torch.from_numpy(ends_gt_d_array).long()

        img = img.to(device)
        img_gt = img_gt.to(device)
        ends_gt_d = ends_gt_d.to(device)

        return img,[img_gt,ends_gt_d]

    def _forward(self,img,img_gt,ends_gt):
        assert isinstance(self.criterion, VoxEndsLoss)
        batch_size = img.size(0)
        num_classes_vox = 2
        num_classes_ends = 3
        bg = 0

        outputs = self.model(img)
        
        loss = 0.0
        for output in outputs:
            vox_output = output[:,:2]
            ends_output = output[:,2:]
            loss += self.criterion(vox_output, ends_output, img_gt, ends_gt)

        vox_output = outputs[0][:,:2]
        ends_output = outputs[0][:,2:]

        _, vox_pred = torch.max(vox_output,1)
        vox_pred = vox_pred.cpu().numpy()
        vox_label = img_gt.cpu().numpy()
        vox_metric = self.metric_func(vox_pred, vox_label, num_classes_vox, bg = 0)

        _, ends_pred = torch.max(ends_output, 1)
        ends_pred = ends_pred.cpu().numpy()
        ends_label = ends_gt.cpu().numpy()

        ends_pred[vox_pred == 0] = 0
        #ends_label[vox_label == 0] = 0
        ends_metric = self.metric_func(ends_pred, ends_label, num_classes_ends, bg = 0)

        self.dice_vox.update(vox_metric[0], batch_size)
        self.prec_vox.update(vox_metric[1], batch_size)
        self.recall_vox.update(vox_metric[2], batch_size)

        self.dice_ends.update(ends_metric[0], batch_size)
        self.prec_ends.update(ends_metric[1], batch_size)
        self.recall_ends.update(ends_metric[2], batch_size)

        dices = [self.dice_vox, self.dice_ends]
        precs = [self.prec_vox, self.prec_ends]
        recalls = [self.recall_vox, self.recall_ends]

        return loss, [dices, precs, recalls] 

class ScoreEmbSegTrainer(BaseTrainer):
    def __init__(self,model,criterion,metric_func,dsc_loss):
        super(ScoreEmbSegTrainer,self).__init__(model, criterion, metric_func)

        self.dice_ins = AverageMeter("dice_ins")
        self.prec_ins = AverageMeter("prec_ins")
        self.recall_ins = AverageMeter("recall_ins")

        self.dice_vox = AverageMeter("dice_vox")
        self.prec_vox = AverageMeter("prec_vox")
        self.recall_vox = AverageMeter("recall_vox")

        self.l2_loss = nn.MSELoss()
        self.dsc_loss = dsc_loss

    def _parse_data(self,inputs):
        device = self.device
        img, gts = inputs
        
        img = img.to(device)
        gts = gts.to(device)

        prev_img = img[:, :, 0]
        cur_img = img[:, :, 1]

        ins = gts[:, :2]
        segs = gts[:, 2:]

        prev_seg = segs[:, 0]
        cur_seg = segs[:, 1]
        cur_pred_gt = segs[:, 2] 
        
        prev_ins_junc = ins[:,0]
        cur_ins_junc = ins[:,1]
    
        ins_mask = ins_junc_cur > 0
        seg_mask = ins_junc_cur != 0
        junc_mask = ins_junc_cur == -1

        cur_full_seg = torch.zeros_like(ins_junc_cur)
        cur_ins = torch.zeros_like(ins_junc_cur)
        cur_junc = torch.zeros_like(ins_junc_cur)
        
        cur_full_seg[seg_mask] = 1
        cur_ins[ins_mask] = ins_junc_cur[ins_mask]
        cur_junc[junc_mask] = 1

        return cur_img, [prev_img, prev_seg, cur_seg, cur_pred_gt, cur_ins, cur_junc, cur_full_seg]
    
    def _forward(self, cur_img, prev_img, prev_seg, cur_seg, cur_pred_gt, \
            cur_ins, cur_junc, cur_full_seg):

        batch_size = img_cur.size(0)
        bg = 0

        output_ins, output_seg, output_junc, cur_match, prev_match, cur_emb = self.model(\
                cur_img, prev_img, prev_seg, cur_seg)

        loss = self.criterion(output_ins, cur_pred_gt) + \
                self.criterion(output_seg, cur_full_seg) + \
                self.criterion(output_junc, cur_junc) + \
                self.l2_loss(cur_match, cur_pred_gt[:,None].float()) + \
                self.l2_loss(prev_match, cur_pred_gt[:,None].float()) + \
                self.dsc_loss(cur_emb, cur_ins)

        _, ins = torch.max(output_ins, dim = 1)
        ins = ins.cpu().numpy()
        cur_pred_gt = cur_pred_gt.cpu().numpy()
        
        _, seg = torch.max(output_seg, dim = 1)
        seg = seg.cpu().numpy()
        cur_full_seg = cur_full_seg.cpu().numpy()
        
        metric = self.metric_func(ins, pred_cur, num_classes = 2, bg = 0)
        metric_seg = self.metric_func(seg, cur_full_seg, num_classes = 2, bg = 0)

        self.dice_ins.update(metric[0], batch_size)
        self.prec_ins.update(metric[1], batch_size)
        self.recall_ins.update(metric[2], batch_size)

        self.dice_vox.update(metric_seg[0], batch_size)
        self.prec_vox.update(metric_seg[1], batch_size)
        self.recall_vox.update(metric_seg[2], batch_size)

        dices = [self.dice_vox, self.dice_ins]
        precs = [self.prec_vox, self.prec_ins]
        recalls = [self.recall_vox, self.recall_ins]

        return loss,[dices,precs,recalls]

class EmbSegTrainer(BaseTrainer):
    def __init__(self,model,criterion,metric_func,dsc_loss):
        super(EmbSegTrainer,self).__init__(model, criterion, metric_func)

        self.dice_ins = AverageMeter("dice_ins")
        self.prec_ins = AverageMeter("prec_ins")
        self.recall_ins = AverageMeter("recall_ins")

        self.dice_vox = AverageMeter("dice_vox")
        self.prec_vox = AverageMeter("prec_vox")
        self.recall_vox = AverageMeter("recall_vox")

        self.l2_loss = nn.MSELoss()
        self.dsc_loss = dsc_loss

        self.featureHead = nn.DataParallel(model.featureHead)
        self.segHead = nn.DataParallel(model.segHead)
        self.embHead = nn.DataParallel(model.embHead)
        self.matchHead = nn.DataParallel(model.matchHead)
        self.insHead = nn.DataParallel(model.insHead)
        
        
    def ins_sel(self, ins_array):
        """
            ins_array: B * h * w * d
        """
        labels_list = []
        num_labels = []
        for ins_ in ins_array:
            labels_ = np.unique(ins_[ins_ > 0]).tolist()
            labels_list.append(labels_)
            num_labels.append(len(labels_))

        max_num_labels = max(num_labels)
        labels_list_pad = [x * (max_num_labels // len(x)) + x[:max_num_labels % len(x)] \
                                    for x in labels_list]
        return labels_list_pad 
    
    def anchor_sel(self, ins, labels, junc = None, anchor_rate = 0.2, noise = True):
        """
            ins: B * h * w * d
            labels: B
            junc: B * h * w * d or None 
        """
        device = ins.device
        ins_array = ins.cpu().numpy()
        labels_array = labels.cpu().numpy()
        
        anchors_list = []
        for ins_, label_ in zip(ins_array, labels_array):
            #print(np.unique(ins_), label_)
            inds = np.arange(ins_.size)[(ins_ == label_).flatten()]
            anchor_area = int(len(inds) * anchor_rate) + 1

            inds_sel = np.random.choice(inds, anchor_area)
            xs_sel, ys_sel, zs_sel = np.unravel_index(inds_sel, ins_.shape)

            anchor = np.zeros_like(ins_)
            anchor[xs_sel, ys_sel, zs_sel] = 1
            if noise:
                inds_ins_n = np.arange(ins_.size)[(np.logical_and(ins_ != label_, ins_ > 0)).flatten()]
                inds_bg = np.arange(ins_.size)[(ins_ == 0).flatten()]
                if len(inds_ins_n) == 0:
                    inds_sel = np.random.choice(inds_bg, anchor_area // 10 + 1)
                else:
                    inds_sel = np.random.choice(inds_ins_n, anchor_area // 10 + 1)

                xs_sel, ys_sel, zs_sel = np.unravel_index(inds_sel, ins_.shape)
                anchor[xs_sel, ys_sel, zs_sel] = 1

            anchors_list.append(anchor)
        
        anchors_array = np.array(anchors_list)
        anchors = torch.from_numpy(anchors_array).to(device)
        return anchors

    def _parse_data(self,inputs):
        device = self.device
        img, gts = inputs
        
        img = img.to(device)

        prev_img = img[:, :, 0]
        cur_img = img[:, :, 1]

        gts = gts.to(device)
        ins = gts[:, :2]
        junc = gts[:, 2:]

        prev_ins = ins[:, 0]
        cur_ins = ins[:, 1]

        prev_junc = junc[:, 0]
        cur_junc = junc[:, 1]

        cur_ins_array = cur_ins.cpu().numpy()
        labels_list_pad = self.ins_sel(cur_ins_array)
        labels_pad = torch.Tensor(labels_list_pad).to(device).t().long()# max_ins_length * B
        
        #print(labels_pad.size(), labels_pad)

        return cur_img, [prev_img, cur_ins, prev_ins, labels_pad]
    
    def _forward(self, cur_img, prev_img, cur_ins, prev_ins, labels_pad):
        """
            cur_img: B * 1 * h * w * d
            cur_ins: B * h * w * d
            labels_pad: max_ins_length * B
        """
        #print(torch.sum(cur_img))
        batch_size = cur_img.size(0)
        bg = 0
        
        cur_full_seg_gt = (cur_ins > 0).long()

        rrb = torch.cat(self.featureHead(cur_img), dim = 1)

        output_full_seg = self.segHead(rrb)
        _, full_seg = torch.max(output_full_seg, dim = 1)
        full_seg_array = full_seg.cpu().numpy()
        cur_full_seg_gt_array = cur_full_seg_gt.cpu().numpy()
        metric_full_seg = self.metric_func(full_seg_array, cur_full_seg_gt_array)
        
        cur_emb = self.embHead(rrb)

        prev_rrb = torch.cat(self.featureHead(prev_img), dim = 1)
        prev_emb = self.embHead(prev_rrb)

        loss = self.criterion(output_full_seg, cur_full_seg_gt) + \
                    self.dsc_loss(cur_emb, cur_ins)
        loss_seg = 0.0
        num_seg = labels_pad.size(0)
        metrics_seg = []
        for label_pad in labels_pad:
            cur_seg_gt = (cur_ins == label_pad[:, None, None, None]).long()
            prev_seg_gt = (prev_ins == label_pad[:, None, None, None]).long()
            cur_seg_anchor = self.anchor_sel(cur_ins, label_pad) 

            cur_match = self.matchHead(cur_emb, cur_emb, cur_seg_anchor)
            prev_match = self.matchHead(cur_emb, prev_emb, prev_seg_gt)

            fs = torch.cat([cur_match, prev_match, cur_seg_anchor[:, None].float(), cur_emb], dim = 1)
            output_seg = self.insHead(fs)
            
            if torch.sum(prev_seg_gt) > 0:
                loss_seg += self.criterion(output_seg, cur_seg_gt) + \
                        self.l2_loss(cur_match, cur_seg_gt[:, None].float()) + \
                        self.l2_loss(prev_match, cur_seg_gt[:, None].float())
            else:
                loss_seg += self.criterion(output_seg, cur_seg_gt) + \
                        self.l2_loss(cur_match, cur_seg_gt[:, None].float())

            _, seg = torch.max(output_seg, dim = 1)
            seg_array = seg.cpu().numpy()
            cur_seg_gt_array = cur_seg_gt.cpu().numpy()
            
            metric_seg = self.metric_func(seg_array, cur_seg_gt_array)
            metrics_seg.append(metric_seg)

        loss_seg /= num_seg    
        loss += loss_seg
        metric_seg = np.mean(metrics_seg, axis = 0) 

        self.dice_ins.update(metric_seg[0], batch_size)
        self.prec_ins.update(metric_seg[1], batch_size)
        self.recall_ins.update(metric_seg[2], batch_size)

        self.dice_vox.update(metric_full_seg[0], batch_size)
        self.prec_vox.update(metric_full_seg[1], batch_size)
        self.recall_vox.update(metric_full_seg[2], batch_size)

        dices = [self.dice_vox, self.dice_ins]
        precs = [self.prec_vox, self.prec_ins]
        recalls = [self.recall_vox, self.recall_ins]

        return loss,[dices,precs,recalls]


class FFNTrainer(BaseTrainer):
    def __init__(self,model,criterion,metric_func):
        super(FFNTrainer,self).__init__(model, criterion, metric_func)

        self.dice_vox = AverageMeter("dice_vox")
        self.prec_vox = AverageMeter("prec_vox")
        self.recall_vox = AverageMeter("recall_vox")
        self.l2_loss = nn.MSELoss()

    def _parse_data(self,inputs):
        device = self.device
        img, gts = inputs
        
        img = img.to(device)
        gts = gts.to(device)

        img_prev = img[:, :, 0]
        img_cur = img[:, :, 1]

        ins = gts[:, :2]
        segs = gts[:, 2:]

        seg_prev = segs[:, 0]
        seg_cur = segs[:, 1]
        pred_cur = segs[:, 2] 
        
        ins_prev = ins[:,0]
        ins_cur = ins[:,1]
        
        return img_cur, [img_prev, seg_prev, seg_cur, pred_cur]
    
    def _forward(self, img_cur, img_prev, seg_prev, seg_cur, pred_cur):

        batch_size = img_cur.size(0)
        bg = 0

        output, cur_match, prev_match = self.model(img_cur, img_prev, seg_prev, seg_cur)
        loss = self.criterion(output, pred_cur) + \
                self.l2_loss(cur_match, pred_cur[:,None].float()) + \
                self.l2_loss(prev_match, pred_cur[:,None].float())
                

        _, seg = torch.max(output, dim = 1)
        seg = seg.cpu().numpy()
        pred_cur = pred_cur.cpu().numpy()
        
        metric = self.metric_func(seg, pred_cur, num_classes = 2, bg = 0)
        #print(metric)

        self.dice_vox.update(metric[0], batch_size)
        self.prec_vox.update(metric[1], batch_size)
        self.recall_vox.update(metric[2], batch_size)

        dices = [self.dice_vox]
        precs = [self.prec_vox]
        recalls = [self.recall_vox]

        img_cur_array = img_cur.cpu().squeeze().numpy().astype(np.uint16)
        img_prev_array = img_prev.cpu().squeeze().numpy().astype(np.uint16)
        seg_prev_array = seg_prev.cpu().squeeze().numpy().astype(np.uint16)
        pred_cur_array = pred_cur[0].astype(np.uint16)
        seg_cur_array = seg_cur.cpu().squeeze().numpy().astype(np.uint16)
            
        '''if torch.isnan(loss).sum() == 1: 
            sys.exit(0)
        tifffile.imsave("tiffs/img_cur.tif", img_cur_array)
        tifffile.imsave("tiffs/img_prev.tif", img_prev_array)
        tifffile.imsave("tiffs/seg_prev.tif", seg_prev_array)
        tifffile.imsave("tiffs/pred_cur.tif", pred_cur_array)
        tifffile.imsave("tiffs/seg_cur.tif", seg_cur_array)'''
        return loss,[dices,precs,recalls]

'''class DfnTrainer(BaseTrainer):
    def _parse_data(self,inputs):
        device = self.device
        #img,img_gt,fname_ind,patch_center_ind = inputs
        img,img_gt = inputs
        img = img.to(device)
        df_gt_array = batch_direct_field_cal(img_gt)
        df_gt = torch.FloatTensor(df_gt_array)
        df_gt = df_gt.to(device)
        img_gt = img_gt.to(device)

        beta = img_gt.sum().cpu().numpy().astype(np.float)/img_gt.numel() + 1e-5
        weight = torch.tensor([beta,1-beta]).to(self.device) 
        return img,(df_gt,img_gt),weight

    def _forward(self,img,gts):
        outputs = self.model(img)
        _,pred = torch.max(outputs[0][-1],1)
        pred = pred.cpu().numpy()
        loss = self.criterion(outputs[0],gts)
        for output in outputs[1:]:
            loss += self.criterion(output,gts)*0.1
        label = gts[-1].cpu().numpy()
        return loss,pred,label

class DfsTrainer(BaseTrainer):
    def _parse_data(self,inputs):
        device = self.device
        #img,img_gt,fname_ind,patch_center_ind = inputs
        img,img_gt = inputs
        img = img.to(device)
        df_gt_array = batch_direct_field_cal(img_gt)
        df_gt = torch.FloatTensor(df_gt_array)
        df_gt = df_gt.to(device)
        img_gt = img_gt.to(device)

        beta = img_gt.sum().cpu().numpy().astype(np.float)/img_gt.numel() + 1e-5
        weight = torch.tensor([beta,1-beta]).to(self.device) 
        return df_gt,img_gt,weight 

    def _forward(self,df_gt,img_gt):
        outputs = self.model(df_gt)
        _,pred = torch.max(outputs,1)
        pred = pred.cpu().numpy()
        loss = self.criterion(outputs,img_gt)
        label = img_gt.cpu().numpy()
        return loss,pred,label

class RsisTrainer(BaseTrainer):
    def __init__(self,model,criterion,max_seq_len):
        self.super(RsisTrainer).__init__(model,criterion)
        self.fov_shape = fov_shape
        self.deltas = deltas 
        self.max_seq_len = max_seq_len 
    def _parse_data(self,inputs):
        device = self.device
        img,ins_gt,sw_gt = inputs
        img = img.to(device)
        #print(ins_gt.size())

        img_gt = torch.sum(ins_gt,dim = 1)>0
        img_gt = img_gt.float()
        #print(img_gt.size())
        #print(torch.sum(img_gt))
        img_gt = img_gt.to(device)
        ins_gt = ins_gt.to(device)
        #print(sw_gt)
        sw_gt = torch.LongTensor(sw_gt)
        sw_gt = sw_gt.to(device)


        beta = img_gt.sum().cpu().numpy().astype(np.float)/img_gt.numel() + 1e-5
        weight = torch.tensor([beta,1-beta]).to(self.device) 

        return img_gt,(ins_gt,sw_gt),weight

    def _forward(self,img_gt,gts):
        ins_gt,sw_gt = gts
        scores = torch.ones(img_gt.size(0),self.max_seq_len,self.max_seq_len)
        T = self.max_seq_len
        out_masks = []
        out_stops = []
        y_mask = ins_gt
        sw_mask = sw_gt
        batch_size = img_gt.size()[0]
        state = None
        #img_gt = torch.FloatTensor(img_gt).cuda()
        img_gt = img_gt.float()
        input_ = img_gt.unsqueeze(1)
        h,w,d = img_gt.size()[-3:]
        for t in range(0,T):
            out_mask,out_stop,state = self.model(input_,state)
            input_ = torch.argmax(out_mask,dim = 1).unsqueeze(1).float()
            y_pred_i = torch.argmax(out_mask,dim = 1)
            y_pred_i = y_pred_i.unsqueeze(1)
            y_pred_i = y_pred_i.repeat(1,y_mask.size(1),1,1,1)

            y_pred_i = y_pred_i.view(y_mask.size(0)*y_mask.size(1),h,w,d).float().cpu().detach()
            y_true_p = y_mask.view(y_mask.size(0)*y_mask.size(1),h,w,d).float().cpu().detach()
            
            c,_ = batch_soft_metric(y_pred_i.numpy(),y_true_p.numpy(),1)
             
            c = torch.from_numpy(1-c[:,0])
            
            #c = softIoU(y_true_p.float(), y_pred_i.float())
            c = c.view(sw_mask.size(0),-1)
            scores[:,:,t] = c
             
            out_masks.append(out_mask.unsqueeze(1))
            out_stops.append(out_stop)
        t = len(out_masks)

        #B*N*2*h*w*d 
        out_masks = torch.cat(out_masks,1)
        out_stops = torch.cat(out_stops,1)

        out_stops_sigmoid = torch.sigmoid(out_stops)

        out_stops_mask = out_stops_sigmoid > 0.5
        
        sw_mask =sw_mask >0
        
        
        mask = sw_mask + out_stops_mask
        mask = mask >0
        
        sw_mask_mult = mask[:,:,None]*mask[:,None,:]
        sw_mask_mult = sw_mask_mult.cpu().float()
        
        
        scores = torch.mul(scores,sw_mask_mult) + (1-sw_mask_mult)*10
        
        print(scores)
        #print(scores[0])
        y_mask_perm,sw_mask,_ = match(y_mask,out_masks,sw_mask,scores)


        pred_sel = out_masks[mask]
        label_sel = y_mask_perm[mask]
        
        loss = self.criterion(label_sel,sw_mask,pred_sel,out_stops,out_stops_mask)
        #loss = self.criterion(label_sel,sw_mask,pred_sel,out_stops,sw_mask)
        pred_sel = torch.argmax(pred_sel,dim = 1)
        pred_sel = pred_sel.cpu().numpy()
        label_sel = label_sel.cpu().numpy()

        label_sel = label_sel.astype(np.uint8)
        return loss,pred_sel,label_sel



def single_run(ind,embedding,ins_gt_,n_object_,img_gt_):
    ins_mask = cluster(img_gt_,embedding)
    ins_mask_gt = np.zeros_like(img_gt_)
    for i,ins_ in enumerate(ins_gt_[:n_object_]):
        ins_mask_gt[ins_>0] = i + 1
    return ins_mask,ins_mask_gt

def cluster(pred,embeddings,bandwidth = 0.6):
    h,w,d = embeddings.shape[:3]
    embeddings_pred = embeddings[pred>0]
    cluster_mean_shift = MeanShift(bandwidth = bandwidth).fit(embeddings_pred)
        
    ins_mask = np.zeros((h,w,d),dtype = np.uint8)
    masks = np.zeros((h,w,d),dtype = np.uint8)

    fg_coords = np.where(pred>0)
    labels_ = cluster_mean_shift.labels_
    i = 0
    for x,y,z in zip(*fg_coords):
        masks[x,y,z] = labels_[i] + 1
        i += 1

    for label_ in np.unique(labels_):
        mask = masks == (label_ + 1)
        if np.sum(mask) > 50 :
            ins_mask[mask] = label_ + 1
    return ins_mask

def batch_cluster(ins_seg,ins_gt,n_objects,img_gt):
        preds = []
        labels = []
        bs = img_gt.size(0)

        embeddings = ins_seg.unsqueeze(-1).transpose(1,-1).squeeze(1)
        embeddings = embeddings.cpu().detach().numpy()
        ins_gt = ins_gt.cpu().numpy()
        n_objects = n_objects.cpu().numpy()
        img_gt = img_gt.cpu().numpy()
        with Pool(bs) as pool:
            infos = pool.starmap(single_run,zip(range(bs),embeddings,ins_gt,n_objects,img_gt))

        for pred,label in infos:
            preds.append(pred)
            labels.append(label)

        preds = np.stack(preds,axis = 0)
        labels = np.stack(labels,axis = 0)
        return preds,labels

class DLFTrainer(BaseTrainer):
    def __init__(self,model,criterion,is_ins = True,delta_var = 0.5,delta_dist = 1.5):
        super(DLFTrainer,self).__init__(model,criterion)
        self.discriminativeLoss = DiscriminativeLoss(delta_var,delta_dist,2)
        self.is_ins = is_ins
        self.delta_var = delta_var
        self.delta_dist = delta_dist

    def ins_to_sequence(self,ins_gts):
        bs,h,w,d = ins_gts.size()
        inds_n_objects = [torch.unique(x[x>0]) for x in ins_gts]
        n_objects = [len(x) for x in inds_n_objects]
        
        max_n_objects = max(n_objects)
        ins_seq_gts = []

        for ins_gt,ind_objects in zip(ins_gts,inds_n_objects):
            ins_seq_gt = torch.zeros(max_n_objects,h,w,d)
            for i,ind_object in enumerate(ind_objects):
                ins_seq_gt[i][ins_gt == ind_object] = 1
            ins_seq_gts.append(ins_seq_gt.unsqueeze(0))

        ins_seq_gts = torch.cat(ins_seq_gts,dim = 0)
        n_objects = torch.LongTensor(n_objects)
        return ins_seq_gts,n_objects
            
    def _parse_data(self,inputs):
        device = self.device
        img,label = inputs 
        img_gt,ins_gt = label
        
        ins_seq_gts,n_objects = self.ins_to_sequence(ins_gt)

        mask = (img_gt > 0)[:,None]

        img_ = torch.zeros_like(img)
        img_[mask] = img[mask]

        img_ = img_.to(device)
        img_gt = (ins_gt > 0).long().to(device)
        ins_seq_gts = ins_seq_gts.to(device)
        n_objects = n_objects.to(device)

        beta = img_gt.sum().cpu().numpy().astype(np.float)/img_gt.numel() + 1e-5
        weight = torch.tensor([beta,1-beta]).to(self.device) 
        return img_,(img_gt,ins_seq_gts,n_objects),weight

    def _forward(self,img,gts):
        img_gt,ins_gt,n_objects = gts
        
        loss_,sem_seg,ins_seg = self.model(img,ins_gt.float(),n_objects)

        loss = 0
        #print(torch.isnan(loss_))
        loss = loss_.mean()
        
        loss += self.criterion(sem_seg,img_gt)

        if self.cluster_evaluate:
            preds,labels = batch_cluster(ins_seg,ins_gt,n_objects,img_gt)
        else:
            preds = torch.max(sem_seg,dim = 1)[1].cpu().numpy()
            labels = img_gt.cpu().numpy()
        return loss,preds,labels


class DLFDFTrainer(BaseTrainer):
    def __init__(self,model,criterion,is_ins = True,delta_var = 0.5,delta_dist = 1.5):
        super(DLFDFTrainer,self).__init__(model,criterion)
        self.discriminativeLoss = DiscriminativeLoss(delta_var,delta_dist,2)
        self.is_ins = is_ins
        self.delta_var = delta_var
        self.delta_dist = delta_dist

    def ins_to_sequence(self,ins_gts):
        bs,h,w,d = ins_gts.size()
        inds_n_objects = [torch.unique(x[x>0]) for x in ins_gts]
        n_objects = [len(x) for x in inds_n_objects]
        
        max_n_objects = max(n_objects)
        ins_seq_gts = []

        for ins_gt,ind_objects in zip(ins_gts,inds_n_objects):
            ins_seq_gt = torch.zeros(max_n_objects,h,w,d)
            for i,ind_object in enumerate(ind_objects):
                ins_seq_gt[i][ins_gt == ind_object] = 1
            ins_seq_gts.append(ins_seq_gt.unsqueeze(0))

        ins_seq_gts = torch.cat(ins_seq_gts,dim = 0)
        n_objects = torch.LongTensor(n_objects)
        return ins_seq_gts,n_objects
            

    def _parse_data(self,inputs):
        device = self.device
        img,label = inputs 
        img_gt,ins_gt = label

        ins_seq_gts,n_objects = self.ins_to_sequence(ins_gt)
        #img = img.to(device)
        

        img_gt = img_gt.long().to(device)
        #img_gt = (ins_gt>0).long().to(device)

        df_gt_array = batch_direct_field_cal(img_gt)
        df_gt = torch.FloatTensor(df_gt_array)
        df_gt = df_gt.to(device)

        ins_seq_gts = ins_seq_gts.to(device)
        n_objects = n_objects.to(device)
        img = torch.cat([img_gt[:,None].float(),df_gt],dim = 1)
        #return img_gt[:,None].float(),(img_gt,ins_seq_gts,n_objects)

        beta = img_gt.sum().cpu().numpy().astype(np.float)/img_gt.numel() + 1e-5
        weight = torch.tensor([beta,1-beta]).to(self.device) 
        return img,(img_gt,ins_seq_gts,n_objects),weight

    def _forward(self,img,gts):
        img_gt,ins_gt,n_objects = gts
        
        max_n_objects = torch.max(n_objects)
        
        #loss_,sem_seg,ins_seg = self.model(img,ins_gt.float(),n_objects,max_n_objects)
        #print(img.size())
        loss_,sem_seg,ins_seg = self.model(img,ins_gt.float(),n_objects)
        #sem_seg,ins_seg = self.model(img)

        loss = 0
        loss = loss_.mean()
        loss += self.criterion()
        if self.cluster_evaluate:
            preds,labels = batch_cluster(ins_seg,ins_gt,n_objects,img_gt)
        else:
            preds = torch.max(img_gt,dim = 1)[1].cpu().numpy()
            labels = img_gt.cpu().numpy()
        return loss,preds,labels'''


