import numpy as np 

from scipy.spatial.distance import cdist 
from skimage.morphology import ball,dilation, skeletonize_3d
from .nearest_neighbor_box import nearest_neighbor_box,center_fixed_box

def batch_soft_metric(preds,labels,thres = 1):
    assert len(preds.shape) == len(labels.shape) == 4
    metric = np.zeros((preds.shape[0],4))
    assert preds.shape == labels.shape
    i = 0
    for pred,label in zip(preds,labels):
        metric[i] = soft_metric(pred,label,thres)
        i += 1
    return np.mean(metric,axis = 0)

def batch_multi_soft_metric(preds, labels, num_classes = 2, bg = 0, thres = 1):
    #print(preds.shape, labels.shape)
    assert len(preds.shape) == len(labels.shape) == 4
    metrics = []
    for i in range(num_classes):
        if i == bg:continue
        preds_ = (preds == i).astype(int)
        labels_ = (labels == i).astype(int)
        metric = batch_soft_metric(preds_, labels_, thres)

        metrics.append(metric)
    metrics = np.array(metrics)
    return np.mean(metrics, axis = 0)

def multi_soft_metric(preds, labels, num_classes = 2, bg = 0, thres = 1):
    metrics = []
    for i in range(num_classes):
        if i == bg:continue
        preds_ = (preds == i).astype(int)
        labels_ = (labels == i).astype(int)
        metric = soft_metric(preds_, labels_, thres)
        #print(metric)
        metrics.append(metric)
    metrics = np.array(metrics)
    #metrics = np.concatenate(metric, axis = 0)
    #print(metrics.shape)
    return np.mean(metrics, axis = 0)
    

def soft_metric(pred,label,thres = 1):
    label_expand = dilation(label,ball(thres))
    pred_expand = dilation(pred,ball(thres))
    
    if np.sum(pred) == 0 :
        if np.sum(label) == 0:
            return np.array([1,1,1,0])
        else:
            return np.array([0,1,0,0])
    else:
        if np.sum(label) == 0:
            return np.array([0,0,1,0])

    prec = np.sum(pred*label_expand)/(np.sum(pred)+ 1e-10)
    recall = np.sum(label*pred_expand)/(np.sum(label)+ 1e-10)
    
    rate = np.sum(label)/label.size
    if prec > 0 and recall > 0:
        dice = 2*prec*recall/(prec + recall)
    else:
        dice = 0
    return np.array([dice,prec,recall,rate])

def calc_dic(n_objects_gt, n_objects_pred):
    return np.abs(n_objects_gt - n_objects_pred)


def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice

    

def calc_bd(ins_seg_pred, ins_seg_gt, thres = 1):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))

    best_metrics = []
    best_matchs = []
    if np.sum(ins_seg_pred > 0) == 0:
        if np.sum(ins_seg_gt > 0) == 0:
            return np.array([[1, 1, 1, 0]]), np.array([[0,0]])
        else:
            return np.array([[0, 1, 0, 0]]), np.array([[0,0]])

    
    for pred_idx in pred_object_idxes:
        _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

        metrics = []
        if np.sum(ins_seg_gt > 0) == 0:
            best_metric =  np.array([0, 0, 1, 0])
            best_match = np.array([pred_idx, 0])
        else:
            for gt_idx in gt_object_idxes:
                _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
            
                metric = soft_metric(_pred_seg, _gt_seg, thres)
                metrics.append(metric) 
            metrics = np.array(metrics)
            best_idx = np.argmax(metrics[:, 0])
            best_match = [pred_idx , gt_object_idxes[best_idx]]
            best_metric = metrics[best_idx]

        best_metrics.append(best_metric)
        best_matchs.append(best_match)

    return np.array(best_metrics), np.array(best_matchs)

def calc_sbd(ins_seg_gt, ins_seg_pred, thres = 1):

    _metrics1,_ = calc_bd(ins_seg_gt, ins_seg_pred, thres)

    _metrics2,_ = calc_bd(ins_seg_pred, ins_seg_gt, thres)
    
    _metric1 = np.mean(_metrics1, axis = 0)
    _metric2 = np.mean(_metrics2, axis = 0)

    return np.mean([_metric1, _metric2], axis = 0)

def batch_cal_sbd(ins_seg_preds,ins_seg_gts, thres = 1):
    metric = []
    for ins_seg_gt,ins_seg_pred in zip(ins_seg_gts,ins_seg_preds):
        metric.append(calc_sbd(ins_seg_gt,ins_seg_pred))
    return np.mean(metric, axis = 0)

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, zmin, xmax, ymax, zmax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (np.array) center-size default boxes from priorbox layers.
    Return:
        boxes: (np.array) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return np.hstack([boxes[:, :3] - boxes[:, 3:]/2,
                      boxes[:, :3] + boxes[:, 3:]/2])

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, cz, w, h, l)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (np.array) point_form boxes
    Return:
        boxes: (np.array) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return np.hstack([(boxes[:, 3:] + boxes[:, :3])/2,
                      (boxes[:, 3:] - boxes[:, :3])])

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,3] -> [A,1,3] -> [A,B,3]
    [B,3] -> [1,B,3] -> [A,B,3]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (np.array) bounding boxes, Shape: [A,6].
      box_b: (np.array) bounding boxes, Shape: [B,6].
    Return:
      (np.array) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xyA = np.broadcast_to( np.expand_dims(box_a[:, 3:], axis=1), shape=(A, B, 3) )
    max_xyB = np.broadcast_to( np.expand_dims(box_b[:, 3:], axis=0), shape=(A, B, 3) )
    max_xy = np.min([max_xyA, max_xyB], axis=0)  # 大大取小

    min_xyA = np.broadcast_to( np.expand_dims(box_a[:, :3], axis=1), shape=(A, B, 3) )
    min_xyB = np.broadcast_to( np.expand_dims(box_b[:, :3], axis=0), shape=(A, B, 3) )
    min_xy = np.max([min_xyA, min_xyB], axis=0)  # 小小取大

    inter = np.max([np.zeros_like(max_xy), max_xy - min_xy], axis=0)
    return inter[:, :, 0] * inter[:, :, 1] * inter[:, :, 2]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (np.array) Ground truth bounding boxes, Shape: [num_objects,6]
        box_b: (np.array) Prior boxes from priorbox layers, Shape: [num_priors,6]
    Return:
        jaccard overlap: (np.array) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)  # a.shpae[0] X b.shape[0] [A,B]
    vox_a = np.expand_dims( ((box_a[:, 3] - box_a[:, 0]) *
                             (box_a[:, 4] - box_a[:, 1]) *
                             (box_a[:, 5] - box_a[:, 2])), axis=1 )
    vox_a = np.broadcast_to(vox_a, shape=inter.shape) 

    vox_b = np.expand_dims( ((box_b[:, 3] - box_b[:, 0]) *
                             (box_b[:, 4] - box_b[:, 1]) *
                             (box_b[:, 5] - box_b[:, 2])), axis=0 )
    vox_b = np.broadcast_to(vox_b, shape=inter.shape) 

    union = vox_a + vox_b - inter
    return inter / union

def iou_metric(pred,label,thres = 0.5,box_method = nearest_neighbor_box):
    pred_boxes = box_method(pred)
    label_boxes = box_method(label)
    if len(pred_boxes) ==0 :
        if len(label_boxes) == 0:
            return np.array([1,1,1,0])
        elif len(label_boxes) != 0:
            return np.array([0,1,0,0])
    else:
        if len(label_boxes) ==0:
            return np.array([0,0,1,0])

    #print(pred_boxes.shape)
    iou = jaccard(pred_boxes,label_boxes)
    iou_mask = iou > thres
    #print(iou[iou > 0])
    prec = np.sum(np.sum(iou_mask,axis = 1) > 0 ) / iou_mask.shape[0]
    recall = np.sum(np.sum(iou_mask,axis = 0) > 0) / iou_mask.shape[1]
    dice = 2/(1/prec+1/recall)
    return np.array([dice,prec,recall,0])


def batch_iou_metric(preds,labels,thres = 0.2):
    assert len(preds.shape) == len(labels.shape) == 4
    metric = np.zeros((preds.shape[0],4))
    assert preds.shape == labels.shape
    i = 0
    for pred,label in zip(preds,labels):
        metric[i] = iou_metric(pred,label,thres)
        i += 1
    return np.mean(metric,axis = 0)
if __name__ == "__main__":
    a = np.random.rand(3,3,3)
    b = np.random.rand(3,3,3)
    metric = iou_metric(a,b)
    print(metric)

