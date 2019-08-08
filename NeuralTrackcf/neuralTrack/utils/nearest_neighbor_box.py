import numpy as np
from sklearn.cluster import MeanShift
from skimage.measure import label
from skimage.external import tifffile

def nearest_neighbor_box(mask):
    mask_size = mask.shape
    inds = np.arange(mask.size)
    boxes = []
    labels_ = label(mask)

    for label_ in np.unique(labels_)[1:]:
        mask_ = (labels_ == label_)
        if np.sum(mask_) < 30:continue
        inds_sel = inds[mask_.flatten()]
        coords_ = np.array(np.unravel_index(inds_sel,mask_size)).transpose()

        coord_lt = np.min(coords_, axis = 0).tolist()
        coord_rb = np.max(coords_, axis = 0).tolist()
        boxes.append(coord_lt + coord_rb)
    boxes = np.array(boxes)

    '''if len(boxes) > 0:
        boxes_range = boxes[:,-3:] - boxes[:,:-3]
    else:
        boxes_range = np.array([0,0,0])'''

    return boxes

def center_fixed_box(mask,thres = 8):
    mask_size = mask.shape
    inds = np.arange(mask.size)
    boxes = []
    labels_ = label(mask)

    for label_ in np.unique(labels_)[1:]:
        mask_ = (labels_ == label_)
        if np.sum(mask_) < 30:continue
        inds_sel = inds[mask_.flatten()]
        coords_ = np.array(np.unravel_index(inds_sel,mask_size)).transpose()

        coord_c = np.mean(coords_,axis = 0)

        #coord_lt = np.min(coords_, axis = 0).tolist()
        #coord_rb = np.max(coords_, axis = 0).tolist()
        coord_lt = np.clip(coord_c - thres,0,mask.shape).tolist()
        coord_rb = np.clip(coords_ + thres,0,mask.shape).tolist()
        boxes.append(coord_lt + coord_rb)

    return boxes

if __name__ == "__main__":
    mask_p = "/home/jjx/Biology/DirectField/NeuralTrack/preds_patch/6950_34600_4150/6950_34600_4150_26_pred.tif"
    mask = tifffile.imread(mask_p)
    mask = (mask > 0).astype(int)
    #print(np.sum(mask))
    boxes = nearest_neighbor_box(mask)
    for box in boxes:
        print(box)



