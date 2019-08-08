import numpy as np

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

def funtest():
    # a = np.random.randint(1, 5, size=(2, 4))
    # a1 = np.random.randint(1, 5, size=(3, 4))

    a = np.array([[0, 0, 0, 4, 4, 4],
                  [0, 0, 0, 4, 4, 4]])
    a1 = np.array([[2, 2, 2, 4, 4, 4],
                   [8, 8, 8, 4, 4, 4]])

    # a = np.array([[0, 0, 0, 4, 4, 4]])
    # a1 = np.array([[2, 2, 2, 4, 4, 4]])

    b = point_form(a)
    b1 = point_form(a1)
    print(a), print(a1) 
    print(b), print(b1) 
    print(center_size(b)), print(center_size(b1))

    return jaccard(b, b1)

if __name__ == "__main__":
    iou = funtest()