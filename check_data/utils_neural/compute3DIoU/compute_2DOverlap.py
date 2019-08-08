import numpy as np

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (np.array) center-size default boxes from priorbox layers.
    Return:
        boxes: (np.array) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return np.hstack([boxes[:, :2] - boxes[:, 2:]/2,
                      boxes[:, :2] + boxes[:, 2:]/2])

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (np.array) point_form boxes
    Return:
        boxes: (np.array) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return np.hstack([(boxes[:, 2:] + boxes[:, :2])/2,
                      (boxes[:, 2:] - boxes[:, :2])])

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (np.array) bounding boxes, Shape: [A,4].
      box_b: (np.array) bounding boxes, Shape: [B,4].
    Return:
      (np.array) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xyA = np.broadcast_to( np.expand_dims(box_a[:, 2:], axis=1), shape=(A, B, 2) )
    max_xyB = np.broadcast_to( np.expand_dims(box_b[:, 2:], axis=0), shape=(A, B, 2) )
    max_xy = np.min([max_xyA, max_xyB], axis=0)

    min_xyA = np.broadcast_to( np.expand_dims(box_a[:, :2], axis=1), shape=(A, B, 2) )
    min_xyB = np.broadcast_to( np.expand_dims(box_b[:, :2], axis=0), shape=(A, B, 2) )
    min_xy = np.max([min_xyA, min_xyB], axis=0)

    inter = np.max([np.zeros_like(max_xy), max_xy - min_xy], axis=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (np.array) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (np.array) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (np.array) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)  # a.shpae[0] X b.shape[0] [A,B]
    area_a = np.expand_dims( ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])), axis=1 )
    area_a = np.broadcast_to(area_a, shape=inter.shape) 

    area_b = np.expand_dims( ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])), axis=0 )
    area_b = np.broadcast_to(area_b, shape=inter.shape) 

    union = area_a + area_b - inter
    return inter / union

def funtest():
    # a = np.random.randint(1, 5, size=(2, 4))
    # a1 = np.random.randint(1, 5, size=(3, 4))

    # a = np.array([[0, 0, 4, 4],
    #               [0, 0, 4, 4]])
    # a1 = np.array([[2, 2, 4, 4],
    #                [3, 3, 4, 4]])

    a = np.array([0, 0, 4, 4])
    a1 = np.array([2, 2, 4, 4])

    b = point_form(a)
    b1 = point_form(a1)
    print(a), print(a1) 

    return jaccard(b, b1)

if __name__ == "__main__":
    iou = funtest()