import torch
from torch.autograd import Variable
import torch.nn as nn
from munkres import Munkres
import numpy as np
import time

torch.manual_seed(0)


def StableBalancedMaskedBCE(target, out, balance_weight = None):
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    if balance_weight is None:
        num_positive = target.sum()
        num_negative = (1 - target).sum()
        total = num_positive + num_negative
        balance_weight = num_positive / total

    max_val = (-out).clamp(min=0)
    # bce with logits
    loss_values =  out - out * target + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()
    loss_positive = loss_values*target
    loss_negative = loss_values*(1-target)
    losses = (1-balance_weight)*loss_positive + balance_weight*loss_negative

    return losses


def softIoU(target, out, e=1e-6):
    num = torch.sum(out*target,dim = -1)
    den = torch.sum(out+target-out*target,dim = -1) + e
    iou = num / den
    cost = (1 - iou)
    return cost
def softIoU(target, out, e=1e-6):
    num = np.sum(out*target,axis = -1)
    den = np.sum(out+target-out*target,axis = -1) + e
    iou = num / den
    cost = (1 - iou)

def match(t_mask,p_mask,sw_mask,overlaps):
    overlaps = (overlaps.data).cpu().numpy().tolist()
    m = Munkres()

    #t_mask, p_mask = masks
    # get true mask values to cpu as well
    t_mask_cpu = (t_mask.data).cpu().numpy()
    # init matrix of permutations
    permute_indices = np.zeros((t_mask.size(0),t_mask.size(1)),dtype=int)
    # we will loop over all samples in batch (must apply munkres independently)
    for sample in range(p_mask.size(0)):
        # get the indexes of minimum cost
        indexes = m.compute(overlaps[sample])
        for row, column in indexes:
            # put them in the permutation matrix
            permute_indices[sample,column] = row

        # sort ground according to permutation
        t_mask[sample] = t_mask[sample,permute_indices[sample],:]
        sw_mask[sample] = sw_mask[sample,permute_indices[sample]]
    return t_mask,sw_mask,permute_indices

