import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, ignore_target=-1):
        super().__init__()
        self.ignore_target = ignore_target

    def forward(self, input, target):
        """
        :param input: (N), logit
        :param target: (N), {0, 1}
        :return:
        """
        input = torch.sigmoid(input.view(-1))
        target = target.float().view(-1)
        mask = (target != self.ignore_target).float()
        return 1.0 - (torch.min(input, target) * mask).sum() / torch.clamp((torch.max(input, target) * mask).sum(), min=1.0)

class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focusses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self,
                prediction_tensor,
                target_tensor,
                weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return focal_cross_entropy_loss * weights

def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    # transpose_param = [0] + [param[-1]] + param[1:-1]
    # logits = logits.permute(*transpose_param)
    # loss_ftor = nn.NLLLoss(reduce=False)
    # loss = loss_ftor(F.logsigmoid(logits), labels)
    return loss

def calculate_means(pred, gt, c_labels):
    """ 
        pred: npoint x features
        gt: npoint

        Return: n_cluster x features
    """
    cluster_means = []
    for cl in c_labels:
        # n_points = torch.sum(gt==cl)
        mean = torch.mean(pred[gt==cl], dim=0)
        cluster_means.append(mean)
    
    return torch.stack(cluster_means, dim=0)

def calculate_variance_term(pred, gt, c_labels, n_means, delta_v):
    """ cal L_var
        an intra-cluster pull-force that draws embeddings towards the mean embedding,
        i.e. the cluster center.
    """
    l_var = torch.tensor([0.]).cuda()
    for i, cl in enumerate(c_labels):
        mean = n_means[i]
        lv = torch.mean( torch.pow(torch.max( torch.norm(pred[gt==cl] - mean, dim=1) - delta_v, other=torch.tensor(0., device="cuda") ), 2) )
        l_var = l_var + lv
    l_var = l_var / len(c_labels)
    return l_var

def calculate_distance_term(n_means, delta_d):
    """ cal L_dist
        an inter-cluster push-force that pushes clusters away from each other, increasing the distance
        between the cluster centers.
    """
    l_dist = torch.tensor([0.]).cuda()
    mask = torch.arange(n_means.size(0))
    for i, cA in enumerate(n_means):
        ld = torch.mean( torch.pow( torch.max(2*delta_d - torch.norm(n_means[mask!=i] - cA, dim=1), other=torch.tensor(0., device="cuda")), 2) )
        l_dist = l_dist + ld
    l_dist = l_dist / n_means.size(0)
    return l_dist

def calculate_regularization_term(n_means):
    """ cal L_reg
        a small pull-force that draws all clusters towards the origin, to keep the activations
        bounded.
    """
    l_reg = torch.mean( torch.norm(n_means, dim=1) )
    return l_reg

def discriminative_loss_a_batch(prediction, correct_label,
                                delta_v, delta_d,
                                param_var, param_dist, param_reg):
    """ calculate discriminative loss in a Batch 
    """
    cluster_labels = torch.unique(correct_label, sorted=True)
    n_means = calculate_means(prediction, correct_label, cluster_labels)
    l_var = calculate_variance_term(prediction, correct_label, cluster_labels, n_means, delta_v)
    l_dist = calculate_distance_term(n_means, delta_d)
    l_reg = calculate_regularization_term(n_means)

    loss = param_var * l_var + param_dist * l_dist + param_reg * l_reg
    return loss, l_var, l_dist, l_reg

def discriminative_loss(prediction, correct_label,
                        delta_v=0.5, delta_d=5.,
                        param_var=1., param_dist=2., param_reg=0.001):
    ''' Iterate over a batch of prediction/label and cumulate loss
    :return: discriminative loss and its three components
    '''
    B = prediction.size(0)
    
    Loss = torch.tensor([0.]).cuda()
    L_var = torch.tensor([0.]).cuda()
    L_dist = torch.tensor([0.]).cuda()
    L_reg = torch.tensor([0.]).cuda()
    for b in range(B):
        l, lv, ld, lr = discriminative_loss_a_batch(prediction[b], correct_label[b],
                                                       delta_v, delta_d,
                                                       param_var, param_dist, param_reg)
        Loss = Loss + l
        L_var = L_var + lv
        L_dist = L_dist + ld
        L_reg = L_reg + lr
    Loss, L_var, L_dist, L_reg = Loss / B, L_var / B, L_dist / B, L_reg / B
    return Loss, L_var, L_dist, L_reg

if __name__ == "__main__":
    import os
    import pdb
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    pred = torch.randint(0, 4, (1, 10, 64)).float().cuda()
    # gt = torch.randint(1, 5, [4, 15000]).float()
    gt = torch.arange(10).float()[None].cuda()


    loss, v, d, r = discriminative_loss(pred, gt, delta_v=0.5, delta_d = 1.5,
                                        param_var=1., param_dist=1., param_reg=0.001)
    print("disc loss: ", loss)
    print("loss variance: ", v)
    print("loss distance: ", d)
    print("loss regularization: ", r)

