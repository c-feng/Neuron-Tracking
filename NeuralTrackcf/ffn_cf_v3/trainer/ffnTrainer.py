
from baseTrainer import BaseTrainer
from ..utils.meters import AverageMeter

class SegFFNTrainer(BaseTrainer):
    def __init__(self, model, fov_shape, criterion, metric_func=None, logs_path=None, device="cuda"):
        super(SegFFNTrainer, self).__init__(model, criterion, metric_func, logs_path, device)
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

    def _forward(self, imgs, gts):
        