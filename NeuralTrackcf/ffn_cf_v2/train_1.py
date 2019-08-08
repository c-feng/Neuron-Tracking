import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn

import os
import numpy as np
from queue import Queue
import argparse
from scipy.special import expit, logit
from tensorboardX import SummaryWriter

# from model import FloodFillingNetwork as ffn
from models.voxResnet import VoxResnet as ffn
from utils import prepare_data, patch_subvol, mask_subvol, get_data, set_data, get_new_locs, get_weights, trans3Dto2D
from data import *
from modules import FFNLoss, sigmoidCrossEntropyLoss as sigmoidCELoss, FocalLoss, BinaryFocalLoss

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=9e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/train_1/lr1e-3',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    cfg = neural_train
    dataset =  NeuralData(list_file=cfg['list_file'], data_root=cfg['data_root'])

    fw, fh, fd, fc = cfg['fov_shape']

    # model = ffn(in_planes=2, module_nums=8)
    model = ffn()

    print("Initializing weights...")
    model.init_weights()
    
    if args.cuda:
        model = torch.nn.DataParallel(model)  # 多卡存在问题, priors加倍了
        cudnn.benchmark = True

    if args.cuda:
        model = model.cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = BinaryFocalLoss(gamma=2)

    print('Using the specified args:')
    print(args)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    writer = SummaryWriter('./logs/train_1/')

    epoch = 0
    step_index = 0
    batch_iterator = None
    epoch_size = len(dataset) // args.batch_size

    # Loss counters
    vis_loss = []

    i = 0
    for iteration in range(cfg['max_iter']):
        # load train data .tif voxel
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(data_loader)
            epoch += 1
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        
        images_gt, targets, images_ = next(batch_iterator)
        images = images_
        vis_loss = []

        # if args.cuda:
        #     images = images.to('cuda')
        #     targets = targets.to('cuda')
        # else:
        #     images = torch.Tensor(images)
        #     targets = torch.Tensor(targets)
        locations = prepare_data(labels=targets, patch_shape=cfg['subvol_shape'])

        # 对patch的训练
        indices = np.random.permutation(len(locations))
        for location in locations[indices]:
            # 抽取patch, 同时生成与中心点相对应的监督标签
            patch_img_gt, _, _ = patch_subvol(data=images_gt, labels=targets, subvol_shape=cfg['subvol_shape'],
                                              deltas=np.array(cfg['fov_shape'][:3])//2, location=location)
            subvol_data, subvol_labels, relative_loc = patch_subvol(data=images, labels=targets, subvol_shape=cfg['subvol_shape'],
                                                                    deltas=np.array(cfg['fov_shape'][:3])//2, location=location)
            # 与patch相对应的soft二值mask
            subvol_mask = mask_subvol(subvol_data.shape, relative_loc)
            n, c, w, h, d = subvol_data.shape

            # Create FOV dicts, and center locations
            V = {(relative_loc[0], relative_loc[1], relative_loc[2])}  # set()
            queue = Queue()
            queue.put([relative_loc[0], relative_loc[1], relative_loc[2]])
            # Compute upper and lower bounds
            upper = [w - fw // 2, h - fh // 2, d - fd // 2]
            lower = [fw // 2, fh // 2, fd // 2]

            p_weights = []
            optimizer.zero_grad()
            cnt = 0
            while not queue.empty():
                if cnt > 10:
                    break
                cnt += 1
                # Get new list of FOV locations
                current_loc = np.array(queue.get(), np.int32)
                # Center around FOV
                fov_gt = get_data(patch_img_gt, current_loc, cfg['fov_shape'])
                fov_data = get_data(subvol_data, current_loc, cfg['fov_shape'])
                fov_labels = get_data(subvol_labels, current_loc, cfg['fov_shape'])
                # fov_labels = np.squeeze(fov_labels, axis=1)
                fov_mask = get_data(subvol_mask, current_loc, cfg['fov_shape'])
                
                # Loss-weighted
                weights = get_weights(fov_labels)
                p_weights.append(weights)
                # print("weights:", weights)
                # criterion = nn.BCEWithLogitsLoss(pos_weight=0.005*weights)

                # Add merging of old and new mask
                d_m = np.concatenate([fov_data, fov_mask], axis=1)
                if args.cuda:
                    d_m = torch.Tensor(d_m).to('cuda')
                    fov_labels = torch.Tensor(fov_labels).to('cuda')
                else:
                    d_m = torch.Tensor(d_m)
                    fov_labels = torch.Tensor(fov_labels)
               
                pred = model(d_m)
                # print(type(pred), pred.type())
                # print(torch.from_numpy(fov_mask).type())
                logit_seed = torch.add(torch.from_numpy(fov_mask).to('cuda'), other=pred)
                # logit_seed = pred
                prob_seed = expit(logit_seed.detach().cpu().numpy())
                if len(vis_loss) % 10 == 0:
                    # print(np.max(prob_seed), np.min(prob_seed), np.sum(prob_seed>0.95)/(17*17*17))
                    writer.add_scalars("prob_map", {"max": np.max(prob_seed),
                                                   "min": np.min(prob_seed),
                                                   "pos_ratio": np.sum(prob_seed>0.95)/(33*33*33),
                                                   "1/weights": 1/weights}, i)

                # Loss, Backprop
                optimizer.zero_grad()
                # print(torch.max(pred), torch.min(pred))
                # print(torch.max(torch.sigmoid(pred)), torch.min(torch.sigmoid(pred)))
                loss0 = criterion(logit_seed, fov_labels, weights)
                loss0.backward(retain_graph=True)
                # gradClamp(model.parameters())
                optimizer.step()

                # log
                if i % 10 == 0:
                    writer.add_scalars("Train/Loss", {"loss": loss0.data}, i)
                    for name, layer in model.named_parameters():
                        writer.add_histogram(name+'_grad', layer.grad.cpu().data.numpy(), i)
                    writer.add_image("Target", trans3Dto2D(fov_labels.cpu()), i)
                    writer.add_image("ProbMap", trans3Dto2D(prob_seed), i)
                    writer.add_image("gt", trans3Dto2D(fov_gt), i)

                i += 1
                vis_loss.append(loss0.detach().item())
                if len(vis_loss) % 10 == 0:
                    print("%d of a tif, FOV Loss: %.6f" % (len(vis_loss), loss0.data.item()))

                # 更新patch对应的soft二值mask
                set_data(subvol_mask, current_loc, logit_seed.detach().cpu().numpy())

                # Compute new locations
                new_locations = get_new_locs(logit_seed.detach().cpu().numpy(), cfg['delta'], cfg['tmove'])
                for new in new_locations:
                    new = np.array(new, np.int32) + current_loc
                    bounds = [lower[j] <= new[j] < upper[j] for j in range(3)]
                    stored_loc = tuple([new[i] for i in range(3)])
                    if all(bounds) and stored_loc not in V:
                        V.add(stored_loc)
                        queue.put(new)
            # mask = subvol_mask >= logit(0.6)
            loss1 = len(p_weights) * criterion(torch.Tensor(subvol_mask), torch.Tensor(subvol_labels), np.mean(p_weights))
            loss0.data.zero_()
            loss0.data = loss1.data
            loss0.backward()
            optimizer.step()
            print("One patch ends of Iteration(%d)/Epoch(%d)" % (iteration, epoch))
        print("One tif ends of Iteration(%d)/Epoch(%d)" % (iteration, epoch))

        if iteration % 10 == 0:
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss0.data.item()), end='\n')
            # print('timer: %.4f sec.    %.4f sec.' % (t2 - t1, t1 - t0))
        # if args.visdom:
        #     update_vis_plot(iteration, min(500, np.mean(vis_loss)), iter_plot, epoch_plot, 'append')
        if iteration != 0 and iteration % 20 == 0:
            print('Saving state, iter:', iteration)
            torch.save(model.state_dict(), args.save_folder +'/FFN_' + dataset.name + "_" +
                       repr(iteration) + '.pth')
    torch.save(model.state_dict(),
               args.save_folder + '/FFN_' + dataset.name + '.pth')


def gradClamp(parameters, clip=1.):
    for p in parameters:
        p.grad.data.clamp_(min=-clip, max=clip)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
