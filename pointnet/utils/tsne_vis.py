import os
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import colorsys
import random
import torch
from tqdm import tqdm

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_PATH, "../"))

# from lib.net.neural_ins import NeuralPointNet
from lib.net.neural_ins_syn import NeuralPointNet
from lib.datasets.point_dataset import PointDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        # r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([_r, _g, _b])

    return np.array(rgb_colors)


def infer():
    # DATA_PATH = "/home/fcheng/Neuron/pointnet/data/real_data_FPS_15000/"
    DATA_PATH = "/home/fcheng/Neuron/pointnet/data/syn_data_FPS_32768/"
    data_set = PointDataset(data_root=DATA_PATH, mode="train", augment=True)
    
    MODEL_PATH = "logs/log_syn_noaug/ckpt/checkpoint_epoch_775.pth"
    model = NeuralPointNet(input_channels=0)
    model.cuda()

    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return data_set, model

def scatter(x, labels):
    uni_labels = np.unique(labels)
    colors = ncolors(len(uni_labels))

    # create a scatter plot
    f = plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=20, c=colors[labels])

    # add labels for each class
    txts = []
    for i in range(len(uni_labels)):
        # position of each label
        xtext, ytext = np.median(x[labels==i, :], axis=0)
        txt =  ax.text(xtext, ytext, str(uni_labels[i]), fontsize=10)
        txt.set_path_effects([PathEffects.Stroke(linewidth=2, foreground="w"),
                              PathEffects.Normal()])
        txts.append(txt)
    
    return f, ax, sc, txts

def tsne_vis(idx):
    out_path = BASE_PATH
    train_set, model = infer()
    idx = 0
    point_sets, ins = train_set[idx]
    name = train_set.dataNames[idx]
    point_sets = point_sets[None]

    point_sets = torch.from_numpy(point_sets).cuda().float()
    features = model(point_sets)
    print("Inference Finished!")

    features = features.permute(0, 2, 1)[0]  # [9600, feat_dims]
    features = features.cpu().detach().numpy()
    # labels = None  # [9600]

    ordered_ins = np.zeros(len(ins), dtype=int)
    for cnt, l in enumerate(np.unique(ins)):
        ordered_ins[ins==l] = cnt

    proj = TSNE(random_state=100).fit_transform(features)
    print("Projection Finished!")

    f, _, _, _ = scatter(proj, ordered_ins)
    f.savefig(os.path.join(out_path, name+".png"))
    plt.close(f)

def func():
    out_path = os.path.join(BASE_PATH, "tsne_vis/syn/32768/train")
    os.makedirs(out_path, exist_ok=True)

    train_set, model = infer()

    for it, idx in enumerate(range(len(train_set))):
        print("{}/{}".format(it, len(train_set)))
        point_sets, ins = train_set[idx]
        name = train_set.dataNames[idx]
        point_sets = point_sets[None]

        point_sets = torch.from_numpy(point_sets).cuda().float()
        features = model(point_sets)
        print("Inference Finished!")

        features = features.permute(0, 2, 1)[0]  # [9600, feat_dims]
        features = features.cpu().detach().numpy()
        # labels = None  # [9600]

        ordered_ins = np.zeros(len(ins), dtype=int)
        for cnt, l in enumerate(np.unique(ins)):
            ordered_ins[ins==l] = cnt

        proj = TSNE(random_state=100).fit_transform(features)
        print("Projection Finished!")

        f, _, _, _ = scatter(proj, ordered_ins)
        f.savefig(os.path.join(out_path, name+".png"))
        plt.close(f)
        print(name, " have been processed\n")

if __name__ == "__main__":
    func()
    # tsne_vis(0)