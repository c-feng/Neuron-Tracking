import os
import glob
import numpy as np
from tqdm import tqdm
from skimage.external import tifffile
from skimage import io, color
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
plt.switch_backend('agg')
import pdb


def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def orderedLabel(tif):
    labels = np.unique(tif)[1:]
    tif_ordered = np.zeros_like(tif)

    for cnt, label in enumerate(labels, 1):
        tif_ordered[tif==label] = cnt
    
    return tif_ordered

def draw_points(img0, img1):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    values = range(20)

    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.cm.gist_ncar)

    labels = np.unique(img0)[1:]
    # axs[0].axis('equal')
    for idx, label in enumerate(labels):
        colorVal = scalarMap.to_rgba(values[idx])
        point = np.stack(np.where(img0==label), axis=0).T
        axs[0].scatter(point[:, 0], point[:, 1], c=colorVal)
        axs[0].set_title('pred')
    
    labels = np.unique(img1)[1:]
    # axs[1].axis('equal')
    for idx, label in enumerate(labels):
        colorVal = scalarMap.to_rgba(values[idx])
        point = np.stack(np.where(img1==label), axis=0).T
        axs[1].scatter(point[:, 0], point[:, 1], c=colorVal)
        axs[1].set_title('gt')

    fig.savefig(os.path.join(out_dir, name+"_color.jpg"))
    plt.close(fig)



def func1():

    no_emb_path = "/home/jjx/Biology/logs/test_emb_48_with_branch_no_emb/visual_results/"
    dsc_path = "/home/jjx/Biology/logs/test_emb_48_with_branch_dsc_48/visual_results/"
    root_path = "/home/jjx/Biology/logs/test_emb_48_with_branch_tcl_48/visual_results/"
    paths = glob.glob(os.path.join(root_path, "*"))

    out_dir = "./result/test_emb_48_/"
    mkdir_if_not_exist(out_dir)

    for path in paths:
        name = path.split("/")[-1]

        print(path, name)
        gt = orderedLabel(tifffile.imread(os.path.join(path, "ins_gt.tif")))
        tif_tcl = orderedLabel(tifffile.imread(os.path.join(path, "ins_pred.tif")))
        tif_no_emb = orderedLabel(tifffile.imread(os.path.join(no_emb_path, name, "ins_pred.tif")))
        tif_dsc = orderedLabel(tifffile.imread(os.path.join(dsc_path, name, "ins_pred.tif")))

        w, h, d = tif_tcl.shape
        project_tif_tcl = np.max(tif_tcl, axis=0).astype(int)
        project_tif_dsc= np.max(tif_dsc, axis=0).astype(int)
        project_tif_no_emb = np.max(tif_no_emb, axis=0).astype(int)
        project_gt = np.max(gt, axis=0).astype(int)

        # proj = np.hstack([project_tif, -1*np.ones((h, 1)), project_gt])
        # proj = np.pad(proj, ((1, 1), (1, 1)), 'constant', constant_values=-1)

        # plt.imsave(os.path.join(out_dir, name+"_color.jpg"), proj, cmap=plt.cm.gist_ncar)
        # plt.imsave(os.path.join(out_dir, name+"_color.jpg"), proj, cmap=plt.cm.gray)
        
        fig, axs = plt.subplots(1, 4, figsize=(60, 15))
    
        axs[0].imshow(project_gt, cmap=plt.cm.gist_ncar)
        axs[0].set_title("Ground Truth", fontsize=60)

        axs[1].imshow(project_tif_no_emb, cmap=plt.cm.gist_ncar)
        axs[1].set_title("no emb", fontsize=60)

        axs[2].imshow(project_tif_dsc, cmap=plt.cm.gist_ncar)
        axs[2].set_title("dsc", fontsize=60)

        axs[3].imshow(project_tif_tcl, cmap=plt.cm.gist_ncar)
        axs[3].set_title("tcl", fontsize=60)
        
        # for i in range(4):
        #     axs[i].set_xlabel("Width", fontsize=60)
        #     axs[i].set_xlabel("Height", fontsize=60)
        
        fig.savefig(os.path.join(out_dir, name+"_color.jpg"))
        plt.close(fig)


        # draw_points(project_tif, project_gt)

        # proj_ = color.label2rgb(proj)
        # proj_[proj==0, :] = [1, 1, 1]
        # proj_[proj==-1, :] = [0, 0, 0]
        # io.imsave(os.path.join(out_dir, name+"_color.jpg"), proj_, quality=100)

def func2():
    root_path = "/home/jjx/Biology/data/data_modified_emb_gt_48_center_crop/test/"
    # root_path = "/home/jjx/Biology/test/test"
    root_path = "/home/jjx/Biology/NeuralTrack2/test/test/"

    paths = glob.glob(os.path.join(root_path, "ins/*"))
    paths.sort()
    
    out_dir = "./result/test_test/"
    mkdir_if_not_exist(out_dir)

    for path in tqdm(paths):
        name = os.path.basename(path)
        
        ins = orderedLabel(tifffile.imread(path))
        over_seg = orderedLabel(tifffile.imread(os.path.join(root_path, "over_segs", name)))
        under_seg = orderedLabel(tifffile.imread(os.path.join(root_path, "under_segs", name)))

        proj_ins = np.max(ins, axis=0).astype(int)
        proj_over_seg = np.max(over_seg, axis=0).astype(int)
        proj_under_seg = np.max(under_seg, axis=0).astype(int)

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        
        axs[0].imshow(proj_ins, cmap=plt.cm.gist_ncar)
        axs[0].set_title("Ins", fontsize=40)

        axs[1].imshow(proj_over_seg, cmap=plt.cm.gist_ncar)
        axs[1].set_title("overSeg", fontsize=40)

        axs[2].imshow(proj_under_seg, cmap=plt.cm.gist_ncar)
        axs[2].set_title("underSeg", fontsize=40)

        fig.savefig(os.path.join(out_dir, name.split(".")[0]+"_color.jpg"))
        plt.close(fig)

def func3():
    root_path = "/home/jjx/Biology/logs/test_dsmc_emb_center_crop_synthesize/visual_results/"

    out_path = "./result/test_dsmc_emb_center_crop_synthesize/"
    mkdir_if_not_exist(out_path)

    names = os.listdir(root_path)
    names.sort()
    for name in tqdm(names):
        # print(name)
        ins_paths = glob.glob(os.path.join(root_path, name, "*_gt.tif"))
        
        fig, axs = plt.subplots(len(ins_paths), 4, figsize=(40, 10*len(ins_paths)))
        for i, ins_path in enumerate(ins_paths):
            rd = os.path.basename(ins_path).split(".")[0]
            ins_name = "_".join(rd.split('_')[:-1])
            
            seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_seg.tif")))
            gt = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_gt.tif")))
            over_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_over_seg.tif")))
            under_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_under_seg.tif")))

            proj_seg = np.max(seg, axis=0).astype(int)
            proj_gt = np.max(gt, axis=0).astype(int)
            proj_over_seg = np.max(over_seg, axis=0).astype(int)
            proj_under_seg = np.max(under_seg, axis=0).astype(int)

            if len(ins_paths) == 1:
                axs[0].imshow(proj_gt, cmap=plt.cm.gist_ncar)
                axs[0].set_title(ins_name+"_gt", fontsize=40)

                axs[1].imshow(proj_seg, cmap=plt.cm.gist_ncar)
                axs[1].set_title(ins_name+"_seg", fontsize=40)
                
                axs[2].imshow(proj_over_seg, cmap=plt.cm.gist_ncar)
                axs[2].set_title(ins_name+"_over", fontsize=40)
                
                axs[3].imshow(proj_under_seg, cmap=plt.cm.gist_ncar)
                axs[3].set_title(ins_name+"_under", fontsize=40)
                
            else:
                axs[i, 0].imshow(proj_gt, cmap=plt.cm.gist_ncar)
                axs[i, 0].set_title(ins_name+"_gt", fontsize=40)

                axs[i, 1].imshow(proj_seg, cmap=plt.cm.gist_ncar)
                axs[i, 1].set_title(ins_name+"_seg", fontsize=40)

                axs[i, 2].imshow(proj_over_seg, cmap=plt.cm.gist_ncar)
                axs[i, 2].set_title(ins_name+"_over", fontsize=40)

                axs[i, 3].imshow(proj_under_seg, cmap=plt.cm.gist_ncar)
                axs[i, 3].set_title(ins_name+"_under", fontsize=40)

        fig.savefig(os.path.join(out_path, name+"_color.jpg"))
        plt.close(fig)

def add_patches2axes(axes, coords):
    """ 以 coords为中心, 画矩形
    """
    reso = [10, 20]
    color = ['r', 'y']
    rects = []
    for _ in axes:
        res = []
        for c in coords:
            for i, r in enumerate(reso):
                a = patches.Rectangle((c[1]-r//2, c[0]-r//2), r, r, linewidth=2, edgecolor=color[i], facecolor='none')
                res.append(a)
        rects.append(res)

    for ax, rect in zip(axes, rects):
        for r in rect:
            ax.add_patch(r)

def func4():
    root_path = "/home/jjx/Biology/logs/test_dsc_emb_center_crop/visual_results/"

    out_path = "./result/test_dsc_emb_center_crop/"
    mkdir_if_not_exist(out_path)

    names = os.listdir(root_path)
    names.sort()
    for name in tqdm(names):
        # print(name)
        ins_paths = glob.glob(os.path.join(root_path, name, "*_gt.tif"))
        
        fig, axs = plt.subplots(len(ins_paths), 4, figsize=(40, 10*len(ins_paths)))
        for i, ins_path in enumerate(ins_paths):
            rd = os.path.basename(ins_path).split(".")[0]
            ins_name = "_".join(rd.split('_')[:-1])
            
            seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_seg.tif")))
            gt = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_gt.tif")))
            over_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_over_seg.tif")))
            under_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_under_seg.tif")))
            cross_seg = tifffile.imread(os.path.join(root_path, name, ins_name+"_cross_seg.tif"))

            proj_seg = np.max(seg, axis=0).astype(int)
            proj_gt = np.max(gt, axis=0).astype(int)
            proj_over_seg = np.max(over_seg, axis=0).astype(int)
            proj_under_seg = np.max(under_seg, axis=0).astype(int)
            proj_cross = np.max(cross_seg, axis=0).astype(int)

            # cross box, red3, yellow5, green10
            cross_coords = np.stack(np.where(proj_cross), axis=0).T

            if len(ins_paths) == 1:
                axs[0].imshow(proj_gt, cmap=plt.cm.gist_ncar)
                axs[0].set_title(ins_name+"_gt", fontsize=40)

                axs[1].imshow(proj_seg, cmap=plt.cm.gist_ncar)
                axs[1].set_title(ins_name+"_seg", fontsize=40)
                
                axs[2].imshow(proj_over_seg, cmap=plt.cm.gist_ncar)
                axs[2].set_title(ins_name+"_over", fontsize=40)
                
                axs[3].imshow(proj_under_seg, cmap=plt.cm.gist_ncar)
                axs[3].set_title(ins_name+"_under", fontsize=40)

                add_patches2axes(axs, cross_coords)
            else:
                axs[i, 0].imshow(proj_gt, cmap=plt.cm.gist_ncar)
                axs[i, 0].set_title(ins_name+"_gt", fontsize=40)

                axs[i, 1].imshow(proj_seg, cmap=plt.cm.gist_ncar)
                axs[i, 1].set_title(ins_name+"_seg", fontsize=40)

                axs[i, 2].imshow(proj_over_seg, cmap=plt.cm.gist_ncar)
                axs[i, 2].set_title(ins_name+"_over", fontsize=40)

                axs[i, 3].imshow(proj_under_seg, cmap=plt.cm.gist_ncar)
                axs[i, 3].set_title(ins_name+"_under", fontsize=40)

                add_patches2axes(axs[i], cross_coords)

        fig.savefig(os.path.join(out_path, name+"_color.jpg"))
        plt.close(fig)

def func5():
    root_path = "/home/jjx/Biology/logs/test_synthesize_embseg_dsmc_48_center_crop/visual_results/"

    out_path = "./result/test_synthesize_embseg_dsmc_48_center_crop/"
    mkdir_if_not_exist(out_path)

    names = os.listdir(root_path)
    names.sort()
    for name in tqdm(names):
        # print(name)
        ins_paths = glob.glob(os.path.join(root_path, name, "*_gt.tif"))
        
        fig, axs = plt.subplots(len(ins_paths), 2, figsize=(20, 10*len(ins_paths)))
        for i, ins_path in enumerate(ins_paths):
            rd = os.path.basename(ins_path).split(".")[0]
            ins_name = "_".join(rd.split('_')[:-1])
            
            merge = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_merge.tif")))
            gt = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_gt.tif")))

            proj_merge = np.max(merge, axis=0).astype(int)
            proj_gt = np.max(gt, axis=0).astype(int)

            if len(ins_paths) == 1:
                axs[0].imshow(proj_gt, cmap=plt.cm.gist_ncar)
                axs[0].set_title(ins_name+"_gt", fontsize=20)

                axs[1].imshow(proj_merge, cmap=plt.cm.gist_ncar)
                axs[1].set_title(ins_name+"_merge", fontsize=20)
                
            else:
                axs[i, 0].imshow(proj_gt, cmap=plt.cm.gist_ncar)
                axs[i, 0].set_title(ins_name+"_gt", fontsize=20)

                axs[i, 1].imshow(proj_merge, cmap=plt.cm.gist_ncar)
                axs[i, 1].set_title(ins_name+"_merge", fontsize=20)

        fig.savefig(os.path.join(out_path, name+"_color.jpg"))
        plt.close(fig)

def func6():
    root_path = "/home/jjx/Biology/logs/test_dsmc_emb_center_crop_synthesize/visual_results/"
    root_path = "/home/jjx/Biology/logs/test_synthesize_embcls_tcl_64_center_crop_merge_0.5/visual_results/"

    out_path = "./result/test_synthesize_embcls_tcl_64_center_crop_merge_0.5/"
    mkdir_if_not_exist(out_path)

    names = os.listdir(root_path)
    names.sort()
    for name in tqdm(names):
        # print(name)
        ins_paths = glob.glob(os.path.join(root_path, name, "*_gt.tif"))
        
        fig, axs = plt.subplots(len(ins_paths), 4, figsize=(40, 10*len(ins_paths)))
        for i, ins_path in enumerate(ins_paths):
            rd = os.path.basename(ins_path).split(".")[0]
            ins_name = "_".join(rd.split('_')[:-1])
            
            merge = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_merge.tif")))
            gt = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_gt.tif")))
            over_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, "over_seg.tif")))
            under_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, "under_seg.tif")))

            proj_merge = np.max(merge, axis=0).astype(int)
            proj_gt = np.max(gt, axis=0).astype(int)
            proj_over_seg = np.max(over_seg, axis=0).astype(int)
            proj_under_seg = np.max(under_seg, axis=0).astype(int)

            if len(ins_paths) == 1:
                axs[0].imshow(proj_gt, cmap=plt.cm.gist_ncar)
                axs[0].set_title(ins_name+"_gt", fontsize=40)

                axs[1].imshow(proj_merge, cmap=plt.cm.gist_ncar)
                axs[1].set_title(ins_name+"_merge", fontsize=40)
                
                axs[2].imshow(proj_over_seg, cmap=plt.cm.gist_ncar)
                axs[2].set_title(ins_name+"_over", fontsize=40)
                
                axs[3].imshow(proj_under_seg, cmap=plt.cm.gist_ncar)
                axs[3].set_title(ins_name+"_under", fontsize=40)
                
            else:
                axs[i, 0].imshow(proj_gt, cmap=plt.cm.gist_ncar)
                axs[i, 0].set_title(ins_name+"_gt", fontsize=40)

                axs[i, 1].imshow(proj_seg, cmap=plt.cm.gist_ncar)
                axs[i, 1].set_title(ins_name+"_seg", fontsize=40)

                axs[i, 2].imshow(proj_over_seg, cmap=plt.cm.gist_ncar)
                axs[i, 2].set_title(ins_name+"_over", fontsize=40)

                axs[i, 3].imshow(proj_under_seg, cmap=plt.cm.gist_ncar)
                axs[i, 3].set_title(ins_name+"_under", fontsize=40)

        fig.savefig(os.path.join(out_path, name+"_color.jpg"))
        plt.close(fig)

def func7():
    root_path = "/home/jjx/Biology/NeuralTrack2/test/test/"

    paths = glob.glob(os.path.join(root_path, "ins/*"))
    paths.sort()
    
    out_dir = "./result/test_test/"
    mkdir_if_not_exist(out_dir)

    for path in tqdm(paths):
        name = os.path.basename(path)
        
        ins = orderedLabel(tifffile.imread(path))
        over_seg = orderedLabel(tifffile.imread(os.path.join(root_path, "over_segs", name)))
        under_seg = orderedLabel(tifffile.imread(os.path.join(root_path, "under_segs", name)))

        proj_ins = np.max(ins, axis=0).astype(int)
        proj_over_seg = np.max(over_seg, axis=0).astype(int)
        proj_under_seg = np.max(under_seg, axis=0).astype(int)

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        
        axs[0].imshow(proj_ins, cmap=plt.cm.gist_ncar)
        axs[0].set_title("Ins", fontsize=40)

        axs[1].imshow(proj_over_seg, cmap=plt.cm.gist_ncar)
        axs[1].set_title("overSeg", fontsize=40)

        axs[2].imshow(proj_under_seg, cmap=plt.cm.gist_ncar)
        axs[2].set_title("underSeg", fontsize=40)

        h, w = proj_ins.shape
        rects = []
        for _ in range(len(axs)):
            a = patches.Rectangle((10, 10), h-20, w-20, linewidth=2, edgecolor='r', facecolor='none')
            rects.append(a)

        for ax, rect in zip(axs, rects):
                ax.add_patch(rect)

        fig.savefig(os.path.join(out_dir, name.split(".")[0]+"_color.jpg"))
        plt.close(fig)

if __name__ == "__main__":
    func7()