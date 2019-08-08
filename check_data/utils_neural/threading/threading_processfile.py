import os
import numpy as np
import threading
from multiprocessing import Pool
from functools import partial
from queue import Queue
import glob
from skimage.external import tifffile
import matplotlib.pyplot as plt

def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def produce_file(dir_path):
    file_paths = glob.glob(os.path.join(dir_path, "*.tif"))
    return file_paths

def write_file(label, tif, out_dir, name):
    i = int(label)
    ins = tif==i
    tifffile.imsave(os.path.join(out_dir, name+'_'+str(i)+'.tif'), ins.astype(np.float16))

    w, h, d = ins.shape
    ins_0 = (np.sum(ins, axis=0)>0).astype(int)
    ins_1 = (np.sum(ins, axis=1)>0).astype(int)
    ins_2 = (np.sum(ins, axis=2)>0).astype(int)
    ins_xyz = np.hstack([ins_0, np.ones((h, 1)), ins_1, np.ones((h, 1)), ins_2])
    
    plt.imsave(os.path.join(out_dir, name+'_'+str(i)+'.jpg'), ins_xyz, cmap='gray')
    print("\r{}_{} have been saved.".format(name, i), end='')

def multi_process_func(labels, tif, out_dir, name, num_proc=4):
    with Pool(processes=num_proc) as pool:
        ps = pool.map(partial(write_file, tif=tif, out_dir=out_dir, name=name), labels)
    return ps

class ThreadProcessFile(threading.Thread):
    def __init__(self, queue, tif, file_path):
        threading.Thread.__init__(self)
        self.queue = queue
        self.tif = tif
        self.file_path = file_path
        self.root = os.path.dirname(file_path)
        self.name = os.path.basename(file_path).split('.')[0]
        self.out_dir = os.path.join(self.root, self.name)
        mkdir_if_not_exist(self.out_dir)
    
    def run(self):
        while True:
            # 从队列中获取文件路径, 读取文件, 写入文件
            ins_label = self.queue.get()

            write_file(ins_label, self.tif, self.out_dir, self.name)
            # 通知队列任务完成
            self.queue.task_done()

def main():
    # path = r"C:\Users\cf__e\Desktop\check_data\threading"
    path = r"D:\test"

    file_paths = produce_file(path)
    for file_path in file_paths:
        queue = Queue()
        tif = tifffile.imread(file_path)

        for _ in range(6):
            t = ThreadProcessFile(queue, tif, file_path)
            t.setDaemon(True)
            t.start()
        
        all_labels = np.unique(tif)[1:]
        for label in all_labels:
            queue.put(label)
        
        queue.join()
        queue.put(None)

def multiprocess():
    # path = r"H:\3450_31350_5150_0\vis"
    # file_paths = produce_file(path)
    file_paths = [r"H:\skeleton_test_data\6950_34350_4150_0_direct.tif"]
    for file_path in file_paths:
        tif = tifffile.imread(file_path)
        
        root = os.path.dirname(file_path)
        name = os.path.basename(file_path).split('.')[0]
        out_dir = os.path.join(root, name)
        mkdir_if_not_exist(out_dir)

        labels = np.unique(tif)[1:]
        print("Processing {} ...".format(name))
        multi_process_func(labels, tif, out_dir, name, num_proc=4)

if __name__ == "__main__":
    # main()
    multiprocess()

    