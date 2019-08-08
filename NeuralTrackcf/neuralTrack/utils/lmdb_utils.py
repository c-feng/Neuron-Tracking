import numpy as np

def img_to_lmdb(img):
    img_str = img.flatten().tobytes()
    img_size_str = np.array(img.shape).tobytes()

    return img_str, img_size_str

def lmdb_to_img(img_str, img_size_str):
    img = np.frombuffer(img_str, dtype = np.int)
    img_size = np.frombuffer(img_size_str, dtype = np.int)
    img = img.reshape(img_size)
    return img
