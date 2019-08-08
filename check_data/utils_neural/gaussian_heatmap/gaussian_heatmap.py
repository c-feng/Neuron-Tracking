import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

def CenterLabel_2dHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(0, img_width-1, img_width)
    Y1 = np.linspace(0, img_height-1, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap

# Compute gaussian kernel
def CenterGaussian_2dHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map


def CenterGaussian_3dHeatMap(w, h, d, c_x, c_y, c_z, sigma):
    X1 = np.linspace(0, w-1, w)
    Y1 = np.linspace(0, h-1, h)
    Z1 = np.linspace(0, d-1, d)

    [X, Y, Z] = np.meshgrid(X1, Y1, Z1)

    X = X - c_x
    Y = Y - c_y
    Z = Z - c_z
    D = X * X + Y * Y + Z * Z
    E = 2.0 * sigma * sigma
    Exponent = D / E
    heatmap = np.exp(-Exponent)
    return heatmap


def gaussiantry():
    # 高斯分布
    #mean = [0,0]
    #cov = [[1,0],[0,1]]
    #x, y = np.random.multivariate_normal(mean, cov, 10000).T
    #
    #hist, xedges, yedges = np.histogram2d(x,y, bins=10)
    #hot = hist / np.max(hist)
    #
    #X,Y = np.meshgrid(xedges,yedges)
    #plt.imshow(hist)
    #plt.grid(True)
    #plt.colorbar()
    #plt.show()

    mean = [0,0,0]
    cov = [[1,0,0],[0,1,0],[0,0,1]]
    x, y, z = np.random.multivariate_normal(mean, cov, 10000).T

    hist, (xedges, yedges, zedges) = np.histogramdd((x,y,z), bins=100)
    hot = (hist / np.max(hist)).astype(np.float32)

    X,Y,Z = np.meshgrid(xedges,yedges, zedges)
    #plt.imshow(hist)
    #plt.grid(True)
    #plt.colorbar()
    #plt.show()

def CenterGaussianHeatMap(shape, center, sigma):
    dims = len(shape)
    lins = []
    for i in range(dims):
        lins += [np.linspace(0, shape[i]-1, shape[i]).tolist()]
    
    coords = np.stack(np.meshgrid(*lins), axis=-1)
    D = 0.01 * np.sum(np.power(coords - center, 2), axis=-1)
    E = 2.0 * sigma * sigma
    Exponent = D /E
    heatmap = np.exp(-Exponent)
    return heatmap


if __name__ == "__main__":
    heat0 = CenterGaussian_3dHeatMap(3, 3, 3, 1, 1, 1, 1).astype(np.float32)
    heat1 = CenterGaussianHeatMap(shape=[3, 3, 3], center=[1, 1, 1], sigma=1)
    heat2d0 = CenterLabel_2dHeatMap(5, 5, 2, 2, 1)
    heat2d = CenterGaussianHeatMap(shape=[51, 51, 51], center=[25, 25, 25], sigma=1)
    plt.imshow(heat2d0)
    # plt.figure()
    # plt.imshow(heat2d, cmap='gray')


