import numpy as np
import math

def generate_theta_phi(p1, p2):
    """ Generate the vector of two points
        Return the radius, theta, phi
    """ 
    EPS = 1e-8
    gap_xyz = np.array([p2[0] - p1[0], p2[1]-p1[1], p2[2]-p1[2]])
    direction = gap_xyz / ( np.power(np.power(gap_xyz, 2).sum(), 0.5) )
    r = np.power(np.power(gap_xyz, 2).sum(), 0.5)

    theta = math.acos(gap_xyz[2] / (r+EPS))
    phi = math.atan(gap_xyz[1] / (gap_xyz[0]+EPS) )

    return theta, phi


if __name__ == "__main__":
    p1 = [0, 0, 0]
    p2 = [1, 1, 1]
    t, p = generate_theta_phi(p1, p2)
    print(t, p)