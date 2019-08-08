import turtle
from random import choice
import random
import math
import numpy as np
import matplotlib.pyplot as plt

PI = math.pi


def draw_brach(brach_length):

  if brach_length > 5:
    if brach_length < 40:
      turtle.color('green')

    else:
      turtle.color('red')

    # 绘制右侧的树枝
    turtle.forward(brach_length)
    print('向前',brach_length)
    turtle.right(25)
    print('右转20')
    draw_brach(brach_length-15)
    # 绘制左侧的树枝
    turtle.left(50)
    print('左转40')
    draw_brach(brach_length-15)

    if brach_length < 40:
      turtle.color('green')

    else:
      turtle.color('red')


    # 返回之前的树枝上
    turtle.right(25)
    print('右转20')
    turtle.backward(brach_length)
    print('返回',brach_length)


class GenerateLine2D():
    """A class to generate data."""
    
    def __init__(self, ini_coords=[], init_phi=0,  r_step=10, num_points=5000):
        """Initialize attributes of a data."""
        self.num_points = num_points
        self.r = r_step
        
        # All datas start at (0, 0).
        self.x_values = [ini_coords[0]]
        self.y_values = [ini_coords[1]]
        
        self.phi_ = []
        
        self.theta = 0
        self.phi = init_phi
        self.phi_.append(self.phi)
        self.deltaPhi = 0

    def fill_points(self):
        """Calculate all the points in the data."""
        
        # Keep taking steps until the data reaches the desired length.
        while len(self.x_values) < self.num_points:
            
            # Decide which direction to go, and how far to go in that direction.
            
            self.theta = PI / 2#PI * random.random()
            #self.phi = 1 * random.gauss(mu=self.phi, sigma=random.uniform(PI/6, PI/3))  # sigma=random.uniform(PI/18, PI/3)
            #arctan
            #self.deltaPhi = (PI/6 - (-PI/6)) * random.uniform(0, 1) - PI / 6
            self.deltaPhi = 1/9 * math.atan(random.gauss(mu=0, sigma=0.1))
            self.phi += self.deltaPhi
            self.phi_.append(self.phi)
            r = self.r 
            
            x_direction = 1 * math.sin(self.theta) * math.sin(self.phi)
            y_direction = 1 * math.sin(self.theta) * math.cos(self.phi)
            #z_direction = 
            
            #x_direction = choice([1])
            x_distance = r
            x_step = (x_direction) * x_distance
            
            #y_direction = choice([1])
            y_distance = r
            y_step = (y_direction) * y_distance
            
            # Reject moves that go nowhere.
            if x_step == 0 and y_step == 0:
                continue
            
            # Calculate the next x and y values.
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step
            
            self.x_values.append(next_x)
            self.y_values.append(next_y)

def gene_n_degree_branch2D(num_degree=[1, 3]):
    """n_degree == len(num_degree)
    """
    branch_all = []
    n_degree = len(num_degree)

    # n_degree=0
    num = int(random.gauss(mu=60, sigma=8))
    gl0 = GenerateLine2D(ini_coords=[0, 0], r_step=10, num_points=num)
    gl0.fill_points()

    branch_all.append(gl0)
    
    count = 0
    temp = 1
    for i in range(1, n_degree):
        
        gl_ = []
        for gl in branch_all[count : (count+temp*num_degree[i-1])]:
             
            ind = list(np.random.choice(gl.num_points, size=num_degree[i], replace=False))

            init_coords = [np.array(gl.x_values)[ind], np.array(gl.y_values)[ind]]
            init_phis = np.array(gl.phi_)[ind]

            for j in range(num_degree[i]):
                ini_coord = [init_coords[0][j], init_coords[1][j]]
                gl_i = GenerateLine2D(ini_coords=ini_coord, r_step=10/i,
                                    init_phi=init_phis[j]+random.choice([PI/6, -PI/6]),
                                    num_points=int(random.gauss(mu=60/(i+1), sigma=8/(i+1))))
                gl_i.fill_points()
                gl_.append(gl_i)
        
        branch_all += gl_
        count += temp * num_degree[i-1]
        temp = temp * num_degree[i-1]

    return branch_all

def vis_2Ddata():
    while True:
        
        #gl_all = generate_branch2D(num_branch=3)
        gl_all = gene_n_degree_branch2D(num_degree=[1, 5, 5, 3, 2, 1, 1, 1, 1, 1])
        
        x_max_v = -1000; x_min_v = 10000
        y_max_v = -1000; y_min_v = 10000
        for gl in gl_all:
            x_max_v = max(x_max_v, max(gl.x_values))
            x_min_v = min(x_min_v, min(gl.x_values))
            y_max_v = max(y_max_v, max(gl.y_values))
            y_min_v = min(y_min_v, min(gl.y_values))

        # Set the size of the plotting window.
        plt.figure(dpi=128, figsize=(3, 3))
        for gl in gl_all:
            # 缩放到(300, 300)
            #x_max = max(gl.x_values); x_min = min(gl.x_values)
            gl.x_values = ((np.array(gl.x_values) - x_min_v) / (x_max_v - x_min_v)) * 300
            
            #y_max = max(gl.y_values); y_min = min(gl.y_values)
            gl.y_values = ((np.array(gl.y_values) - y_min_v) / (y_max_v - y_min_v)) * 300

            
            point_numbers = list(range(gl.num_points))
            #plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues,
            #    edgecolor='none', s=1)
            plt.plot(gl.x_values, gl.y_values) 

            # Emphasize the first and last points.
            # plt.scatter(gl.x_values[0], gl.y_values[0], marker='x', c='green', edgecolors='none', s=10)
            # plt.scatter(gl.x_values[-1], gl.y_values[-1], marker='x', c='red', edgecolors='none', s=10)
            
        # Remove the axes.
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
            
        plt.show()
        
        keep_running = input("Make another data? (y/n): ")
        if keep_running == 'n':
            break

def main():
  # turtle.left(90)
  # turtle.penup()
  # turtle.backward(150)
  # turtle.pendown()
  # turtle.color('red')

  # draw_brach(100)

  # turtle.exitonclick()
  vis_2Ddata()


if __name__ == '__main__':
  main()