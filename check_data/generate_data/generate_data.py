from random import choice
import random
import math
import numpy as np

PI = math.pi

def deltaAngle(a=1/10):
    angle1 = a * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
    angle2 = 0.5 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
    angle = np.random.choice([angle1, angle2], replace=False, p=[0.95, 0.05])
    return angle


class GenerateLine2D():
    """A class to generate data."""
    
    def __init__(self, ini_coords=[], init_phi=PI/4, num_points=5000):
        """Initialize attributes of a data."""
        self.num_points = num_points
        
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
            self.deltaPhi = deltaAngle(a=1/3)  #1/3 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
            self.phi += self.deltaPhi
            self.phi_.append(self.phi)
            r = 10 * random.uniform(0, 1)

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

def generate_branch2D(ini_coords=[0, 0], num_branch=4):
   
    gl = []
    num_branch = num_branch

    num = int(random.gauss(mu=60, sigma=8))
    gl0 = GenerateLine2D(ini_coords=ini_coords, num_points=num)
    gl0.fill_points()
    gl.append(gl0)
    
    ind = list(np.random.choice(len(gl0.x_values), size=num_branch, replace=False))
    #ind = [i for i in ind]
    init_coords = [np.array(gl0.x_values)[ind], np.array(gl0.y_values)[ind]]
    init_phis = np.array(gl0.phi_)[ind]

    for i in range(num_branch):
        ini_coord = [init_coords[0][i], init_coords[1][i]]
        gl_i = GenerateLine2D(ini_coords=ini_coord,
                            init_phi=init_phis[i]+random.uniform(0, PI/2),
                            num_points=int(random.gauss(mu=12, sigma=3)))
        gl_i.fill_points()
        gl.append(gl_i)

    return gl

def generate_multi_branch2D():
    branch_all = []
    branch0 = generate_branch2D(ini_coords=[0, 0], num_branch=3)
    branch_all = branch0[:]

    for branch in branch0[1:]:
        ind = list(np.random.choice(branch.num_points, size=2, replace=False))
        init_coords = [np.array(branch.x_values)[ind], np.array(branch.y_values)[ind]]
        init_phis = np.array(branch.phi_)[ind]
        for i in range(2):
            ini_coord = [init_coords[0][i], init_coords[1][i]]
            gl_i = GenerateLine2D(ini_coords=ini_coord,
                                  init_phi=init_phis[i]+random.uniform(0, PI/2),
                                  num_points=int(random.gauss(mu=8, sigma=2)))
            gl_i.fill_points()
            branch_all.append(gl_i)

    return branch_all

def gene_n_degree_branch2D(num_degree=[1, 3]):
    """n_degree == len(num_degree)
    """
    branch_all = []
    n_degree = len(num_degree)

    # n_degree=0
    num = int(random.gauss(mu=1200, sigma=15))
    gl0 = GenerateLine_2Direct_2D(ini_coords=[0, 0], num_points=num)
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
                gl_i = GenerateLine_2Direct_2D(ini_coords=ini_coord,
                                    init_phi=init_phis[j]+random.uniform(0, PI/2),
                                    num_points=int(random.gauss(mu=1200/(i+1), sigma=15/(i+1))))
                gl_i.fill_points()
                gl_.append(gl_i)
        
        branch_all += gl_
        count += temp * num_degree[i-1]
        temp = temp * num_degree[i-1]

    return branch_all


class GenerateLine_2Direct_2D():
    """A class to generate data."""
    
    def __init__(self, ini_coords=[], init_phi=PI/4, num_points=5000):
        """Initialize attributes of a data."""
        self.num_points = num_points
        
        # All datas start at (0, 0).
        self.ini_coords = ini_coords
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
        #while len(self.x_values) < self.num_points:
            
        direct1_num_points = random.randint(1, self.num_points)
        print(direct1_num_points)
        direct2_num_points = self.num_points - direct1_num_points

        for _ in range(direct1_num_points):
            # Decide which direction to go, and how far to go in that direction.
            
            self.theta = PI / 2#PI * random.random()
            #self.phi = 1 * random.gauss(mu=self.phi, sigma=random.uniform(PI/6, PI/3))  # sigma=random.uniform(PI/18, PI/3)
            #arctan
            #self.deltaPhi = (PI/6 - (-PI/6)) * random.uniform(0, 1) - PI / 6
            self.deltaPhi = deltaAngle(a=1/3) #1/3 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
            self.phi += self.deltaPhi
            self.phi_.append(self.phi)
            r = 5 * random.uniform(0, 1)
            
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

        
        self.x_values.append(self.ini_coords[0])
        self.y_values.append(self.ini_coords[1])
    
        self.theta = 0
        self.phi = self.phi_[0] + PI
        self.phi_.append(self.phi)
        self.deltaPhi = 0
        for _ in range(direct2_num_points):
            # Decide which direction to go, and how far to go in that direction.
            
            self.theta = PI / 2#PI * random.random()
            #self.phi = 1 * random.gauss(mu=self.phi, sigma=random.uniform(PI/6, PI/3))  # sigma=random.uniform(PI/18, PI/3)
            #arctan
            #self.deltaPhi = (PI/6 - (-PI/6)) * random.uniform(0, 1) - PI / 6
            self.deltaPhi = deltaAngle(a=1/3)  #1/3 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
            self.phi += self.deltaPhi
            self.phi_.append(self.phi)
            r = 5 * random.uniform(0, 1)
            
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
        
        # 删除第二个起始点
        del self.x_values[direct1_num_points]
        del self.y_values[direct1_num_points]
        del self.phi_[direct1_num_points]


def GenerateLine_2Direct_2D_Disturb():
    pass





class GenerateLine3D():
    """A class to generate random datas."""
    
    def __init__(self, init_coords=[], init_theta=0, init_phi=0, num_points=5000):
        """Initialize attributes of a data."""
        self.num_points = num_points
        
        # All datas start at (0, 0).
        self.x_values = [init_coords[0]]
        self.y_values = [init_coords[1]]
        self.z_values = [init_coords[2]]
        
        self.theta_ = []
        self.phi_ = []
                
        self.theta = 0
        self.theta_.append(self.theta)

        self.phi = init_phi
        self.phi_.append(self.phi)

        self.deltaTheta = 0
        self.deltaPhi = 0


    def fill_points(self):
        """Calculate all the points in the data."""
        
        # Keep taking steps until the data reaches the desired length.
        while len(self.x_values) < self.num_points:
            
            # Decide which direction to go, and how far to go in that direction.
            
            self.deltaTheta = 1/4 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
            self.theta += self.deltaTheta #PI * random.random()
            self.theta_.append(self.theta)
            
            #self.phi = 1 * random.gauss(mu=self.phi, sigma=random.uniform(PI/6, PI/3))  # sigma=random.uniform(PI/18, PI/3)
            #self.deltaPhi = (PI/6 - (-PI/6)) * random.uniform(0, 1) - PI / 6
            self.deltaPhi = 1/4 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
            self.phi += self.deltaPhi
            self.phi_.append(self.phi)
            
            r = 10 * random.uniform(0, 1)
            
            x_direction = 1 * math.sin(self.theta) * math.sin(self.phi)
            y_direction = 1 * math.sin(self.theta) * math.cos(self.phi)
            z_direction = 1 * math.cos(self.theta)
            
            #x_direction = choice([1])
            x_distance = r
            x_step = (x_direction) * x_distance
            
            #y_direction = choice([1])
            y_distance = r
            y_step = (y_direction) * y_distance
            
            z_distance = r
            z_step = (z_direction) * z_distance
            # Reject moves that go nowhere.
            if x_step == 0 and y_step == 0 and z_step:
                continue
            
            # Calculate the next x and y values.
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step
            next_z = self.z_values[-1] + z_step

            self.x_values.append(next_x)
            self.y_values.append(next_y)
            self.z_values.append(next_z)


def generate_branch3D(num_branch=4):
   
    gl = []
    num_branch = num_branch

    num = int(random.gauss(mu=30, sigma=8))
    gl0 = GenerateLine3D(init_coords=[0, 0, 0], num_points=num)
    gl0.fill_points()
    gl.append(gl0)
    
    ind = list(np.random.choice(len(gl0.x_values), size=num_branch, replace=False))
    #ind = [i for i in ind]
    init_coords = [np.array(gl0.x_values)[ind], np.array(gl0.y_values)[ind], np.array(gl0.z_values)[ind]]
    init_thetas = np.array(gl0.theta_)[ind]
    init_phis = np.array(gl0.phi_)[ind]

    for i in range(num_branch):
        ini_coord = [init_coords[0][i], init_coords[1][i], init_coords[2][i]]
        gl_i = GenerateLine3D(init_coords=ini_coord,
                            init_theta=init_thetas[i]+random.uniform(0, PI/2),
                            init_phi=init_phis[i]+random.uniform(0, PI/2),
                            num_points=int(random.gauss(mu=20, sigma=8)))
        gl_i.fill_points()
        gl.append(gl_i)

    return gl

def gene_n_degree_branch3D(num_degree=[1, 3]):
    """n_degree == len(num_degree)
    """
    branch_all = []
    n_degree = len(num_degree)

    # n_degree=0
    num = int(random.gauss(mu=1500, sigma=12))
    gl0 = GenerateLine_2Direct_3D(init_coords=[0, 0, 0], num_points=num)
    gl0.fill_points()

    branch_all.append(gl0)
    
    count = 0
    temp = 1
    for i in range(1, n_degree):
        
        gl_ = []
        for gl in branch_all[count : (count+temp*num_degree[i-1])]:
             
            ind = list(np.random.choice(gl.num_points, size=num_degree[i], replace=False))

            init_coords = [np.array(gl.x_values)[ind], np.array(gl.y_values)[ind], np.array(gl.z_values)[ind]]
            init_thetas = np.array(gl.theta_)[ind]
            init_phis = np.array(gl.phi_)[ind]

            for j in range(num_degree[i]):
                ini_coord = [init_coords[0][j], init_coords[1][j], init_coords[2][j]]
                gl_i = GenerateLine_2Direct_3D(init_coords=ini_coord,
                                      init_theta=init_thetas[j]+random.uniform(0, PI/2),
                                      init_phi=init_phis[j]+random.uniform(0, PI/2),
                                      num_points=int(random.gauss(mu=1500/(i+1), sigma=12/(i+1))))
                gl_i.fill_points()
                gl_.append(gl_i)
        
        branch_all += gl_
        count += temp * num_degree[i-1]
        temp = temp * num_degree[i-1]

    return branch_all


class GenerateLine_2Direct_3D():
    """A class to generate random datas."""
    
    def __init__(self, init_coords=[], init_theta=0, init_phi=0, num_points=5000):
        """Initialize attributes of a data."""
        self.num_points = num_points
        
        # All datas start at (0, 0).
        self.init_coords = init_coords
        self.x_values = [init_coords[0]]
        self.y_values = [init_coords[1]]
        self.z_values = [init_coords[2]]
        
        self.theta_ = []
        self.phi_ = []
                
        self.theta = 0
        self.theta_.append(self.theta)

        self.phi = init_phi
        self.phi_.append(self.phi)

        self.deltaTheta = 0
        self.deltaPhi = 0


    def fill_points(self):
        """Calculate all the points in the data."""
        
        # Keep taking steps until the data reaches the desired length.
        #while len(self.x_values) < self.num_points:
        direct1_num_points = random.randint(1, self.num_points)
        #print(direct1_num_points)
        direct2_num_points = self.num_points - direct1_num_points

        for _ in range(direct1_num_points):

            # Decide which direction to go, and how far to go in that direction.
            
            self.deltaTheta = deltaAngle(1/10)  #1/10 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
            self.theta += self.deltaTheta #PI * random.random()
            self.theta_.append(self.theta)
            
            #self.phi = 1 * random.gauss(mu=self.phi, sigma=random.uniform(PI/6, PI/3))  # sigma=random.uniform(PI/18, PI/3)
            #self.deltaPhi = (PI/6 - (-PI/6)) * random.uniform(0, 1) - PI / 6
            self.deltaPhi = deltaAngle(1/10)  #1/10 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
            self.phi += self.deltaPhi
            self.phi_.append(self.phi)
            
            r = 1 * random.uniform(0, 1)
            
            x_direction = 1 * math.sin(self.theta) * math.sin(self.phi)
            y_direction = 1 * math.sin(self.theta) * math.cos(self.phi)
            z_direction = 1 * math.cos(self.theta)
            
            #x_direction = choice([1])
            x_distance = r
            x_step = (x_direction) * x_distance
            
            #y_direction = choice([1])
            y_distance = r
            y_step = (y_direction) * y_distance
            
            z_distance = r
            z_step = (z_direction) * z_distance
            # Reject moves that go nowhere.
            if x_step == 0 and y_step == 0 and z_step:
                continue
            
            # Calculate the next x and y values.
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step
            next_z = self.z_values[-1] + z_step

            self.x_values.append(next_x)
            self.y_values.append(next_y)
            self.z_values.append(next_z)


        self.x_values.append(self.init_coords[0])
        self.y_values.append(self.init_coords[1])
        self.z_values.append(self.init_coords[2])
                    
        self.theta = PI - self.theta_[0]
        self.theta_.append(self.theta)

        self.phi = self.phi_[0] + PI
        self.phi_.append(self.phi)

        self.deltaTheta = 0
        self.deltaPhi = 0
        for _ in range(direct2_num_points):

            # Decide which direction to go, and how far to go in that direction.
            
            self.deltaTheta = deltaAngle(1/10)  #1/10 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
            self.theta += self.deltaTheta #PI * random.random()
            self.theta_.append(self.theta)
            
            #self.phi = 1 * random.gauss(mu=self.phi, sigma=random.uniform(PI/6, PI/3))  # sigma=random.uniform(PI/18, PI/3)
            #self.deltaPhi = (PI/6 - (-PI/6)) * random.uniform(0, 1) - PI / 6
            self.deltaPhi = deltaAngle(1/10)  #1/10 * math.atan(random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3)))
            self.phi += self.deltaPhi
            self.phi_.append(self.phi)
            
            r = 1 * random.uniform(0, 1)
            
            x_direction = 1 * math.sin(self.theta) * math.sin(self.phi)
            y_direction = 1 * math.sin(self.theta) * math.cos(self.phi)
            z_direction = 1 * math.cos(self.theta)
            
            #x_direction = choice([1])
            x_distance = r
            x_step = (x_direction) * x_distance
            
            #y_direction = choice([1])
            y_distance = r
            y_step = (y_direction) * y_distance
            
            z_distance = r
            z_step = (z_direction) * z_distance
            # Reject moves that go nowhere.
            if x_step == 0 and y_step == 0 and z_step:
                continue
            
            # Calculate the next x and y values.
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step
            next_z = self.z_values[-1] + z_step

            self.x_values.append(next_x)
            self.y_values.append(next_y)
            self.z_values.append(next_z)

        # 删除第二个起始点
        del self.x_values[direct1_num_points]
        del self.y_values[direct1_num_points]
        del self.z_values[direct1_num_points]
        del self.phi_[direct1_num_points]
        del self.theta_[direct1_num_points]
