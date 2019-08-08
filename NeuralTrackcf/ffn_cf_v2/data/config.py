
neural_train = {
    'list_file': "./data/train.json",
    'data_root': "/media/fcheng/datasets/data_modified/",
    'max_iter': 12000,
    'lr_steps': [30, 80, 120],
    'num_classes': 2,
    'subvol_shape': [45,45,45,1],
    'fov_shape': [33,33,33,1],
    'delta': [6,6,6],
    'tmove': 0.9,
}

neural_test = {
    'list_file': "./data/test.json",
    'data_root': "/media/fcheng/datasets/data_modified/",
    'max_iter': 12000,
    'num_classes': 2,
    'sample_shape': [33,33,33,1],
    'subvol_shape': [300,300,300,1],
    'fov_shape': [33, 33, 33,1],
    'delta': [6,6,6],
    'tmove': 0.9,
}