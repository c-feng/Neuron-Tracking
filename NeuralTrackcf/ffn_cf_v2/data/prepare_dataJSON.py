import os
import json
import glob

def write_json(obj,fpath):
    dir = os.path.dirname(fpath)
    #osutils.mkdir_if_missing(dir)
    with open(fpath,'w') as f:
        json.dump(obj,f,indent=4)

dataset_dir = "/home/jjx/Biology/data_modified/"

output_dir = './'

train = {"tif": [],
        "ins": [],
        "gt": []}
test = {"tif": [],
        "ins": [],
        "gt": []}

tiff_paths = glob.glob(os.path.join(dataset_dir, 'tiffs', "*.tif"))
ins_paths = glob.glob(os.path.join(dataset_dir, 'ins_modified', "*.tif"))
gts_paths = glob.glob(os.path.join(dataset_dir, 'gts', "*.tif"))

tiff_names = [os.path.split(i)[1] for i in tiff_paths]
ins_names = [os.path.split(i)[1] for i in ins_paths]
gts_names = [os.path.split(i)[1] for i in gts_paths]

img_paths = [os.path.join("tiffs", i) for i in tiff_names]
ins_paths = [os.path.join("ins_modified", i) for i in ins_names]
gt_paths = [os.path.join("gts", i) for i in ins_names]

# img_paths, ins_paths = tiff_paths, ins_paths
img_paths.sort(), ins_paths.sort(), gt_paths.sort()
num = len(tiff_paths)

train["tif"] = img_paths[:int(num*0.8)]
test["tif"] = img_paths[int(num*0.8):]

train["ins"] = ins_paths[:int(num*0.8)]
test["ins"] = ins_paths[int(num*0.8):]

train["gt"] = gt_paths[:int(num*0.8)]
test["gt"] = gt_paths[int(num*0.8):]

write_json(train, os.path.join(output_dir, "train.json"))
write_json(test, os.path.join(output_dir, "test.json"))
