Yes, Thank you for your response.
Yes I'm working on a classification problem. Although I understand what you have said here, this is 
my first time working with point cloud data and h5 files. I just have a follow up question just to 
know if I've understood this correctly

I have downloaded the ModelNet40 dataset(since the paper uses the same) and just to better understand 
the pre-processing step(writing data to the h5 file), I tried to create h5 file for the dataset.

Do we need to follow the following steps:

1.mesh sampling to overly sample more than 10k points
2.Use farthest point sampling to get 2048 points
3.Write these 2048 point to the h5 file
4.Use the generated h5 files for training the model
Is that correct? Please let me know.

有关数据处理：https://github.com/charlesq34/pointnet/issues/75