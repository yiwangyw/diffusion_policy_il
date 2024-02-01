import os
import h5py

import numpy as np
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import pandas as pd


# # # 从HDF5文件中读取前10行数据
# # # df = pd.read_hdf(r'./data/mimicgen/datasets/core/square_d0.hdf5', '/data/demo_259/obs/agentview_image')

# # # print(df.shape)



# # f = h5py.File("./data/mimicgen/datasets/core/square_d0.hdf5", "r")
# # ls = list(f.keys())

# # print(ls)

# # G1 = f.get('data')

# # G1_items = list(G1.items())

# # # print(G1_items)

# # sg999 = G1.get('demo_999')
# # sg999_items = list(sg999.items())

# # # print(sg999_items)

# # obs999 = sg999.get('obs')
# # obs999_items = list(obs999.items())

# # print(obs999_items)


# # aimg999 = obs999.get(aimg999)





list_path_file = []
list_path_target = []
file_name = 'eval_log.csv'

    ##get all path of file
for root, dirs, files in os.walk("./data/eval/2024.01.14/", topdown=False):
    for name in files:
        list_path_file.append(os.path.join(root, name))
print(list_path_file)

for root, dirs, files in os.walk("./data/eval/2024.01.15/", topdown=False):
    for name in files:
        list_path_file.append(os.path.join(root, name))
print(list_path_file)

for i in list_path_file:
    if file_name in i:
        list_path_target.append(i)
print("list===", list_path_target)

    

mean_score_list = []

for j in list_path_target:
        # print(j)
    data = np.loadtxt(open(j,"rb"),delimiter=",",skiprows=1,usecols=[-1]) 
    mean_score_list.append(data[0])
    mean_score_list.append(data[1])


mean_scores = np.array(mean_score_list)
mean_scores = mean_scores.flatten()

mean_cv = np.mean(mean_scores)
std_cv = np.std(mean_scores)
cv_ddp = std_cv/mean_cv

print("==== mean==== ",mean_cv, "==== std ==== ", std_cv, "==== cv ==== ", cv_ddp,"=======")



# # Print the keys of groups and datasets under '/'.
# print(f.filename, ":")
# print([key for key in f.keys()], "\n")  

# # Read dataset 'dset' under '/'.
# d = f["agentview_image"]

# # Print the data of 'dset'.
# print(d.name, ":")
# print(d[:])

# # Print the attributes of dataset 'dset'.
# for key in d.attrs.keys():
# 	print(key, ":", d.attrs[key])

# print()

# def traverse_datasets(hdf_file):
#     import h5py
#     def h5py_dataset_iterator(g, prefix=''):
#         # num = 0
#         for key in g.keys():
#             item = g[key]
#             # if item != None:
#             #   num = num+1
#             path = '{}/{}'.format(prefix, key)
#             if isinstance(item, h5py.Dataset): # test for dataset
#                 yield (path, item)
#             elif isinstance(item, h5py.Group): # test for group (go down)
#                 yield from h5py_dataset_iterator(item, path)
#         # print(num)
#     with h5py.File(hdf_file, 'r') as f:
#         for (path, dset) in h5py_dataset_iterator(f):
#             print(path, dset)
#             print()
#     return None



# # import h5py

# # 打开HDF5文件
# file = h5py.File('./data/mimicgen/datasets/core/square_d0.hdf5', 'r')

# # # 列出文件中的所有数据集
# # datasets = list(file.keys())

# # # 打印每个数据集的详细信息
# # num = 0
# # for dataset_name in datasets:
# #     dataset = file[dataset_name]
# #     item = datasetkey
# #     # print("数据集名称:", dataset_name)
# #     # print("数据类型:", dataset.dtype)
# #     # print("维度:", dataset.shape)
# #     if dataset != None:
# #       num = num + 1
# # print(num)
      
# traverse_datasets("./data/robomimic/datasets/square/ph/image_abs.hdf5")
# # # "./data/mimicgen/datasets/core/square_d0.hdf5"
# # # ./data/robomimic/datasets/square/ph/image_abs.hdf5





# #python train.py --config-dir=./configs/image/square_d0_mimicgen/diffusion_policy_cnn --config-name=config.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'