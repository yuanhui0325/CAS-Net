import torch
import os,glob
import random,csv
from plyfile import PlyData, PlyElement
import numpy as np
from torch.utils.data import Dataset
from settings import point_operation

def load_ply(file_name, with_faces=False, with_color=False):
    ply_data = PlyData.read(file_name)
    points = ply_data["vertex"]
    points = np.vstack([points["x"], points["y"], points["z"]]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data["face"]["vertex_indices"])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data["vertex"]["red"])
        g = np.vstack(ply_data["vertex"]["green"])
        b = np.vstack(ply_data["vertex"]["blue"])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val

# def pc_normalize(pc):
#     centroid = np.mean(pc[:, 0:3], axis=0, keepdims=True)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / m
#     return pc


def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0) # 求取点云的中心
    pc = pc - centroid # 将点云中心置于原点 (0, 0, 0)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1))) # 求取长轴的的长度
    pc_normalized = pc / m # 依据长轴将点云归一化到 (-1, 1)
    return pc_normalized, centroid, m  # centroid: 点云中心, m: 长轴长度, centroid和m可用于keypoints的计算



# def pc_normalize(pc):
#     centroid = np.mean(pc[:, :, 0:3], axis=1, keepdims=True)
#     pc[:, :, 0:3] = pc[:, :, 0:3] - centroid
#     furthest_distance = np.amax(np.sqrt(np.sum(pc[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
#     pc[:, :, 0:3] = pc[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
#     # input[:, :, 0:3] = input[:, :, 0:3] - centroid
#     # input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
#     return pc

class Get_data(Dataset):
    def __init__(self,root):
        self.root=root
        self.name2label = {}

        # self.mode=mode
        label=[]#存放标签内容
        for name in sorted(os.listdir(os.path.join(root))):#os.listdir(）：返回输入路径下的文件和列表名称,os.path.join():拼接待操作对象
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())#class list
            label.append(name)
        # print(self.name2label)
        # print(label)
        self.num_class=len(self.name2label.keys())

        # print(self.name2label)
        self.pointcloud_file = self.load_data()
        # print(len(self.pointcloud_file))

    def load_data(self):

        pointcloud = []
        for name in self.name2label.keys():
            # ModelNet10//bathtub//0001
            pointcloud += glob.glob(os.path.join(self.root, name, '*.ply'))  # glob匹配所有符合的文件返回list
        # print(pointcloud)
        # random.shuffle(pointcloud)
        # print(pointcloud)
        return pointcloud
    def __len__(self):
        return len(self.pointcloud_file)
    def __getitem__(self, index):
        one_file = self.pointcloud_file[index]#bathtub_0001.txt
        # print(one_file)

        name = one_file.split('/')[-2]
        label = self.name2label[name]
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        point_set = load_ply(one_file)
        # choice = np.random.choice(len(points), 2048, replace=False)
        # point_set = points[choice, :]

        _, point_set = point_operation.farthest_point_sample(point_set, 512)

        # print(point_set.shape)
        point_set,_,_ = normalize_point_cloud(point_set)
        point_set = torch.from_numpy(point_set.astype(np.float32))


        # pointcloud_data=np.loadtxt(os.path.join(one_file), delimiter = ',', dtype = float)[:,:3]
        return one_file,label,point_set



if __name__ == '__main__':
    datapath_test = r'../data/output2048/test'#
    # datapath_train = r'../data/Modelnet40ply/train'
    datapath_train = r'../data/output2048/test'
    datapath_test = r'../data/model40new/test'
    dataset_train=Get_data(datapath_train)
    dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=5,
        shuffle=True,
        num_workers=4
    )
    for step, data in enumerate(dataloader, 0):
        name,label,point=data
        # print(point.dtype)
        print(point.shape)
        # print(label)
        # print(step)
