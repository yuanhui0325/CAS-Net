# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:26 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
# @File        : point_operation.py
import torch
import numpy as np

def nonuniform_sampling(num = 4096, sample_num = 1024):
    sample = set()
    loc = np.random.rand()*0.8+0.1
    while(len(sample)<sample_num):
        a = int(np.random.normal(loc=loc,scale=0.3)*num)
        if a<0 or a>=num:
            continue
        sample.add(a)
    return list(sample)

def shuffle_point_cloud_and_gt(batch_data,batch_gt=None):
    B,N,C = batch_data.shape

    idx = np.arange(N)
    np.random.shuffle(idx)
    batch_data = batch_data[:,idx,:]
    if batch_gt is not None:
        np.random.shuffle(idx)
        batch_gt = batch_gt[:,idx,:]
    return batch_data,batch_gt


def rotate_point_cloud_and_gt(batch_data,batch_gt=None,z_rotated=True):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=(3)) * 2 * np.pi # 随机生成旋转角度
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        if z_rotated:
            rotation_matrix = Rz  # 只在 z轴上旋转一定角度
        else:
            rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))   # 3个轴都旋转一定角度， np.dot(A,B) : 一般的矩阵乘法 A * B

        # rotation_angle = np.random.uniform(size=(3)) * 2 * np.pi
        # cosval = np.cos(rotation_angle)
        # sinval = np.sin(rotation_angle)
        # rotation_matrix = np.array([[cosval, 0, sinval],
        #                             [0, 1, 0],
        #                             [-sinval, 0, cosval]])

        batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)  # 对 input 进行旋转
        if batch_data.shape[-1]>3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

        if batch_gt is not None:
            batch_gt[k, ..., 0:3]   = np.dot(batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix) # 对 gurond truth 进行旋转
            if batch_gt.shape[-1] > 3:
                batch_gt[k, ..., 3:] = np.dot(batch_gt[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

    return batch_data,batch_gt   # 对 batch_data (input) 和 batch_gt (ground truth) 进行一抹一样的空间旋转操作

def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)    # numpy.clip(a, a_min, a_max, out=None) 让数据在一个很小的范围内抖动
    jittered_data[:,:,3:] = 0  # 几何以外的特征清零，因为只抖动几何信息
    jittered_data += batch_data  # 增加 抖动值
    return jittered_data

def shift_point_cloud_and_gt(batch_data, batch_gt = None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] += shifts[batch_index,0:3]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] += shifts[batch_index, 0:3]

    return batch_data,batch_gt


def random_scale_point_cloud_and_gt(batch_data, batch_gt = None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)  # numpy.random.uniform(low,high,size) 从均匀分布中 生成值 在区间 [low,high) 的数值，size 为生成数值的 size
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] *= scales[batch_index]  # 对 batch 内的每个 样本 放大或缩小 scale 倍

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] *= scales[batch_index]

    return batch_data,batch_gt,scales


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        batch_data[k, ...,0:3] = np.dot(batch_data[k, ...,0:3].reshape((-1, 3)), R)
        if batch_data.shape[-1]>3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), R)

    return batch_data

def guass_noise_point_cloud(batch_data, sigma=0.005, mu=0.00):
    """ Add guassian noise in per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    batch_data += np.random.normal(mu, sigma, batch_data.shape)
    return batch_data

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    point = point.squeeze()
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    # print(point[[51])#
    point = point[centroids.astype(np.int32)]
    return centroids.astype(np.int32),point

def farthest_point_sample_new(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

if __name__ == '__main__':
    pc =torch.randn(3,1024,3)
    idx =farthest_point_sample_new(pc,256)