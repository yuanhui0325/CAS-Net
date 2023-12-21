#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import os
import math

def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B , N, S]
    Return:
        new_points:, indexed points data, [B, N, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # torch.manual_seed(1024)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn(nsample, points, new_points):
    """
    :param nsample: knn neighbor number, int
    :param points: all points, [B, N, C]
    :param new_points: query points, [B, S ,C]
    :return: knn idx, [B, S, nsample]
    """
    pairwise_distance = square_distance(new_points, points) # [B, S, N]
    idx = pairwise_distance.topk(k=nsample, dim=-1)[1]  # [B, S, nsample]
    return idx


def sort_points_by_distance(org_points,center_points):
    """
    Input:
        org_points: [B, N, S, C]
    Return:
        sort_points: [B, N, S, C]
    """
    device = org_points.device
    B, N, S, C = org_points.shape
    # sample_points_idx = torch.arange(S, dtype=torch.long).to(device).view(1, 1, S).repeat([B, N, 1])
    center_points = center_points.view(B, N, 1, C).repeat([1, 1, S, 1])
    point_norm = org_points-center_points
    sqrdists = torch.norm(point_norm, p=2, dim=-1)
    sqrdistsclone = sqrdists.clone()
    sort_idx = torch.sort(sqrdistsclone,dim=-1)[1]
    # B_indices = torch.arange(B, dtype=torch.long).to(device).view(B,1,1).repeat(1,N,S)
    # N_indices = torch.arange(N, dtype=torch.long).to(device).view(1,N,1).repeat(B,1,S)
    # sqrdists[B_indices, N_indices, sort_idx]
    B_indices = torch.arange(B, dtype=torch.long).to(device).view(B,1,1).repeat(1,N,S)
    N_indices = torch.arange(N, dtype=torch.long).to(device).view(1,N,1).repeat(B,1,S)
    sort_points = org_points[B_indices,N_indices,sort_idx,:]
    return sort_points


def get_gamma(epoch, epoch_max, gamma_max, gamma_min):
    gamma = gamma_min + (gamma_max - gamma_min)*(1 + math.cos(epoch*math.pi/epoch_max))/2
    return gamma


def seed_torch(seed=2018):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

