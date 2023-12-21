import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import query_ball_point, index_points, knn, square_distance


# torch.manual_seed(1024)

class RelationEncoding(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp):
        super(RelationEncoding, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, points):
        """
            relation encoding on original points, no sampling
            :param points: input original points, [B, C, N]
            :return: new points after relation encoding, [B, D, N]
        """
        points = points.permute(0, 2, 1)
        B, N, C = points.shape

        # grouping points
        idx = query_ball_point(self.radius, self.nsample, points, points)  # [B, N, nsample]
        grouped_points = index_points(points, idx)  # [B, N, nsample, C]

        # encoding
        points = points.view(B, N, 1, C).repeat(1, 1, self.nsample, 1)  # [B, N, nsample, C]
        edge_fea = torch.cat([points, grouped_points-points],  dim=-1)  # [B, N, nsample, 2C]
        edge_fea = edge_fea.permute(0, 3, 2, 1)  # [B, 2C, nsample, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            edge_fea = F.leaky_relu(bn(conv(edge_fea)), negative_slope=0.2)
        new_points = torch.max(edge_fea, 2)[0]  # [B, D, N]

        return new_points

class RelationEncoding_SEG(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp):
        super(RelationEncoding_SEG, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, points, points_ft):
        """
            relation encoding on original points, no sampling
            :param points: input original points, [B, C, N]
            :return: new points after relation encoding, [B, D, N]
        """
        points = points.permute(0, 2, 1)
        points_ft = points_ft.permute(0, 2, 1)
        if points_ft is None:
            points = points
            B, N, C = points.shape
        else:
            points = points_ft
            B, N, C = points.shape

        # grouping points
        idx = query_ball_point(self.radius, self.nsample, points, points)  # [B, N, nsample]
        grouped_points = index_points(points, idx)  # [B, N, nsample, C]

        # encoding
        points = points.view(B, N, 1, C).repeat(1, 1, self.nsample, 1)  # [B, N, nsample, C]
        edge_fea = torch.cat([points, grouped_points-points],  dim=-1)  # [B, N, nsample, 2C]
        edge_fea = edge_fea.permute(0, 3, 2, 1)  # [B, 2C, nsample, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            edge_fea = F.leaky_relu(bn(conv(edge_fea)), negative_slope=0.2)
        new_points = torch.max(edge_fea, 2)[0]  # [B, D, N]

        return new_points


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if self.group_all:
            sampling_channel = in_channel // 2
        else:
            sampling_channel = (in_channel - 6) // 2 + 3

        self.SLRS = SLRS(sampling_channel, npoint)


    def forward(self, xyz, points, gamma=1, hard=False):
        """
        set abstraction level
        :param xyz: input points position data, [B, C, N]
        :param points: input points data, [B, D, N]
        :param hard: whether to use straight through, Bool

        :return new_xyz: sampled points position data, [B, C, S]
        :return new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        device = xyz.device
        B, N, C = xyz.shape
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
            _, _, D = points.shape
        else:
            points = xyz
            _, _, D = points.shape

        # grouping module
        if self.radius == None:
            idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, N, 1])
        else:
            idx = query_ball_point(self.radius, self.nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)

        grouped_points_agg = torch.cat((points, xyz), dim=-1)

        select, Cos_loss, unique_num_points = self.SLRS(grouped_points_agg, xyz, hard=hard, gamma=gamma)  # [B, npoint, D], [B, npoint, N]
        sample_xyz = select.matmul(xyz)
        sample_points = select.matmul(points)
        if self.group_all:
            sample_grouped_points = points.view(B, 1, N, D)
            sample_points = sample_points.view(B, 1, 1, D).repeat(1, 1, N, 1)
            sample_grouped_xyz = xyz.view(B, 1, N, C)
            center_xyz = sample_xyz.view(B, 1, 1, C).repeat(1, 1, N, 1)
        else:
            grouped_points = grouped_points.view(B, N, -1)
            sample_grouped_points = select.matmul(grouped_points)
            sample_grouped_points = sample_grouped_points.view(B, self.npoint, self.nsample, D)  # [B, npoint//partition, nsample, D]
            sample_points = sample_points.view(B, self.npoint, 1, D).repeat(1, 1, self.nsample, 1)
            grouped_xyz = grouped_xyz.view(B, N, -1)
            sample_grouped_xyz = select.matmul(grouped_xyz)
            sample_grouped_xyz = sample_grouped_xyz.view(B, self.npoint, self.nsample, C)  # [B, npoint, nsample, C]
            center_xyz = sample_xyz.view(B, self.npoint, 1, C).repeat(1, 1, self.nsample, 1)

        # coding
        sample_grouped_xyz_norm = sample_grouped_xyz - center_xyz
        sample_grouped_points_norm = sample_grouped_points - sample_points
        edge_fea = torch.cat([sample_points, sample_grouped_points_norm, center_xyz, sample_grouped_xyz_norm], dim=-1)
        del  sample_grouped_points_norm, sample_grouped_points, sample_points, sample_grouped_xyz_norm, sample_grouped_xyz, center_xyz
        edge_fea = edge_fea.permute(0, 3, 2, 1)  # [B, 2D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            edge_fea = F.leaky_relu(bn(conv(edge_fea)), negative_slope=0.2)

        new_points = torch.max(edge_fea, 2)[0]

        new_xyz = sample_xyz.permute(0, 2, 1)
        return new_xyz, new_points, Cos_loss, unique_num_points

class SetAbstraction_SEG(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, last_layer):
        super(SetAbstraction_SEG, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        self.last_layer = last_layer

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if self.group_all:
            sampling_channel = in_channel // 2
        else:
            sampling_channel = (in_channel - 6) // 2 + 3

        self.SLRS = SLRS(sampling_channel, npoint, seg=True)


    def forward(self, re_xyzft, xyz, points, gamma=1, hard=False):
        """
        set abstraction level
        :param xyz: input points position data, [B, C, N]
        :param points: input points data, [B, D, N]
        :param hard: whether to use straight through, Bool

        :return new_xyz: sampled points position data, [B, C, S]
        :return new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        device = xyz.device
        B, N, C = xyz.shape
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
            _, _, D = points.shape
        else:
            points = xyz
            _, _, D = points.shape

        # grouping module
        if self.radius == None:
            idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, N, 1])
        else:
            idx = query_ball_point(self.radius, self.nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)

        grouped_points_agg = torch.cat((points, xyz), dim=-1)

        select, Cos_loss, unique_num_points = self.SLRS(grouped_points_agg, xyz, hard=hard, gamma=gamma)  # [B, npoint, D], [B, npoint, N]
        sample_xyz = select.matmul(xyz)
        sample_points = select.matmul(points)
        sample_re_xyzft = select.matmul(re_xyzft)
        sample_points_org = sample_points.clone()

        if self.group_all:
            sample_grouped_points = points.view(B, 1, N, D)
            sample_points = sample_points.view(B, 1, 1, D).repeat(1, 1, N, 1)
            sample_grouped_xyz = xyz.view(B, 1, N, C)
            center_xyz = sample_xyz.view(B, 1, 1, C).repeat(1, 1, N, 1)
        else:
            grouped_points = grouped_points.view(B, N, -1)
            sample_grouped_points = select.matmul(grouped_points)
            sample_grouped_points = sample_grouped_points.view(B, self.npoint, self.nsample, D)  # [B, npoint//partition, nsample, D]
            sample_points = sample_points.view(B, self.npoint, 1, D).repeat(1, 1, self.nsample, 1)
            grouped_xyz = grouped_xyz.view(B, N, -1)
            sample_grouped_xyz = select.matmul(grouped_xyz)
            sample_grouped_xyz = sample_grouped_xyz.view(B, self.npoint, self.nsample, C)  # [B, npoint, nsample, C]
            center_xyz = sample_xyz.view(B, self.npoint, 1, C).repeat(1, 1, self.nsample, 1)

        # coding
        sample_grouped_xyz_norm = sample_grouped_xyz - center_xyz
        sample_grouped_points_norm = sample_grouped_points - sample_points
        edge_fea = torch.cat([sample_points, sample_grouped_points_norm, center_xyz, sample_grouped_xyz_norm], dim=-1)
        del  sample_grouped_points_norm, sample_grouped_points, sample_points, sample_grouped_xyz_norm, sample_grouped_xyz, center_xyz
        edge_fea = edge_fea.permute(0, 3, 2, 1)  # [B, 2D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            edge_fea = F.leaky_relu(bn(conv(edge_fea)), negative_slope=0.2)

        new_points = torch.max(edge_fea, 2)[0]

        new_xyz = sample_xyz.permute(0, 2, 1)
        return new_xyz, new_points, Cos_loss, unique_num_points, sample_points_org, sample_re_xyzft



class SLRS(nn.Module):

    def __init__(self, in_features, select_N,seg=False):
        super(SLRS, self).__init__()
        self.seg = seg
        self.w1 = nn.Conv2d(in_features, in_features, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_features)
        self.w2 = nn.Conv2d(in_features, select_N, 1, 1)
        self.bn2 = nn.BatchNorm2d(select_N)

    def forward(self, points, xyz, hard=False, gamma=1):

        """
        selecting module
        :param x: local region descriptor, [B, N, D]
        :param hard: whether to use straight through
        :return: selecting matrix ret, [B, select_N, N]
                 cosine loss
        """
        device = points.device
        B, N, D = points.shape

        points = points.permute(0, 2, 1).view(B, D, N, 1)
        points = self.bn1(self.w1(points))
        # print("points",points.shape)
        points = self.bn2(self.w2(points))

        select_weights = points.squeeze(-1)
        B, select_N, N = select_weights.shape
        if self.seg:
            # # normal distribution
            normals = torch.randn_like(select_weights)  # ~N(0,1)
            select_weights = (select_weights + 5*normals) / gamma
        else:
            select_weights = select_weights / gamma
        select_weights = select_weights.softmax(dim=-1)

        cos_loss = 0
        if select_N != 1:
            # cosine loss
            inner_product = select_weights.matmul(select_weights.permute(0, 2, 1))
            norm = torch.sqrt(select_weights.mul(select_weights).sum(dim=-1, keepdim=True))
            norm_matrix = norm.matmul(norm.permute(0, 2, 1))
            cosine_matrix = torch.div(inner_product, norm_matrix.add_(1e-10))
            ones = torch.ones([B, select_N, select_N], device=device)
            I = torch.eye(select_N, device=device).view(1, select_N, select_N).repeat(B, 1, 1)
            cosine_matrix_nodiag = cosine_matrix.mul(ones - I)
            cos_loss = torch.sqrt(torch.sum(cosine_matrix_nodiag.mul(cosine_matrix_nodiag), dim=(1, 2))).mean()

        if hard:
            # Straight through.
            index = select_weights.max(dim=-1, keepdim=True)[1]
            # print("index", index.shape)
            select_hard = torch.zeros_like(select_weights).scatter_(-1, index, 1.0)
            ret = select_hard - select_weights.detach() + select_weights
        else:
            ret = select_weights

        index = torch.max(ret, dim=-1)[1]
        unique = torch.unique(index[0, :], return_counts=True)[0]
        # return ret
        return ret, cos_loss, unique.shape


class PointFeatureGlobalPooling(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointFeatureGlobalPooling, self).__init__()
        # self.npoint = npoint
        # self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.activation = nn.LeakyReLU(negative_slope=0.2)



    def forward(self, points,xyz):
        """
        Input:
            xyz:input points xyz, [B, 3, N]
            points: input points data, [B, D, N]
        Return:
            new_points:: output points data, [B, P, 1]
        """
        # points= points.permute(0, 2, 1)  # [B, N, D]
        device = points.device
        B, D, N = points.shape

        mlp_fea =  points.view(B, D, N, 1)
        mlp_xyz = xyz.view(B, 3, N, 1)
        mlp_fea = torch.cat((mlp_fea, mlp_xyz), dim=1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            mlp_fea = self.activation(bn(conv(mlp_fea)))

        new_points = torch.max(mlp_fea, 2)[0]
        return new_points

class SetAbstraction_simp(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(SetAbstraction_simp, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if self.group_all:
            sampling_channel = in_channel // 2
        else:
            sampling_channel = (in_channel - 6) // 2 + 3

        self.SLRS = SLRS(sampling_channel, npoint)


    def forward(self, xyz, points, gamma=1, hard=False):
        """
        set abstraction level
        :param xyz: input points position data, [B, C, N]
        :param points: input points data, [B, D, N]
        :param hard: whether to use straight through, Bool

        :return new_xyz: sampled points position data, [B, C, S]
        :return new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        device = xyz.device
        B, N, C = xyz.shape
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
            _, _, D = points.shape
        else:
            points = xyz
            _, _, D = points.shape

        # grouping module
        if self.radius == None:
            idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, N, 1])
        else:
            idx = query_ball_point(self.radius, self.nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)

        grouped_points_agg = torch.cat((points, xyz), dim=-1)

        select, Cos_loss, unique_num_points = self.SLRS(grouped_points_agg, xyz, hard=hard, gamma=gamma)  # [B, npoint, D], [B, npoint, N]
        sample_xyz = select.matmul(xyz)


        new_xyz = sample_xyz.permute(0, 2, 1)
        return new_xyz, Cos_loss

if __name__ == '__main__':
    input_pc = torch.randn(1,8,3)
    print(input_pc)
    # pc1 = torch.randn(4,3,128)
    model = SLRS(in_features=3, select_N=4)
    output_matric,_,_ = model(input_pc,input_pc,hard=True)
    print(output_matric)#(4,8)
    pc = torch.matmul(output_matric,input_pc)
    print(pc)
    # sa1=SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=64+6, mlp=[64, 64], group_all=False)

    # re = RelationEncoding(radius=0.2, nsample=32, in_channel=6, mlp=[32, 32])
    # out = re(pc1)
    # print(out.shape)
