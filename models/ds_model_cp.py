import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.slrsa_util import  RelationEncoding, SetAbstraction
from settings.transformer_module import *
from models import pointnet_cls



class ds_model_cosloss(nn.Module):
    def __init__(self,ds_num =512,hard=False):
        super(ds_model_cosloss, self).__init__()
        self.hard =hard
        self.ds_num =ds_num
        self.re = RelationEncoding(radius=0.2, nsample=32, in_channel=6, mlp=[64, 64])
        self.sa1 = SetAbstraction(npoint=ds_num, radius=0.02, nsample=32, in_channel=192*2+6, mlp=[256, 256], group_all=False)
        self.oa1 = OA(channels=64)
        self.oa2 = OA(channels=64)
        self.oa3 = OA(channels=64)
        # self.oa4 = OA(channels=64)
        self.cls_net= pointnet_cls.get_model(k=40)


    def forward(self, xyz, gamma=1, hard=False):
        """
            classification task
            :param xyz: input points, [B, C ,N]
            :param hard: whether to use straight-through, Bool
            :return: prediction, [B, 40]
        """
        device = xyz.device
        B, _, _ = xyz.shape

        #   Relation Encoding Layer
        points = self.re(xyz)

        fea1 = self.oa1(points)
        fea2 = self.oa2(fea1)
        fea3 = self.oa3(fea2)
        # fea4 = self.oa4(fea3)
        fea =torch.cat([fea1,fea2,fea3],dim=-2)#points,

        # print(fea.shape)

        # Set Abstraction Levels
        l1_xyz, l1_points, Cos_loss1, unique_num_points1 = self.sa1(xyz, fea, gamma=gamma, hard=self.hard)
        # points = self.oa1(points)
        cls,_ = self.cls_net(l1_xyz)
        return l1_xyz,cls,Cos_loss1,l1_points


if __name__ == '__main__':
    # in_put = torch.randn(2,3,2048)
    # # in_put = torch.randn(2, 2048, 6)
    # model = ds_model3()
    #
    #
    # # ds1_point,ds2_point,output = model(in_put)
    # ds1_point,output,_ = model(in_put)
    # # print(output.shape)
    #
    # # model=PointTransformerLayer(6,64)
    # # output =model(in_put)
    # print(output.shape)

    in_put = torch.randn(2,3,2048)
    # in_put = torch.randn(2, 2048, 6)
    model = ds_model_cosloss(hard=True)


    # ds1_point,ds2_point,output = model(in_put)
    ds1_point,output,cosine_loss,l1 = model(in_put)
    # print(output.shape)

    # model=PointTransformerLayer(6,64)
    # output =model(in_put)
    print(ds1_point.shape)