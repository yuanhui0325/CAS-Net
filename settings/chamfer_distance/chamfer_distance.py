import torch
from torch.utils.cpp_extension import load
import os

script_dir = os.path.dirname(__file__)
sources = [
    os.path.join(script_dir, "chamfer_distance.cpp"),
    os.path.join(script_dir, "chamfer_distance.cu"),
]

cd = load(name="cd", sources=sources)
# wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
#     sudo unzip ninja-linux.zip -d /usr/local/bin/
#     sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

CUDA_NUM = 0


class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda(CUDA_NUM)
            dist2 = dist2.cuda(CUDA_NUM)
            idx1 = idx1.cuda(CUDA_NUM)
            idx2 = idx2.cuda(CUDA_NUM)
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(
                xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            )
        else:
            gradxyz1 = gradxyz1.cuda(CUDA_NUM)
            gradxyz2 = gradxyz2.cuda(CUDA_NUM)
            cd.backward_cuda(
                xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            )

        return gradxyz1, gradxyz2

chamfer_distance = ChamferDistanceFunction.apply


def get_cd_loss(pred, gt):
    cost_for, cost_bac = chamfer_distance(gt, pred)
    # print(cost_for.shape)
    cost_for =torch.mean(cost_for)
    # print(cost_for)
    cost_bac = torch.mean(cost_bac)
    # print(cost_for,cost_bac)
    cost =  1 *  cost_for + 1 * cost_bac
    # cost /= pcd_radius
    # cost = torch.mean(cost)
    return cost

# def get_cd_loss(pred, gt,rad=1):
#     cost_for, cost_bac = chamfer_distance(gt, pred)
#     # print(cost_for.shape)
#     cost_for =torch.mean(cost_for,dim=1)
#     cost_bac = torch.mean(cost_bac,dim=1)
#     cost = 0.8 * cost_for + 0.2 * cost_bac
#     # cost /= pcd_radius
#     # cost = torch.mean(cost)
#     return cost


if __name__ == '__main__':
    # a=torch.randn(4,3,10)
    # b = torch.randn(4, 3, 6)
    a =torch.tensor([[[1,0,0],[2,0,0],[3,0,0],[4,0,0]]]).float()
    b = torch.tensor([[[0, 0, 0], [1, 0, 0],
[3, 0, 0], [4, 0, 1]]]).float()
    # a=a.transpose(2,1)
    # b = b.transpose(2, 1)
    print(a)
    # rad = torch.randn(4, 1)
    loss = get_cd_loss(a,b)
    print(loss)