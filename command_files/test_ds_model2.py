import torch,os
from torch import nn,optim
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from command_files.dataloader_ply import Get_data
# from models.sp_cls_model import *
from models.ds_model import ds_model3,ds_model_cosloss
import provider
from settings.chamfer_distance.chamfer_distance import get_cd_loss
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batchsize=12
datapath_train = r'../data/output1024/train'
dataset_train = Get_data(datapath_train)
dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4
)

datapath_test = r'../data/output1024/test'
dataset_test = Get_data(datapath_test)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batchsize,
    shuffle=False,
    num_workers=4
)
device=torch.device('cuda')

def evaluate(model,loader):
    model.eval()
    correct=0
    total=len(loader.dataset)
    # print(total)
    for step, data in enumerate(loader, 0):
        _, target, points = data
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        with torch.no_grad():
            ds_point, output, _,_  = model(points)
            predict=output.argmax(dim=1)
            target = np.squeeze(target)
            # print(predict.shape)
            # print(target.shape)
            # print('predict:',predict)
            # print('target:',target)
        # print(step)

        correct += torch.eq(predict, target).sum().float().item()
    print('correct:',correct,'total:',total)
    return correct/total

if __name__ == '__main__':
    ##   与 ds_model3()
    save_dir = '../save_model/best_512_cash_8962.mdl'
    model = ds_model_cosloss(hard=True).to(device)
    model.load_state_dict(torch.load(save_dir))
    val_acc = evaluate(model, dataloader_test)
    print(val_acc)