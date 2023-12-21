
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
lr=5e-5
epochs=400
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
    shuffle=True,
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
            # ds_point, output, _ = model(points)
            ds_point, output, _,_ = model(points)
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
    model = ds_model_cosloss(hard=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    # model.load_state_dict(torch.load('../save_model/best_512_cas.mdl'))
    for epoch in range(epochs):
        print("epoch:",epoch)

        for step, data in enumerate(dataloader_train, 0):
            _,target,points_gt  = data
            # print(points.dtype)
            points =points_gt.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.from_numpy(points)
            points, target = points.to(device), target.to(device)
            points_input = points.transpose(2, 1)
            optimizer.zero_grad()
            model.train()
            ds_point, output, cos_loss, _ = model(points_input)
            ds_point =ds_point.transpose(2, 1)
            target = np.squeeze(target)
            loss_cls = criteon(output, target)

            loss_ds = get_cd_loss(ds_point, points)
            # print(loss_cls.item(),loss_ds.item())
            loss =loss_cls+loss_ds+cos_loss

            loss.backward()
            optimizer.step()

        if epoch%1 == 0:
            val_acc = evaluate(model, dataloader_test)
            print("epoch:",epoch,"acc:",val_acc)

            if val_acc>best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), '../save_model/best_512_cash.mdl')
    print("best_acc:",best_acc)
