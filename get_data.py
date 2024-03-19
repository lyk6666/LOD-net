import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
from torch.nn.parallel import DataParallel
import numpy as np
import cv2
def get_test_pred_od():
    model=PolypPVT()
    model=DataParallel(model,device_ids=[0,1,2])
    model_pth="/root/share/LiYuKe/Polyp-PVT-main/model_pth/29PolypPVT.pth"

    model.load_state_dict(torch.load(model_pth))
    model.cuda()
    model.eval()
    for _data_name in ['CVC-300','CVC-ClinicDB','Kvasir','CVC-ColonDB','ETIS-LaribPolypDB']:
        data_path='./dataset/TestDataset/{}'.format(_data_name)
        save_path='./pred_od*att_map/PolypPVT/{}/'.format(_data_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root='{}/images/'.format(data_path)
        gt_root='{}/masks/'.format(data_path)
        num1=len(os.listdir(gt_root))
        test_loader=test_dataset(image_root,gt_root,352)
        for i in range(num1):
            image,gt,name=test_loader.load_data()
            gt=np.asarray(gt,np.float32)
            gt/=(gt.max()+1e-8)
            image=image.cuda()
            P1,P2,pred_od=model(image)
            N,C,H,W=pred_od.shape
            P1=F.interpolate(P1,size=(H,W),mode='bilinear',align_corners=False)
            K=F.sigmoid(P1)
            ATT=0.9*(1-torch.cos(2*np.pi*K))+0.1
            sum_od=torch.zeros([N,1,H,W],device=pred_od.device)
            for i in range(N):
                for num in range(8):
                    sum_od[i,:,:,:]=sum_od[i,:,:,:]+pred_od[i,num,:,:].abs()
            x=(sum_od*ATT).abs()
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)   
            res = F.upsample((x>0.1).float(), size=gt.shape, mode='bilinear', align_corners=True)
            res = res.data.cpu().numpy().squeeze()
           # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
        print(_data_name, 'Finish!')

def get_train_pred_od():
    model=PolypPVT()
    model=DataParallel(model,device_ids=[0,1,2])
    model_pth="/root/share/LiYuKe/Polyp-PVT-main/model_pth/PolypPVT/38PolypPVT.pth"
    model.load_state_dict(torch.load(model_pth))
    model.cuda()
    model.eval()
    data_path='./dataset/TrainDataset/images/' 
    gt_path='./dataset/TrainDataset/masks/'  
    save_path='./traindata/pred_od_map/'
    num1=len(os.listdir(data_path))
    print(num1)
    train_loader=test_dataset(data_path,gt_path,352)
    for i in range(num1):
        image,gt,name=train_loader.load_data()
        gt=np.asarray(gt,np.float32)
        gt/=(gt.max()+1e-8)
        image=image.cuda()
        P1,P2,pred_od=model(image)
        N,C,H,W=pred_od.shape
        P1=F.interpolate(P1,size=(H,W),mode='bilinear',align_corners=False)
        K=F.sigmoid(P1)
        ATT=0.9*(1-torch.cos(2*np.pi*K))+0.1
        res = F.upsample(pred_od.float(), size=gt.shape, mode='bilinear', align_corners=True)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res*255)
        print(save_path+name)
get_test_pred_od()