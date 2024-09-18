from torchvision import transforms
from PIL import Image
import torch
from .models.classifier import Classifier
from easydict import EasyDict as edict
import json
import numpy as np
import torch.nn.functional as F
from .cxr_utils import transform


def getImg(imgPath: str,idx: int=None):
    # idx should be 1 or 2
    img= Image.open(imgPath).convert('RGB')
    reportTrans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    imgTrans=transforms.Compose([
        transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    img1,img2=reportTrans(img),imgTrans(img)
    res=[img1.unsqueeze(dim=0), img2.unsqueeze(dim=0)]
    # img0 for text generation, img1 for inference
    if idx is not None:
        return res[idx-1]
    return res

def getJFImg(imgPath: str,imgcfg,idx: int=None):
    # idx should be 1 or 2
    img1= Image.open(imgPath).convert("RGB")
    img2= Image.open(imgPath)
    reportTrans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    img1=reportTrans(img1)
    img2 = np.array(img2)
    img2 = transform(img2,imgcfg)
    img2=torch.tensor(img2,dtype=torch.float32)
    res=[img1.unsqueeze(dim=0), img2.unsqueeze(dim=0)]
    # img0 for text generation, img1 for inference
    if idx is not None:
        return res[idx-1]
    return res

def get_pred(output, cfg):
    if cfg.criterion == 'BCE' or cfg.criterion == "FL":
        # for num_class in cfg.num_classes:
        #     assert num_class == 1
        # pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
        pred = torch.sigmoid(output.view(-1))
    elif cfg.criterion == 'CE':
        for num_class in cfg.num_classes:
            assert num_class >= 2
        prob = F.softmax(output)
        # pred = prob[:, 1].cpu().detach().numpy()
        pred = prob[:, 1]
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))
    return pred

def JFinfer(img_model, img, imgcfg):
    img_model.eval()
    outs_classes, _ = img_model(img)
    assert isinstance(outs_classes,list)
    # outs_classes= model.infer(img)
    prob = np.zeros((len(imgcfg.Data_CLASSES), 1))
    prob= get_pred(torch.Tensor(outs_classes), imgcfg)
    return prob

def JFinit(cfg_path,weight_path):
    imgcfg = edict(json.load(open(cfg_path)))
    img_model = Classifier(imgcfg)
    # model.to(torch.cuda())
    if torch.cuda.is_available():
        img_model.load_state_dict(torch.load(weight_path))
    else:
        img_model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    return img_model, imgcfg