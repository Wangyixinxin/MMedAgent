import json
import numpy as np
from torch import Tensor
import torch

class prob2text:
    def __init__(self, prob:Tensor, disease:dict):
        self.prob = prob.squeeze().detach().cpu().numpy()
        self.disease=disease
        self.levelprompt={
            "0":"No sign of",
            "1":"Small possibility of",
            "2":"Patient is likely to have",
            "3":"Definitely have",
        }
        self.grade=self.grading()

    def grading(self):
        level = {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
            }
        for k in self.disease.keys():
            score=self.prob[self.disease[k]]
            if score>=0 and score<0.2:
                level['0'].append(k)
            elif score>=0.2 and score<0.5:
                level['1'].append(k)
            elif score>=0.5 and score<0.9:
                level['2'].append(k)
            elif score>=0.9:
                level['3'].append(k)
        return level

    def retrievaldisease(self, namelist: list):
        diseases = ''
        for j in namelist:
            diseases += j
            diseases += ', '
        diseases = diseases[:-2]
        return diseases
    
    def promptA(self):
        promptHead="Higher disease score means higher possibility of illness.\n"
        promptA = promptHead+"Network A: "
        for k in self.disease.keys():
            promptA+=f"{k} score: {self.prob[self.disease[k]]:.3f}, "
        promptA=promptA[:-2]+'.'
        return promptA

    def promptB(self):
        level=self.grade
        # level=self.grading(self.logits)
        promptHead='Network A:\n'
        promptB=promptHead
        for l in level.keys():
            if len(level[l])==0:
                continue
            diseases = self.retrievaldisease(level[l])
            promptB += f"{self.levelprompt[l]} {diseases}."
        return promptB

    def promptC(self):
        promptHead="Network A's diagnosis prediction is"
        promptC=promptHead
        noFinding=True
        for d in self.disease.keys():
            if self.prob[self.disease[d]]>=0.5:
                noFinding=False
                promptC+=f" {d},"
        if noFinding:
            promptC+="No Finding."
        else:
            promptC=promptC[:-1]+'.'
        return promptC

class prob2text_zh:
    def __init__(self, prob:Tensor, disease:dict):
        self.prob = prob.squeeze().detach().cpu().numpy()
        self.disease=disease
        self.levelprompt={
            "0":"未发现",
            "1":"较低概率患有",
            "2":"疑似患有",
            "3":"明确发现",
        }
        self.grade=self.grading()

    def grading(self):
        level = {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
            }
        for k in self.disease.keys():
            score=self.prob[self.disease[k]]
            if score>=0 and score<0.2:
                level['0'].append(k)
            elif score>=0.2 and score<0.5:
                level['1'].append(k)
            elif score>=0.5 and score<0.9:
                level['2'].append(k)
            elif score>=0.9:
                level['3'].append(k)
        return level

    def retrievaldisease(self, namelist: list):
        diseases = ''
        for j in namelist:
            diseases += j
            diseases += ', '
        diseases = diseases[:-2]
        return diseases
    
    def promptA(self):
        promptHead="越高的分数代表患病的可能性越高\n"
        promptA = promptHead+"网络A："
        for k in self.disease.keys():
            promptA+=f"{k}（{self.prob[self.disease[k]]:.3f}），"
        promptA=promptA[:-1]+'\n'
        return promptA

    def promptB(self):
        level=self.grade
        # level=self.grading(self.logits)
        promptHead='网络A：\n'
        promptB=promptHead
        for l in level.keys():
            if len(level[l])==0:
                continue
            diseases = self.retrievaldisease(level[l])
            promptB += f"{self.levelprompt[l]}{diseases}\n"
        return promptB

    def promptC(self):
        promptHead="网络A的诊断结果为："
        promptC=promptHead
        noFinding=True
        for d in self.disease.keys():
            if self.prob[self.disease[d]]>=0.5:
                noFinding=False
                promptC+=f"{d}，"
        if noFinding:
            promptC+="未见异常。"
        else:
            promptC=promptC[:-1]+'\n'
        return promptC



if __name__=="__main__":
    disease={
        "Atelectasis":0,
        "Cardiomegaly":1,
        "Pleural_Effusion":2,
        "Infiltration":3,
        "Mass":4,
        "Nodule":5,
        "Pneumonia":6,
        "Pneumothorax":7,
        "Consolidation":8,
        "Edema":9,
        "Emphysema":10,
        "Fibrosis":11,
        "Pleural_Thickening":12,
        "Hernia":13,
        }
    fivedisease_zh={
    "心脏肥大":0,
    "肺水肿":1,
    "肺实变":2,
    "肺不张":3,
    "胸腔积液":4,
        }
    disease_zh={
        "肺不张":0,
        "心脏肥大":1,
        "胸腔积液":2,
        "浸润":3,
        "纵隔肿块 ":4,
        "肺结节":5,
        "肺炎":6,
        "气胸":7,
        "肺实变":8,
        "肺水肿":9,
        "肺气肿":10, # 慢性阻塞性肺病
        "胸膜纤维化":11,
        "胸膜增厚":12,
        "膈疝":13,
        }

    prob=torch.sigmoid(torch.randn(1,5))
    converter=prob2text_zh(prob,fivedisease_zh)
    res=converter.promptA()
    # print(converter.promptA())
    # print("\n")
    # print(converter.promptB())
    # print("\n")
    # print(converter.promptC())
    # print("\n")
    # print("hold")