import os
from os.path import join
import re
import sys
from .search_engine.src import unit,dataloader
from .search_engine.mpg.log import info,warn
import json

# for root,dlist,flist in os.walk():

def query_prompt(path_list: list)-> str:
    report_en=json.load(open('./report_en_dict.json'))
    prompt=""
    for i,path in enumerate(path_list):
        ex=f"Example {i}:\n"+report_en[path].replace('Findings:\n','').replace('Impression:\n','')
        prompt+=ex+'\n'
    return prompt

def query_msd(query:str,dataset="mimic_try",k=5,verbose=False):
    q=query.replace('\\n','\n')
    res=unit.query([q],config=dataset,k=k)[0]
    path_list=[]
    for path,dist in res:
        path_list.append(path)
        if verbose:
            print("Path: ",join("search_engine/data/datasets/mimic",os.path.splitext(path)[0]+".txt"))
            # print("Result: ", open(join("data/datasets/mimic",os.path.splitext(path)[0]+".txt")).read())
            print("L2 Distance: ",dist) # 2sin(\theta/2)
            print("Similarity: ",1-dist**2/2) # cos(\theta)=2sin^2(\theta/2)-1
            print()
    return path_list

if __name__=="__main__":
    
    q="as compared to the previous radiograph there is no relevant change . the monitoring and support devices are in constant position . \
            constant appearance of the cardiac silhouette and of the lung parenchyma . no newly appeared parenchymal opacities ."
    # q = "asdasd asd asd asda sda sd. "
    ret_list=query_msd(q,dataset='mimic_try',k=3,verbose=True)
    res=query_prompt(ret_list)
    print(res)
    # print("Input Findings:")
    # while True:
    #     # q=input()
    #     q="as compared to the previous radiograph there is no relevant change . the monitoring and support devices are in constant position . \
    #         constant appearance of the cardiac silhouette and of the lung parenchyma . no newly appeared parenchymal opacities ."
    #     if len(sys.argv)<2:
    #         dataset="mimic_try"
    #     else:
    #         dataset=sys.argv[1]
    #     query_msd(q,dataset=dataset,k=5,verbose=False)

    # print(query_msd("Findings:\nThe cardiac, mediastinal and hilar contours are normal. Pulmonary vasculature is normal.  Lungs are clear. No pleural effusion or pneumothorax is present. Multiple clips are again seen projecting over the left breast.  Remote left-sided rib fractures are also re- demonstrated.\nImpression:\nNo acute cardiopulmonary abnormality."))