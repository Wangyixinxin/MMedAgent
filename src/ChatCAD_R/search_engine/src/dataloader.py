import json

def get_flist(path):
    # path=kwargs["data_path"]
    with open("search_engine/data/datasets/"+path) as f:
        rplist=json.load(f)
        for rp in rplist:
            txt_path, report=rp["txt_path"],rp["report"]
            report.replace('\\n','\n')
            yield txt_path, report

if __name__=="__main__":
    print(__path__)
    j=0
    for i in get_flist(data_path="search_engine/data/datasets/mimic.json"):
        # print(i)
        j+=1
        # if j==10: break