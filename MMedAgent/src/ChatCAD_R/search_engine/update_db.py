import os
from os.path import join
import sys
from src import unit,dataloader
from mpg.log import info,warn

if __name__=="__main__":
    # for path in glob.glob(sys.argv[0]+"/**/"):
    
    unit.build(config=sys.argv[1])
    print(sys.argv[1]+" training done.")
    # for txt_path, fi in flist:
    #     try:
    #         res_list=unit.build(fi)
    #         info(f"{txt_path} query result:")
    #         for res in res_list:
    #             info(res)
    #     except:
    #         warn(f"{txt_path} query failed!")
