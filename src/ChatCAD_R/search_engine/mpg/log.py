import os,sys,time
from datetime import datetime
import pandas as pd
import shutil
import logging
import subprocess
from os.path import join

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',level=logging.WARNING)
work_path="."
runner_id="{:%Y%m%d-%H%M%S}".format(datetime.now())

from functools import wraps

def info(msg):
    logging.log(logging.INFO,msg)

def warn(msg):
    logging.log(logging.WARNING,msg)

def cp(src,dst):
    info(f"copy {src} -> {dst}")
    try:
        assert(os.path.exists(src))
        if os.path.isfile(src):
            shutil.copy(src,dst)
        else:
            shutil.copytree(src,dst,dirs_exist_ok=True)
    except Exception as e:
        warn(f"Error occur on copying {src} -> {dst}!!\n{e}")
        raise e
        
def rm(dst):
    info(f"remove {dst}")
    try:
        if os.path.isfile(dst):
            os.remove(dst)
        else:
            shutil.rmtree(dst)
    except Exception as e:
        warn(f"Error occur on removing {dst}!!\n{e}")
        raise e

def run_sh(cmd,name="unknown",base_dir="/tmp",pname="unknown"):
    outputs=[]
    logging.info(cmd)
    ret=subprocess.run(f"cd {base_dir} && {cmd} 2>&1",shell=True,stdout=subprocess.PIPE,encoding="utf",executable="bash")
    
    if ret.stdout is not None:
        outputs.append(ret.stdout+'\n')
        logging.info(ret.stdout)
    if ret.stderr is not None:
        outputs.append(ret.stderr+'\n')
        logging.info(ret.stderr)
    if ret.returncode!=0:
        logging.error(f"{pname}.{name}: {cmd.split()[0]} Failed!")
        raise Exception(f"{pname}.{name} Error!")
    logging.info(f"{pname}.{name}: {cmd.split()[0]} Successfully!")
    
    return outputs

def save_log(log_df):
    os.makedirs(join(work_path,"logs"),exist_ok=True)
    log_name=runner_id
    pd.DataFrame(log_df).to_csv(join(work_path,"logs",log_name+".csv"))
