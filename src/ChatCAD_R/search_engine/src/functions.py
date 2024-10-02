import re
import json
def is_chinese(s):
    """
    Returns True if the input string contains only Chinese characters.
    """
    pattern = re.compile(r'[\u4e00-\u9fff]+')  # Match any Chinese character
    return bool(pattern.fullmatch(s))

def unpack_kwargs(kwargs:dict):
    if "config" in kwargs:
        with open("search_engine/data/dataset_conf/"+kwargs["config"]+".json") as f_config:
            conf_args=json.load(f_config)
        tmp=kwargs
        for key,value in kwargs.items():
            conf_args[key]=value
        kwargs=conf_args

    return kwargs