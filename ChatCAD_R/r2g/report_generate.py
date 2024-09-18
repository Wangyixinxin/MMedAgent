from .models import BaseCMNModel
from .modules.tokenizers import Tokenizer
from .modules.generator import Generator



def reportGen():
    cfg={
        "visual_extractor":"resnet101",
        "ann_path":r".\r2g\annotation.json",
        "threshold":10,
        "cmm_dim":512,
        "cmm_size":2048,
        'logit_layers':1,
        "d_model":512,
        "d_ff":512,
        "d_vf":2048,
        "num_layers":3,
        "num_heads":8,
        "drop_prob_lm":0.5,
        "dropout":0.1,
        "max_seq_length":100,
        "bos_idx":0,
        "eos_idx":0,
        "pad_idx":0,
        "use_bn":0,
        "n_gpu":1,
        "topk":32,
        "sample_method":"beam_search",
        "sample_n":1,
        "beam_size":3,
        "temperature":1.0,
        "load":r'.\weights\r2gcmn_mimic-cxr.pth',
        "group_size":1,
        "output_logsoftmax":1,
        "decoding_constraint":0, 
        "block_trigrams":1, 
        }
    tokenizer = Tokenizer(cfg)
    model = BaseCMNModel(cfg, tokenizer)
    generator= Generator(cfg, model)
    return generator


