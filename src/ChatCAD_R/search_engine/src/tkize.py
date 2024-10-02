import jieba
def tokenize(para_list,**kwargs):
    if "user_dict" in kwargs:
        # Load user dictionary with jieba
        jieba.load_userdict(kwargs["user_dict"])
        
    para=[jieba.cut_for_search(para) for para in para_list]
    
    # raise Exception("NotImplemented")

    return para

# def tokenize(para,**kwargs):
       
#     para=(jieba.cut_for_search(para) for para in para_list)
    
#     raise Exception("NotImplemented")

#     return para

if __name__=='__main__':
    for i in tokenize(['''结果：
肺量较低。心脏大小正常。纵隔和肺门形态无异常。在左上肺叶及较少见的右上肺叶可见新的结节状不透明影。未见气胸或左侧胸膜积液。肺血管量在正常范围内。右侧胸部有手术改变，包括右6肋部分切除，右侧胸膜增厚和慢性折叠隐窝模糊。
诊断：
左右上肺叶新的结节状不透影，左较右为多。结果与同一天晚些时候进行的腹部和盆腔CT检查中肺底的转移相符。''']):
        print(list(i))