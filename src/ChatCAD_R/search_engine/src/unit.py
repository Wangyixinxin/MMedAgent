from .vecize import vectorize,save_vectorizer,load_vectorizer,generate_vec_from_kw
from .db import save_db,load_db
from .dataloader import get_flist
from .functions import unpack_kwargs
import json

def build(**kwargs):

    kwargs=unpack_kwargs(kwargs)
    # path_list,para_list=[],[]

    flist=get_flist(kwargs["data_path"])
    path_list=[]
    para_list=[]
    for path,para in flist:
        path_list.append(path)
        para_list.append(para)
    # flist=list(flist)
    # token_list=tokenize(para_list,**kwargs)
    tfidf_matrix, token_names, vectorizer=vectorize(para_list,**kwargs)

    # print(tfidf_matrix.toarray()[0])

    save_vectorizer(vectorizer,**kwargs)
    save_db(path_list=path_list,token_names=token_names,data=tfidf_matrix,**kwargs)

    # import numpy as np
    # print(np.sum(tfidf_matrix.toarray()[:,-5]))
    # print(token_names)

def query(q_list=None,**kwargs):

    kwargs=unpack_kwargs(kwargs)
    # print(kwargs)
    if "k" in kwargs:
        k=kwargs["k"]
    else:
        k=5
    db=load_db(**kwargs)
    # flist=list(flist)
    # token_list=tokenize(para_list,**kwargs)
    if q_list is None:
        flist=get_flist(kwargs["test_path"])
        path_list=[]
        para_list=[]
        for path,para in flist:
            path_list.append(path)
            para_list.append(para)
        q_list=para_list
        
    vectorizer=load_vectorizer(**kwargs)
    tfidf_matrix, _, _=vectorize(q_list,vectorizer,**kwargs)

    # print(tfidf_matrix.toarray()[0])

    req_list=[]
    for q in tfidf_matrix.toarray():
        req_list.append(db.query(q,k=k))
    return req_list

def query_kw(kw_list,**kwargs):
    kwargs=unpack_kwargs(kwargs)
    if "k" in kwargs:
        k=kwargs["k"]
    else:
        k=5
    db=load_db()
    vectorizer=load_vectorizer(**kwargs)
    tfidf_matrix=generate_vec_from_kw(kw_list,vectorizer)
    req_list=[]
    for q in tfidf_matrix.toarray():
        req_list.append(db.query(q,k=k))
    return req_list


# def add(fpath):
#     keywords,feature=get_feature(fpath,feature=tf)