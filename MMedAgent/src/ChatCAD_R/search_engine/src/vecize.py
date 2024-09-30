from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import jieba
from .functions import is_chinese
import pickle
import os

def generate_vec_from_kw(kw_list,vectorizer):
    raise NotImplementedError()
def chi_analyser(x):
    return list(jieba.cut_for_search(x))

def save_vectorizer(vectorizer,**kwargs):
    if "vec_path" in kwargs:
        vec_path=kwargs["vec_path"]
    else:
        vec_path=f"db/{kwargs['name']}_model"
    if not os.path.splitext(vec_path)[1]:
        vec_path+='.pt'
    with open(vec_path,'wb') as f:
        pickle.dump(vectorizer,f)

def load_vectorizer(**kwargs):
    if "vec_path" in kwargs:
        vec_path=kwargs["vec_path"]
    else:
        vec_path=f"search_engine/db/{kwargs['name']}_model"
    if not os.path.splitext(vec_path)[1]:
        vec_path+='.pt'
    with open(vec_path,'rb') as f:
        vectorizer=pickle.load(f)
    return vectorizer


def vectorize(para_list,vectorizer:TfidfVectorizer=None,**kwargs):# -> tuple(sparse_matrix,list[token_name]):
    is_chi=False
    # for para in para_list:
    #     is_chi=is_chinese(para)
    #     break
    if 'lang' in kwargs and kwargs['lang']=='CN':is_chi=True
    if "vocabulary" in kwargs:
        t=[]
        for vi in kwargs["vocabulary"]:
            for s in vi.split():
                t.append(s)
        kwargs["vocabulary"]=list(set(t))
        # print("Vocabulary list detected!")
        # print(t)

    if is_chi:
        if "user_dict" in kwargs:
            # Load user dictionary with jieba
            jieba.load_userdict(kwargs["user_dict"])
        analyzer=chi_analyser
    else:
        analyzer="word"

    if vectorizer is None:
        
        vectorizer = TfidfVectorizer(analyzer=analyzer,vocabulary=kwargs["vocabulary"])

        # Fit and transform the corpus using TfidfVectorizer
        tfidf_matrix = vectorizer.fit_transform(para_list)
        
        # print(para_list[0])
        # # print(analyzer(para_list[0]))
        # print(vectorizer.build_analyzer()(para_list[0]))
        # print(tfidf_matrix.toarray()[0])
    else:
        tfidf_matrix = vectorizer.transform(para_list)
        
        # print(para_list[0])
        # print(vectorizer.build_analyzer()(para_list[0]))
        # print(tfidf_matrix.toarray()[0])
        

    token_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, token_names, vectorizer
    


if __name__=="__main__":
    corpus = [
        "This is the first document",
        "This is the second document",
        "And this is the third one",
        "Is this the first document?"
    ]

    tfidf_matrix, token_names, vectorizer =vectorize(corpus)
    print(token_names)
    for i in tfidf_matrix.toarray():
        print(i)
        
    corpus_cn = [
        "这是第一个文件",
        "这是第二个文件",
        "这是第三个文件",
        "这是第一个文件吗？"
    ]
    tfidf_matrix, token_names, vectorizer=vectorize(corpus_cn)
    print(token_names)
    for i in tfidf_matrix.toarray():
        print(i)

# Create a TfidfVectorizer object
# vectorizer = TfidfVectorizer(vocabulary=)
# vectorizer = TfidfVectorizer()

# # Fit and transform the corpus using TfidfVectorizer
# tfidf_matrix = vectorizer.fit_transform(corpus)

# # Convert the sparse matrix to a dense matrix for easier manipulation
# dense_matrix = tfidf_matrix.toarray()

# # Print the feature names (i.e., unique words) found in the corpus
# print(vectorizer.get_feature_names_out())

# # Print the TF-IDF feature matrix
# print(dense_matrix)

# print(vectorize(corpus))