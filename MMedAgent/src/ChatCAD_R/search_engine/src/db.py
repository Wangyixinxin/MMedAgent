import os
import pickle

def save_db_kdtree(path_list,token_names,data,**kwargs):
    from scipy.spatial import KDTree

    if "db_path" in kwargs:
        db_path=kwargs["db_path"]
    else:
        db_path=f"search_engine/db/{kwargs['name']}"
    if not os.path.splitext(db_path)[1]:
        db_path+='.pt'

    # Create a KDTree object
    data=data.toarray()
    for v in data:
        if sum(v)!=0: v/=sum(v)
    tree = KDTree(data,copy_data=True)
    # Find the 5 nearest neighbors of the first point
    # distances, indices = tree.query(data[0], k=5)
    # print(path_list)
    db={"path_list":path_list,"token_names":token_names,"tree":tree}
    with open(db_path,'wb') as f:
        pickle.dump(db,f)

def load_db_kdtree(**kwargs):
    from scipy.spatial import KDTree

    if "db_path" in kwargs:
        db_path=kwargs["db_path"]
    else:
        db_path=f"search_engine/db/{kwargs['name']}"
    if not os.path.splitext(db_path)[1]:
        db_path+='.pt'

    # Create a KDTree object
    
    with open(db_path,'rb') as f:
        db = pickle.load(f)
    
    path_list,token_names,tree=db["path_list"],db["token_names"],db["tree"]
    # print(path_list)
    return Query_kdtree(path_list,token_names,tree)
    # Find the 5 nearest neighbors of the first point
    # distances, indices = tree.query(data[0], k=5)
    # db={"path_list":path_list,"token_names":token_names,"tree":tree}
    
    
class Query_kdtree:
    def __init__(self,path_list,token_names,tree) -> None:
        self.path_list,self.token_names,self.tree=path_list,token_names,tree
    def query(self,feature_vector,k=5):
        if sum(feature_vector)!=0: feature_vector/=sum(feature_vector)
        distances, indices=self.tree.query(feature_vector,k,workers=-1)
        # print(len(self.path_list))
        return [(self.path_list[pid],distances[i]) for i,pid in enumerate(indices)]

save_db=save_db_kdtree
load_db=load_db_kdtree

#test pass
if __name__=="__main__":
    
    import scipy.sparse as sp
    # sparse_matrix = sp.lil_matrix((3, 3))
    save_db_kdtree([114,203],['a','b'],sp.csr_matrix([[.5,.5],[-.5,.5]]),name="try_1")
    q=load_db_kdtree(name="try_1")
    print(q.query([1,1],k=2)) # [(114, 0.7071067811865476), (203, 1.5811388300841898)]
