# Block 1
import numpy as np
import torch
import pickle
import time

import matplotlib.pyplot as plt
import scipy.sparse






# Block 2
def schuffle(W,c):
    # relabel the vertices at random
    idx=np.random.permutation( W.shape[0] )
    #idx2=np.argsort(idx) # for index ordering wrt classes
    W_new=W[idx,:]
    W_new=W_new[:,idx]
    c_new=c[idx]
    return W_new , c_new , idx 


def block_model(c,p,q):
    n=len(c)
    W=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if c[i]==c[j]:
                prob=p
            else:
                prob=q
            if np.random.binomial(1,prob)==1:
                W[i,j]=1
                W[j,i]=1     
    return W


def unbalanced_block_model(nb_of_clust, clust_size_min, clust_size_max, p, q):  
    c = []
    for r in range(nb_of_clust):
        if clust_size_max==clust_size_min:
            clust_size_r = clust_size_max
        else:
            clust_size_r = np.random.randint(clust_size_min,clust_size_max,size=1)[0]
        val_r = np.repeat(r,clust_size_r,axis=0)
        c.append(val_r)
    c = np.concatenate(c)  
    W = block_model(c,p,q)  
    return W,c


def random_pattern(n,p):
    W=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if np.random.binomial(1,p)==1:
                W[i,j]=1
                W[j,i]=1     
    return W    


    
def add_pattern(W0,W,c,nb_of_clust,q):
    n=W.shape[0]
    n0=W0.shape[0]
    V=(np.random.rand(n0,n) < q).astype(float)
    W_up=np.concatenate(  ( W , V.T ) , axis=1 )
    W_low=np.concatenate( ( V , W0  ) , axis=1 )
    W_new=np.concatenate( (W_up,W_low)  , axis=0)
    c0=np.full(n0,nb_of_clust)
    c_new=np.concatenate( (c, c0),axis=0)
    return W_new,c_new


class generate_SBM_graph():

    def __init__(self, SBM_parameters): 

        # parameters
        nb_of_clust = SBM_parameters['nb_clusters']
        clust_size_min = SBM_parameters['size_min']
        clust_size_max = SBM_parameters['size_max']
        p = SBM_parameters['p']
        q = SBM_parameters['q']
        p_pattern = SBM_parameters['p_pattern']
        q_pattern = SBM_parameters['q_pattern']
        vocab_size = SBM_parameters['vocab_size']
        W0 = SBM_parameters['W0']
        u0 = SBM_parameters['u0']

        # block model
        W, c = unbalanced_block_model(nb_of_clust, clust_size_min, clust_size_max, p, q)
        
        # signal on block model
        u = np.random.randint(vocab_size, size=W.shape[0])
        
        # add the subgraph to be detected
        W, c = add_pattern(W0,W,c,nb_of_clust,q_pattern)
        u = np.concatenate((u,u0),axis=0)
        
        # shuffle
        W, c, idx = schuffle(W,c)
        u = u[idx]
    
        # target
        target = (c==nb_of_clust).astype(float)
        
        # convert to pytorch
        W = torch.from_numpy(W)
        W = W.to(torch.int8)
        idx = torch.from_numpy(idx) 
        idx = idx.to(torch.int16)
        u = torch.from_numpy(u) 
        u = u.to(torch.int16)                      
        target = torch.from_numpy(target)
        target = target.to(torch.int16)
        
        # attributes
        self.nb_nodes = W.size(0)
        self.W = W
        self.rand_idx = idx
        self.node_feat = u
        self.node_label = target
        
        
# configuration
SBM_parameters = {}
SBM_parameters['nb_clusters'] = 10
SBM_parameters['size_min'] = 5
SBM_parameters['size_max'] = 15 # 25
SBM_parameters['p'] = 0.5 # 0.5
SBM_parameters['q'] = 0.25 # 0.1
SBM_parameters['p_pattern'] = 0.55 # 0.5
SBM_parameters['q_pattern'] = 0.25 # 0.1    
SBM_parameters['vocab_size'] = 3
SBM_parameters['size_subgraph'] = 10
SBM_parameters['W0'] = random_pattern(SBM_parameters['size_subgraph'],SBM_parameters['p_pattern'])
SBM_parameters['u0'] = np.random.randint(SBM_parameters['vocab_size'],size=SBM_parameters['size_subgraph'])
        
print(SBM_parameters)













# Block 3
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def plot_histo_graphs(dataset, title):
    # histogram of graph sizes
    graph_sizes = []
    for graph in dataset:
        graph_sizes.append(graph.nb_nodes)
    plt.figure(1)
    plt.hist(graph_sizes, bins=50)
    plt.title(title)
    plt.show()

    


start = time.time()


# configuration for 100 patterns 100/20 
nb_pattern_instances = 100 # nb of patterns
nb_train_graphs_per_pattern_instance = 100 # train per pattern
nb_test_graphs_per_pattern_instance = 20 # test, val per pattern

# # debug
# nb_pattern_instances = 10 # nb of patterns
# nb_train_graphs_per_pattern_instance = 10 # train per pattern
# nb_test_graphs_per_pattern_instance = 2 # test, val per pattern
# # debug

SBM_parameters = {}
SBM_parameters['nb_clusters'] = 5 
SBM_parameters['size_min'] = 5 
SBM_parameters['size_max'] = 35 
#SBM_parameters['p'] = 0.5 # v1
#SBM_parameters['q'] = 0.2 # v1
#SBM_parameters['p'] = 0.5 # v2
#SBM_parameters['q'] = 0.5 # v2
#SBM_parameters['p'] = 0.5; SBM_parameters['q'] = 0.25 # v3
SBM_parameters['p'] = 0.5; SBM_parameters['q'] = 0.45 # v4
SBM_parameters['p_pattern'] = 0.5
SBM_parameters['q_pattern'] = 0.55  
SBM_parameters['vocab_size'] = 3 
#SBM_parameters['size_subgraph'] = 20 # v1
SBM_parameters['size_subgraph_min'] = 5 # v2
SBM_parameters['size_subgraph_max'] = 35 # v2
print(SBM_parameters)
    

dataset_train = []
dataset_val = []
dataset_test = []
for idx in range(nb_pattern_instances):
    
    print('pattern:',idx)
    
    #SBM_parameters['W0'] = random_pattern(SBM_parameters['size_subgraph'],SBM_parameters['p']) # v1
    #SBM_parameters['u0'] = np.random.randint(SBM_parameters['vocab_size'],size=SBM_parameters['size_subgraph']) # v1
    size_subgraph = np.random.randint(SBM_parameters['size_subgraph_min'],SBM_parameters['size_subgraph_max'],size=1)[0] # v2
    SBM_parameters['W0'] = random_pattern(size_subgraph,SBM_parameters['p']) # v2
    SBM_parameters['u0'] = np.random.randint(SBM_parameters['vocab_size'],size=size_subgraph) # v2
    
    for _ in range(nb_train_graphs_per_pattern_instance):
        data = generate_SBM_graph(SBM_parameters)
        graph = DotDict()
        graph.nb_nodes = data.nb_nodes
        graph.W = data.W
        graph.rand_idx = data.rand_idx
        graph.node_feat = data.node_feat
        graph.node_label = data.node_label
        dataset_train.append(graph)

    for _ in range(nb_test_graphs_per_pattern_instance):
        data = generate_SBM_graph(SBM_parameters)
        graph = DotDict()
        graph.nb_nodes = data.nb_nodes
        graph.W = data.W
        graph.rand_idx = data.rand_idx
        graph.node_feat = data.node_feat
        graph.node_label = data.node_label
        dataset_val.append(graph)

    for _ in range(nb_test_graphs_per_pattern_instance):
        data = generate_SBM_graph(SBM_parameters)
        graph = DotDict()
        graph.nb_nodes = data.nb_nodes
        graph.W = data.W
        graph.rand_idx = data.rand_idx
        graph.node_feat = data.node_feat
        graph.node_label = data.node_label
        dataset_test.append(graph)


print(len(dataset_train),len(dataset_val),len(dataset_test))


plot_histo_graphs(dataset_train,'train')
plot_histo_graphs(dataset_val,'val')
plot_histo_graphs(dataset_test,'test')



with open('SBM_PATTERN_train.pkl',"wb") as f:
    pickle.dump(dataset_train,f)
with open('SBM_PATTERN_val.pkl',"wb") as f:
    pickle.dump(dataset_val,f)
with open('SBM_PATTERN_test.pkl',"wb") as f:
    pickle.dump(dataset_test,f)
    
    
print('Time (sec):',time.time() - start) # 163s