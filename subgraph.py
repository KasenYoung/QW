import torch
from torch_geometric.datasets import WikipediaNetwork
import torch_geometric.transforms as T
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.datasets import Amazon
import random
import torch_geometric.utils as pyg_utils
import os
import argparse
from dataset_loader import DataLoader


def sampling(data, num_partitions):
    num_nodes=data.num_nodes
    partitioned_data=[]
    length=num_nodes//num_partitions+1
    unvis_nodes=[i for i in range(num_nodes)]
    for i in range(num_partitions):
        partitioned_data.append(bfs(data,unvis_nodes,length))
    
    return partitioned_data

def bfs(data,unvis_nodes,length):
    cnt=0
    Map={}
    edges=data.edge_index
    while cnt<length and len(unvis_nodes)>0:
        v=int(random.choice(unvis_nodes))
        
        queue=[]
        queue.append(v)
        while cnt<length and len(unvis_nodes)>0 and len(queue)>0:
            v=queue[0]
            print(v)
            queue.pop(0)
            #print(v)
            Map[v]=cnt
            cnt+=1
            unvis_nodes.remove(v)
            #print('len',len(unvis_nodes))
            for e in edges:
                if e[0]==v and e[1] in unvis_nodes:
                    queue.append(int(e[1]))
    src=list(Map.keys())
    dst=list(Map.values())
    x_sampled=data.x[src]
    y_sampled=data.y[src]
    e_sampled=[]
    for e in range(edges.shape[1]):
        if int(edges[0][e]) in src and int(edges[1][e]) in src:
            e_sampled.append([Map[int(edges[0][e])],Map[int(edges[1][e])]])
    e_sampled=torch.tensor(e_sampled,dtype=torch.long).T

    subgraph=Data(x=x_sampled,edge_index=e_sampled,y=y_sampled)

    return subgraph







if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='squirrel',help='name of the large dataset.')
    parser.add_argument('--num_subgraphs',type=int,default=5,help='num of subgraphs.')
    
   
    args=parser.parse_args()
    data=DataLoader(args.dataset)[0]
    cnt=0
    if not os.path.exists('./subgraphs'):
        os.mkdir('./subgraphs')
    for subgraph in sampling(data,num_partitions=args.num_subgraphs):
        print(subgraph.x.shape,subgraph.edge_index.shape,subgraph.y.shape)
        torch.save(subgraph,f'./subgraphs/{args.dataset}_{cnt}.pt')
        cnt+=1
