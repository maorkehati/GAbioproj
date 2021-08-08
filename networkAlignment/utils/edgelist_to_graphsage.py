from collections import defaultdict
import argparse
import numpy as np
import json
import os
import networkx as nx
from networkx.readwrite import json_graph
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Generate graphsage format from edgelist")
    parser.add_argument('--out_dir', default=None, help="Output directory")
    parser.add_argument('--prefix', default="karate", help="seed")
    parser.add_argument('--seed', default=121, type=int)
    return parser.parse_args()

def preprocess_edgelist(edgelist_pre, edgelist, edgelist_dic):
    os.remove(edgelist)
    os.remove(edgelist_dic)
    d = {}
    c = 0
    fc = []
    f = open(edgelist_pre,'r')
    for l in f.readlines():
        ns = l.strip().split()[:2]
        for n in ns:
            if n not in d:
                d[n] = str(c)
                c += 1
                
                
        l = l.replace(ns[0], d[ns[0]]).replace(ns[1], d[ns[1]])
        l = " ".join(l.strip().split()[:2])
                
        fc.append(l)
        
    f.close()
    
    fc = "\n".join(fc)
        
    f = open(edgelist,'w')
    f.write(fc)
    f.close()
    
    with open(edgelist_dic, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def edgelist_to_graphsage(dir, seed=121):
    np.random.seed(seed)
    edgelist_pre = dir + "/edgelist/edgelist_pre"
    edgelist = dir + "/edgelist/edgelist"
    edgelist_dic = dir + "/edgelist/dic.pkl"
    preprocess_edgelist(edgelist_pre, edgelist, edgelist_dic)

    print(edgelist)
    G = nx.read_edgelist(edgelist)
    
    #get 2 core of largest connected component as suggested in comparison article
    #G = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0] #largest connected component
    #G = nx.algorithms.core.k_core(G, k=2) #2-core
    
    print(nx.info(G))
    num_nodes = len(G.nodes())
    rand_indices = np.random.permutation(num_nodes)
    train = rand_indices[:int(num_nodes * 0.81)]
    val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
    test = rand_indices[int(num_nodes * 0.9):]

    id2idx = {}
    for i, node in enumerate(G.nodes()):
        id2idx[str(node)] = i

    res = json_graph.node_link_data(G)
    res['nodes'] = [
        {
            'id': node['id'],
            'val': id2idx[str(node['id'])] in val,
            'test': id2idx[str(node['id'])] in test
        }
        for node in res['nodes']]

    res['links'] = [
        {
            'source': link['source'],
            'target': link['target']
        }
        for link in res['links']]

    if not os.path.exists(dir + "/graphsage/"):
        os.makedirs(dir + "/graphsage/")

    with open(dir + "/graphsage/" + "G.json", 'w') as outfile:
        json.dump(res, outfile)
    with open(dir + "/graphsage/" + "id2idx.json", 'w') as outfile:
        json.dump(id2idx, outfile)

    print("GraphSAGE format stored in {0}".format(dir + "/graphsage/"))
    print("----------------------------------------------------------")

if __name__ == "__main__":
    args = parse_args()
    datadir = args.out_dir
    dataset = args.prefix
    edgelist_to_graphsage(datadir)



