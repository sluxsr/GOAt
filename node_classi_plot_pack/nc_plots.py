import pickle
import argparse
import torch 
import matplotlib.pyplot as plt
import numpy as np
import random

from Utils.utils import show, NC_vis_graph
from Utils.datasets import get_dataset

def main(args):
    print(f"-----------\n{args.explainer} plots and means:")
    emb_path = 'embeddings/'+args.dataset.lower()+'_'+args.gnn+'_'+args.explainer+'_mutual'+str(args.mutual)+'_'+str(args.topk)+'.pkl'
    emb_means_diffs = show_scatter_and_cluster(emb_path)
    print(f"-----------\noriginal (w/o explainers) plots and means:")
    orig_emb_path = 'embeddings/'+args.dataset.lower()+'_'+args.gnn+'_orig'+'.pkl'
    orig_mean_diffs = show_scatter_and_cluster(orig_emb_path)

    diff_means = emb_means_diffs-orig_mean_diffs
    print(f'Sum of distance improved compared with original: {sum(diff_means)}')

    Hedges = get_Hedges(args)
    check_range = range(511,530) if args.dataset.lower()=='tree_grid' else range(300,310)
    print(f"--------------\nsample Hedges for {args.explainer}: ")
    dataset = get_dataset(args.dataset)
    for i in check_range:
        print(f'{i}: ',end='')
        if Hedges[i] is None: 
            print()
            continue
        print('[',end='')
        print(*Hedges[i], sep = ",",end='],\n')
        plot(dataset,args.dataset,i,Hedges[i])
    exit(0)

def plot(dataset, dataname, i, Hedges):
    NC_vis_graph(edge_index=dataset.edge_index, y=dataset.y, datasetname=dataname, node_idx=i, H_edges=Hedges)
    show()

def show_scatter_and_cluster(path):
    emb_list = get_embeddings(path)
    exp_embs = {i: emb_list[i] for i in range(len(emb_list))}

    colors = ['red', 'blue', 'pink', '#FFC300', 'aqua', 'green', '#8300AA', 'black']
    means = []
    # embedding variance of the explanations in each class
    for j in exp_embs.keys():
        if len(exp_embs[j])==0:continue
        if len(exp_embs.keys())>2 and j%4==0: continue
        all_embs = torch.stack(exp_embs[j])
        plt.scatter(all_embs[:,3].cpu().detach(), all_embs[:,4].cpu().detach(), marker='.', alpha=0.2, c=[colors[j]]*len(all_embs))
        var,mean = torch.var_mean(all_embs, dim=0, keepdim=True)
        means.append(mean)
        print(f'Class{j}, topk={args.topk}, var={torch.mean(var)}, mean={torch.mean(mean)}')

    diffs = []
    for i in range(len(means)):
        for k in range(len(means)):
            if i<k:
                diff = float((means[i]-means[k]).norm())
                print(f'mean distance between classes {i,k}: {diff}')
                diffs.append(diff)
    
    plt.xlabel('embedding at dimension 0', size=8)
    plt.ylabel('embedding at dimension 1', size=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    show()
    return np.asarray(diffs)


def get_Hedges(args):
    Hedges_path = 'Hedges/'+args.dataset.lower()+'_'+args.gnn+'_'+args.explainer+'_mutual'+str(args.mutual)+'_'+str(args.topk)+'.pkl'
    with open(Hedges_path, "rb") as f:
        Hedges = pickle.load(f)
    return Hedges

def get_embeddings(embedding_pkl_file):
    with open(embedding_pkl_file, "rb") as f:
        all_emb = [x for x in pickle.load(f)]
    return all_emb

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ba_shape')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--explainer', type=str, default='goat')
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--mutual', type=int, default=1)

    return parser.parse_args()

if __name__ == "__main__":

    args = build_args()
    main(args)
    print("done")