import pickle
import argparse
import torch 
import matplotlib.pyplot as plt
import numpy as np
from fast_pytorch_kmeans import KMeans

from Utils.utils import show

def main(args):
    # path = 'embeddings/'+args.dataset+'_'+args.gnn+'_'+args.explainer+'_'+str(int(args.sparsity))+'.pkl'
    # # show_scatter_and_cluster(path)
    # show_orig_scatter(args)
    # get_topk_indices(path,args)
    # # print(get_orig_mean_dist(args.dataset, args))
    compare_scatter(['sgx','goat'], ['Mutagenicity', 'NCI1'], ['SubgraphX','GOAt (Ours)'], args)
    exit(0)

    # explainers = ['gnn','pg','pgm','rg','rc','goat','rc','goat']
    # titles = ['GNNExplainer','PGExplainer','PGM-Explainer','RG-Explainer','RCExplainer','GOAt (Ours)','RCExplainer','GOAt (Ours)']
    
    explainers = ['gnn','pg','pgm','rg','sgx','rc','degree','goat']
    # explainers = ['goat']
    titles = ['GNNExplainer','PGExplainer','PGM-Explainer','RG-Explainer','SubgraphX','RCExplainer','DEGREE','GOAt (Ours)']
    datasets = ['ba_2motifs', 'Mutagenicity', 'NCI1']
    colors = ['red', 'blue', 'pink', '#FFC300', 'aqua', 'green', '#8300AA', 'black']

    # show_all_scatter(explainers, titles, args)
    # show_cover_curve(explainers, titles, args)
    
    # show_all_mean_var(explainers, args)
    # cluster_all(explainers, args)

    show_discriminability(explainers, datasets, titles, colors, args)
    # show_all_cover_curve(explainers, datasets, titles, colors, args)

def compare_scatter(explainers, datasets, titles, args):
    classes = {0:None, 1:None}
    colors = ['#02C39A', '#FF2A00', '#9FE2BF', '#DAF7A6']
    index = ['a','b','c','d','e','f','g','h','i']

    nrows=len(datasets)
    ncols = len(explainers)+1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=True)
    for i,ax in enumerate(axs.flat):
        for dir in ax.spines:
            ax.spines[dir].set_color('#dddddd')
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=7)
        if (i+1)%ncols==0:
            ax.set_title('('+index[i]+')Original on '+datasets[i//ncols], size=8)
            path = 'embeddings/'+datasets[i//ncols].lower()+'_'+args.gnn+'_orig'+'.pkl'
            exp_embs =get_embeddings(path)
            for j in range(len(exp_embs)):
                if len(exp_embs[j])==0:continue
                all_embs = exp_embs[j]
                classes[j]=ax.scatter(all_embs[:,0].cpu().detach(), all_embs[:,1].cpu().detach(), marker='.', s=3, alpha=0.2, c=[colors[j]]*len(all_embs))
        else:
            ax.set_title('('+index[i]+')'+titles[i%ncols]+' on '+datasets[i//ncols], size=8)
            path = 'embeddings/'+datasets[i//ncols].lower()+'_'+args.gnn+'_'+explainers[i%ncols]+'_'+str(int(args.sparsity))+'.pkl'
            exp_embs =get_embeddings(path)
            for j in range(len(exp_embs)):
                if len(exp_embs[j])==0:continue
                all_embs = exp_embs[j]
                classes[j]=ax.scatter(all_embs[:,0].cpu().detach(), all_embs[:,1].cpu().detach(), marker='.', s=3, alpha=0.2, c=[colors[j]]*len(all_embs))
    show()
    return 

def show_all_cover_curve(explainers, datasets, titles, colors, args):
    fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    index = ['a','b','c','d','e','f','g','h','i']
    fig_titles = ['BA-2Motifs', 'Mutagenicity', 'NCI1']

    for i, ax in enumerate(axs.flat):
        ax.set_xlabel('Top k motifs (k)', size=8)
        ax.set_ylabel('Coverage (%)', size=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=7)
        ax.set_title('('+index[i]+')'+fig_titles[i], size=8)
        
        try:
            if datasets[i].lower()!='ba_2motifs':
                # _explainers = ['gnn','pg','pgm','rg','gnn','rc','degree','goat']
                all_k = range(5,500,20)
                ax.set_xticks(np.arange(-95, 600, step=100)) 
            else: 
                all_k = range(1,10)
                ax.set_xticks(np.arange(-1, 15, step=2)) 
            all_coverage = []
            for k in all_k:
                coverage, total_counts = show_top_kexp_cover(explainers, args,k=k, dataset=datasets[i].lower())
                if datasets[i].lower()=="mutagenicity": m=1
                else: m=1
                all_coverage.append(coverage[m])
            all_ys = 100*np.asarray(all_coverage)/total_counts[m]
            labels = []
            for j in range(all_ys.shape[1]):
                line = '-' if explainers[j]=='goat' else'--'
                labels.append(ax.errorbar(all_k, all_ys[:,j], color=colors[j], linestyle=line, marker=".", linewidth=.5, markersize=1))
        except: break
    
    plt.figlegend(
        (labels),
        (titles),
        loc='lower center',
        ncol = 4,
        framealpha=0.5,
        prop={'size': 6})
    show()
    return 

def show_cover_curve(explainers, titles, colors, args):
    all_coverage = {0:[],1:[]}
    for k in range(1,10):
        coverage, total_counts = show_top_kexp_cover(explainers, args,k=k)
        for j in all_coverage.keys():
            all_coverage[j].append(coverage[j])
    # print(all_coverage)

    labels = []
    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    for i, ax in enumerate(axs.flat):
        ax.set_xlabel('Top k motifs (k)', size=8)
        ax.set_ylabel('Coverage (%)', size=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=7)
        all_ys = 100*np.asarray(all_coverage[i])/total_counts[j]
        common_x = list(range(1,10))
        for j in range(all_ys.shape[1]):
            line = '-' if explainers[j]=='goat' else'--'
            labels.append(ax.errorbar(common_x, all_ys[:,j], color=colors[j], linestyle=line, marker=".", linewidth=.5, markersize=1))
    
    plt.figlegend(
        (labels),
        (titles),
        loc='lower center',
        ncol = 4,
        framealpha=0.5,
        prop={'size': 6})
    show()
    return 

def show_top_kexp_cover(explainers, args, k=3, dataset=None):
    if dataset is None: dataset=args.dataset
    coverage={0:[],1:[]}
    total_counts = []
    for i, explainer in enumerate(explainers):
        path = 'embeddings/'+dataset+'_'+args.gnn+'_'+explainer+'_'+str(int(args.sparsity))+'.pkl'
        exp_embs = {}
        exp_embs[0]=get_embeddings(path)[0]
        exp_embs[1]=get_embeddings(path)[1]

        # print(f'\n>> {explainer}:')
        for j in exp_embs.keys():
            if len(exp_embs[j])==0:continue
            all_embs = exp_embs[j]
            total_counts.append(int(all_embs.shape[0]))
            unq_emb, counts = torch.unique(all_embs, sorted=True, dim=0, return_counts=True)
            topkcounts = sum(torch.topk(counts,k=min(k,unq_emb.shape[0]))[0])
            coverage[j].append(int(topkcounts))
    # print(coverage)
    return coverage, total_counts

def show_discriminability(explainers, datasets, titles, colors, args):
    fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    index = ['a','b','c','d','e','f','g','h','i']
    fig_titles = ['BA-2Motifs', 'Mutagenicity', 'NCI1']
    sparsities = ['7', '75', '8', '85', '9']
    for i,ax in enumerate(axs.flat):
        ax.set_xlabel('Average sparsity (%)', size=8)
        ax.set_ylabel('Discriminability', size=8)
        ax.set_xticks(np.arange(65, 98, step=5)) 
        # if datasets[i] =='ba_2motifs': ax.set_yscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=6)
        ax.set_title('('+index[i]+')'+fig_titles[i], size=8)

        common_x = [70, 75, 80, 85, 90]
        all_ys = []
        try:
            for spa in sparsities:
                sp_ys = []
                for j, explainer in enumerate(explainers):
                    path = 'embeddings/'+datasets[i].lower()+'_'+args.gnn+'_'+explainer+'_'+spa+'.pkl'
                    exp_embs = {}
                    exp_embs[0]=get_embeddings(path)[0]
                    exp_embs[1]=get_embeddings(path)[1]
                    means = []
                    for j in exp_embs.keys():
                        if len(exp_embs[j])==0:continue
                        all_embs = exp_embs[j]
                        var,mean = torch.var_mean(all_embs, dim=0, keepdim=True)
                        means.append(mean)
                    sp_ys.append(float((means[0]-means[1]).norm()))
                all_ys.append(sp_ys)
            all_ys = np.asarray(all_ys)
            
            labels = []
            for j in range(all_ys.shape[1]):
                line = '-' if explainers[j]=='goat' else'--'
                labels.append(ax.errorbar(common_x, all_ys[:,j], color=colors[j], capsize=1, linestyle=line, marker=".", linewidth=.5, markersize=1))
            orig_dist = float(get_orig_mean_dist(datasets[i], args))
            labels.append(ax.errorbar(common_x, [orig_dist]*len(common_x), color='#B6B6B6', capsize=1, linestyle='--', marker='.', linewidth=.5, markersize=0))
        except:
            break
    
    plt.figlegend(
        (labels),
        (titles+['Original']),
        loc='lower center',
        ncol = 5,
        framealpha=0.5,
        prop={'size': 6})
    show()
    return 

def show_all_mean_var(explainers, args, sparsity=None): 
    if sparsity is None: sparsity = args.sparsity
    for i, explainer in enumerate(explainers):
        path = 'embeddings/'+args.dataset+'_'+args.gnn+'_'+explainer+'_'+str(int(sparsity))+'.pkl'
        exp_embs = {}
        exp_embs[0]=get_embeddings(path)[0]
        exp_embs[1]=get_embeddings(path)[1]

        print(f'\n>> {explainer}:')
        means = []
        for j in exp_embs.keys():
            if len(exp_embs[j])==0:continue
            all_embs = exp_embs[j]
            var,mean = torch.var_mean(all_embs, dim=0, keepdim=True)
            means.append(mean)
            print(f'Class{j}, sparsity={args.sparsity}, var={torch.mean(var)}, mean={torch.mean(mean)}')
        print(f'mean distance: {(means[0]-means[1]).norm()}')
    return 

def cluster_all(explainers, args):
    for i, explainer in enumerate(explainers):
        path = 'embeddings/'+args.dataset+'_'+args.gnn+'_'+explainer+'_'+str(int(args.sparsity))+'.pkl'
        exp_embs = {}
        exp_embs[0]=get_embeddings(path)[0]
        exp_embs[1]=get_embeddings(path)[1]

        print(f'\n>> {explainer}:')
        for j in exp_embs.keys():
            if len(exp_embs[j])==0:continue
            all_embs = exp_embs[j]
            kmeans = KMeans(n_clusters=args.clusters, mode='euclidean', tol=1e-8, verbose=1)
            # kmeans._show=True
            labels, dist = kmeans.fit_predict(all_embs.detach())
            torch.set_printoptions(sci_mode=False)
            _,label_idx = torch.unique(labels,return_inverse=True)
            for clus in range(args.clusters):
                all_idx = torch.nonzero(label_idx==clus).view(-1)
                # print(clus, all_idx.cpu(), dist[all_idx].norm().cpu())
                print(clus, all_idx.shape, dist[all_idx].norm().cpu())
    return 

def get_orig_mean_dist(dataset, args):
    path = 'embeddings/'+dataset.lower()+'_'+args.gnn+'_orig'+'.pkl'
    means = []
    exp_embs = {}
    exp_embs[0]=get_embeddings(path)[0]
    exp_embs[1]=get_embeddings(path)[1]
    for j in exp_embs.keys():
        if len(exp_embs[j])==0:continue
        all_embs = exp_embs[j]
        var,mean = torch.var_mean(all_embs, dim=0, keepdim=True)
        means.append(mean)
    return (means[0]-means[1]).norm()

def show_orig_scatter(args):
    classes = {0:None, 1:None}
    colors = ['#02C39A', '#FF2A00', '#9FE2BF', '#DAF7A6']
    path = 'embeddings/'+args.dataset.lower()+'_'+args.gnn+'_orig'+'.pkl'

    means = []
    exp_embs = {}
    exp_embs[0]=get_embeddings(path)[0]
    exp_embs[1]=get_embeddings(path)[1]
    for j in exp_embs.keys():
        if len(exp_embs[j])==0:continue
        all_embs = exp_embs[j]
        classes[j]=plt.scatter(all_embs[:,0].cpu().detach(), all_embs[:,1].cpu().detach(), marker='.', s=15, alpha=0.2, c=[colors[j]]*len(all_embs))
        var,mean = torch.var_mean(all_embs, dim=0, keepdim=True)
        means.append(mean)
        print(f'Class{j}, sparsity={args.sparsity}, var={torch.mean(var)}, mean={torch.mean(mean)}')

    print(f'mean distance: {(means[0]-means[1]).norm()}')
    show()

def show_all_scatter(explainers, titles, args):
    colors = ['#02C39A', '#FF2A00', '#9FE2BF', '#DAF7A6']
    index = ['a','b','c','d','e','f','g','h','i']

    classes = {0:None, 1:None}
    fig = plt.figure(layout="constrained")
    subfigs = fig.subfigures(1, 2, wspace=0.00, width_ratios=[4., 1.5])
    # fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True)
    axs = subfigs[0].subplots(nrows=2, ncols=4)
    for i,ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        for dir in ax.spines:
            ax.spines[dir].set_color('#dddddd')
        ax.set_title('('+index[i]+')'+titles[i], size=8)
        
        path = 'embeddings/'+args.dataset+'_'+args.gnn+'_'+explainers[i]+'_'+str(int(args.sparsity))+'.pkl'
        exp_embs = {}
        exp_embs[0]=get_embeddings(path)[0]
        exp_embs[1]=get_embeddings(path)[1]

        for j in exp_embs.keys():
            if len(exp_embs[j])==0:continue
            all_embs = exp_embs[j]
            classes[j]=ax.scatter(all_embs[:,0].cpu().detach(), all_embs[:,1].cpu().detach(), marker='.', s=3, alpha=0.2, c=[colors[j]]*len(all_embs))
    if True:
        ax = subfigs[1].subplots(nrows=1, ncols=1)
        # for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for dir in ax.spines:
            ax.spines[dir].set_color('#dddddd')
            ax.set_title('('+index[-1]+') Original', size=8)
        path = 'embeddings/'+args.dataset.lower()+'_'+args.gnn+'_orig'+'.pkl'
        exp_embs = {}
        exp_embs[0]=get_embeddings(path)[0]
        exp_embs[1]=get_embeddings(path)[1]
        for j in exp_embs.keys():
            if len(exp_embs[j])==0:continue
            all_embs = exp_embs[j]
            classes[j]=ax.scatter(all_embs[:,0].cpu().detach(), all_embs[:,1].cpu().detach(), marker='.', s=3, alpha=0.2, c=[colors[j]]*len(all_embs))

    plt.figlegend(
        (classes[0],classes[1]),
        ('Class 0', 'Class 1'),
        loc='lower center',
        ncol = len(classes),
        framealpha=0.5,
        prop={'size': 6})
    show()
    # top=0.93,
    # bottom=0.09,
    # left=0.02,
    # right=0.99,
    # hspace=0.17,
    # wspace=0.12
    return 

def get_topk_indices(path, args, k=2):

    global_index = get_global_index(args.dataset)
    total_counts = []
    exp_embs = {}
    exp_embs[0]=get_embeddings(path)[0]
    exp_embs[1]=get_embeddings(path)[1]

    print(f'{args.dataset}, {args.explainer}, sparsity={args.sparsity}:')

    for j in exp_embs.keys():
        if len(exp_embs[j])==0:continue
        all_embs = exp_embs[j]
        total_counts.append(int(all_embs.shape[0]))
        unq_emb, inverse_indices, counts = torch.unique(all_embs, sorted=True, dim=0, return_inverse=True, return_counts=True)
        topkcounts, top_idx = torch.topk(counts,k=min(k,unq_emb.shape[0]))
        for i, idx in enumerate(top_idx):
            print(f'Class={j}, Data id={global_index[j][int(torch.nonzero(inverse_indices==idx)[0,0])]}, Coverage={topkcounts[i]/float(total_counts[j])}')
            # print(f'    >> more ids: {[global_index[j][int(m)] for m in torch.nonzero(inverse_indices==idx)[:5,0]]}')

    return 

def show_scatter_and_cluster(path):
    exp_embs = {}
    exp_embs[0]=get_embeddings(path)[0]
    exp_embs[1]=get_embeddings(path)[1]

    colors = ['red', 'blue', 'green', 'orange']
    means = []

    for j in exp_embs.keys():
        if len(exp_embs[j])==0:continue
        all_embs = exp_embs[j]
        kmeans = KMeans(n_clusters=args.clusters, mode='euclidean', tol=1e-8, verbose=1)
        # kmeans._show=True
        labels, dist = kmeans.fit_predict(all_embs.detach())
        torch.set_printoptions(sci_mode=False)
        _,label_idx = torch.unique(labels,return_inverse=True)
        for clus in range(args.clusters):
            all_idx = torch.nonzero(label_idx==clus).view(-1)
            print(clus, all_idx.cpu(), dist[all_idx].norm().cpu())
            print(clus, all_idx.shape, dist[all_idx].norm().cpu())
            
    # embedding variance of the explanations in each class
    for j in exp_embs.keys():
        if len(exp_embs[j])==0:continue
        all_embs = exp_embs[j]
        plt.scatter(all_embs[:,0].cpu().detach(), all_embs[:,1].cpu().detach(), marker='.', alpha=0.2, c=[colors[j]]*len(all_embs))
        var,mean = torch.var_mean(all_embs, dim=0, keepdim=True)
        means.append(mean)
        print(f'Class{j}, sparsity={args.sparsity}, var={torch.mean(var)}, mean={torch.mean(mean)}')

    print(f'mean distance: {(means[0]-means[1]).norm()}')
    plt.xlabel('embedding at dimension 0', size=8)
    plt.ylabel('embedding at dimension 1', size=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    show()

def get_global_index(dataset):
    path = 'embeddings/'+dataset.lower()+'_mapping'+'.pkl'
    with open(path, "rb") as f:
        global_index = pickle.load(f)
    return global_index

def get_embeddings(embedding_pkl_file):
    with open(embedding_pkl_file, "rb") as f:
        [class0Tensor, class1Tensor] = pickle.load(f)
    return class0Tensor, class1Tensor

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ba_2motifs')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--explainer', type=str, default='pg')
    parser.add_argument('--sparsity', type=float, default=7)
    parser.add_argument('--clusters', type=int, default=10)
    parser.add_argument('--maxk', type=int, default=10)

    return parser.parse_args()

if __name__ == "__main__":

    args = build_args()
    main(args)
    print("done")