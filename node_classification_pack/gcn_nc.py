import argparse
import torch
from torch_geometric.loader import DataLoader
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt

import copy
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy import stats

from Utils.utils import check_task, load_model, detect_exp_setting, NC_vis_graph, show
from Utils.datasets import get_dataset

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataname = args.dataset
    task_type = check_task(dataname)
    dataset = get_dataset(dataname)
    n_fea, n_cls = dataset.num_features, dataset.num_classes
    explain_ids = detect_exp_setting(dataname, dataset)
    gnn_model = load_model(dataname, args.gnn, n_fea, n_cls)
    gnn_model.eval()
    print(f"GNN Model Loaded. {dataname}, {task_type}. \nnum of samples to explain: {len(explain_ids)}")
    
    # ---------------part of our explainer---------------
    if True:
        activation = {}
        gradient={}
        def get_gradient(name):
            def hook(model, input, output):
                gradient[name] = output[0]
            return hook

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        for (name, module) in gnn_model.named_modules():
            module.register_forward_hook(get_activation(name))
            module.register_backward_hook(get_gradient(name))

        model_params = {}
        for na, param in gnn_model.named_parameters():
            if param.requires_grad:
                model_params[na] = param.data

        h1w = model_params['conv1.lin.weight']
        h2w = model_params['convs.0.lin.weight']
        h3w = model_params['convs.1.lin.weight']
        lin1w = model_params['lin1.weight']
        lin2w = model_params['lin2.weight']

        h1b = model_params['conv1.bias']
        h2b = model_params['convs.0.bias']
        h3b = model_params['convs.1.bias']
        lin1b = model_params['lin1.bias']
        lin2b = model_params['lin2.bias']

        weights = [h1w, h2w, h3w, lin1w, lin2w]
        bias = [h1b, h2b, h3b, lin1b, lin2b]
        weights = [weight.float() for weight in weights]
        bias = [bb.float() for bb in bias]
        eps = 1e-12
        p: float = 2.0
        dim = -1
    # ---------------part of our explainer---------------

    tot_auc = []
    if task_type == "NC":

        dataset.to(device)
        x, edge_index, y = dataset.x, dataset.edge_index, dataset.y
        logits = gnn_model(x, edge_index)
        gnn_preds = torch.argmax(logits, dim=-1)
        interest_ids = range(x.shape[0])

        exp_embs = {cls:[] for cls in range(n_cls)}
        exp_Hedges = {nid:None for nid in range(x.shape[0])}

        # ---------feed into the explainer---------
        econfi, Hedges = gcn_explain(gnn_model, x, edge_index, y, weights, bias, activation, args, p, dim, eps, topk=50, mutual=args.mutual)
        # ---------feed into the explainer---------

        if args.do_plot>0: interest_ids = explain_ids
        for i in interest_ids: 
            if int(torch.argmax(y[i])) != int(gnn_preds[i]): continue
            topk = args.topk
            # ---------------Only for our explainer---------------
            if args.mutual==0 and i in explain_ids: 
                topk = min(topk, sum(econfi[i]>stats.mode(econfi[i])[0]), sum(econfi[i]>0))
            # ---------------Only for our explainer---------------

            # the node embedding of node i with only the explantion subgraph fed into the GCN
            exp_embs[int(gnn_preds[i])].append(gnn_model.get_emb(x,edge_index[:,Hedges[i,:topk]])[i])
            # global indices of the critical edges 
            exp_Hedges[i] = Hedges[i,:topk]
            print(f'{i}, class={gnn_preds[i]}')

            if args.do_plot>0:
                print(econfi[i])
                print(edge_index[:,Hedges[i]],"\n")
                NC_vis_graph(edge_index=dataset.edge_index, y=dataset.y, datasetname=dataname, node_idx=i, H_edges=Hedges[i,:topk])
                show()
        
        # save embeddings of the explanations and the GLOBAL edge indices of the explantions
        emb_path = 'embeddings/'+args.dataset.lower()+'_'+args.gnn+'_goat_mutual'+str(args.mutual)+'_'+str(args.topk)+'.pkl'
        save_emb = list(exp_embs.values())
        Hedges_path = 'Hedges/'+args.dataset.lower()+'_'+args.gnn+'_goat_mutual'+str(args.mutual)+'_'+str(args.topk)+'.pkl'
        import pickle
        with open(emb_path, "wb") as f:
            pickle.dump(save_emb, f)
        with open(Hedges_path, "wb") as f:
            pickle.dump(exp_Hedges, f)

# ---------------part of our explainer---------------
def gcn_explain(gnn_model, _x, _edge_index, y, weights, bias, activation, args, p, dim, eps, topk=None, mutual=1):
    x = F.normalize(_x, p=2.0, dim=-1).float()
    if True: 
        h1 = activation['conv1']
        h2 = activation['convs.0']
        h3 = activation['convs.1']
        lin1out = activation['lin1']
        lin2out = activation['lin2']
        h1_a = (activation['conv1']>0).float()
        h2_a = (activation['convs.0']>0).float()
        h3_a = (activation['convs.1']>0).float()
        lin1out_a = (activation['lin1']>0).float()
        denom_1 = (h1*h1_a).norm(p, dim, keepdim=True).clamp_min(eps).expand_as(h1*h1_a)  
        denom_2 = (h2*h2_a).norm(p, dim, keepdim=True).clamp_min(eps).expand_as(h2*h2_a)

        denoms = [denom_1, denom_2]
        edges = np.transpose(np.asarray(_edge_index.cpu()))
        act_p = [h1_a, h2_a, h3_a, lin1out_a]
        h_logits = [h1, h2, h3, lin1out, lin2out]
        denoms = [den.float() for den in denoms]
        h_logits = [hlo.float() for hlo in h_logits]

        # offset logits
        off_edge_index = torch.tensor([[],[]]).to(_x.device).long()
        off_logits = gnn_model.fwd_base(x, off_edge_index)
        h3_a_off = (activation['convs.1']>0).float()
        lin1out_a_off = (activation['lin1']>0).float()
        off_act_p = [h3_a_off, lin1out_a_off]

    # second-hop and third-hop edges
    first_hop_edges={a:_edge_index[:, torch.nonzero(_edge_index[0]==a).view(-1).tolist()] for a in range(x.shape[0])}
    second_hop_edges={a:0 for a in range(x.shape[0])}
    third_hop_edges={a:0 for a in range(x.shape[0])}
    for a in range(x.shape[0]):
        all_b = list(set(first_hop_edges[a][1].view(-1).tolist()))
        second_hop_edges[a]=next_hop_neigh(_edge_index, all_lefts=all_b, all_rights=None)
        all_c = list(set(second_hop_edges[a][1].view(-1).tolist()))
        third_hop_edges[a]=next_hop_neigh(_edge_index, all_lefts=all_c, all_rights=None)
    nhop_edges = [first_hop_edges, second_hop_edges, third_hop_edges]

    LAYER = 3
    edge_overall_score = {edge2key(a,b,x.shape[0]):0 for (a,b) in edges} 
    for (a,b) in edges:
        ekey = edge2key(a,b,x.shape[0])
        for loc in range(LAYER):
            out = gcn_layer_prop(loc, (a,b), gnn_model.conv1.propagate, _edge_index, x, weights, denoms, act_p)
            edge_overall_score[ekey] += out[-1]/(2*LAYER+1)
            for k in range(LAYER-1):
                if loc > k:
                    out = gcn_bias_prop(k, loc, (a,b), gnn_model.conv1.propagate, _edge_index, x, weights, bias, denoms, act_p)
                    edge_overall_score[ekey] += out[-1]/max(1,2*(LAYER-k))

    for act_id in range(LAYER):
        for a in range(x.shape[0]):
            tmp_act = torch.zeros(act_p[act_id].shape).to(x.device).float()
            tmp_act[a] = copy.deepcopy(act_p[act_id][a])
            ap = copy.deepcopy(act_p)
            ap[act_id] = tmp_act
            out = gcn_layer_prop(None, None, gnn_model.conv1.propagate, _edge_index, x, weights, denoms, ap)
            all_edges = []
            for l in range(act_id+1):
                all_edges += np.transpose(np.asarray(nhop_edges[l][a].cpu())).tolist()
            for (left, right) in all_edges:
                ekey = edge2key(left,right,x.shape[0])
                edge_overall_score[ekey] += out[-1]/(2*LAYER+1)/len(all_edges)
            for k in range(LAYER):
                if act_id >= k:
                    out = gcn_bias_prop(k, None, None, gnn_model.conv1.propagate, _edge_index, x, weights, bias, denoms, ap)
                    for (left, right) in all_edges:
                        ekey = edge2key(left,right,x.shape[0])
                        edge_overall_score[ekey] += out[-1]/max(1,2*(LAYER-k))/len(all_edges)

    for act_id in range(LAYER, len(act_p)):
        out = gcn_layer_prop(None, None, gnn_model.conv1.propagate, _edge_index, x, weights, denoms, act_p)
        for ekey in edge_overall_score.keys():
            edge_overall_score[ekey] += out[-1]/(2*LAYER+1)/_edge_index.shape[1]                        
        for k in range(len(act_p)):
            if act_id >= k:
                out = gcn_bias_prop(k, None, None, gnn_model.conv1.propagate, _edge_index, x, weights, bias, denoms, act_p)
                for ekey in edge_overall_score.keys():                        
                    edge_overall_score[ekey] += out[-1]/max(1,2*(LAYER-k))/_edge_index.shape[1]                      

    # '''     
    # substract offsets -- last conv layer
    act_id = 2
    k=LAYER-1
    for a in range(x.shape[0]):
        tmp_act = torch.zeros(off_act_p[0].shape).to(x.device).float()
        tmp_act[a] = copy.deepcopy(off_act_p[0][a])
        ap = copy.deepcopy(act_p)
        ap[LAYER-1] = tmp_act
        ap[LAYER] = copy.deepcopy(off_act_p[1])
        all_edges = []
        for l in range(act_id+1):
            all_edges += np.transpose(np.asarray(nhop_edges[l][a].cpu())).tolist()
        if act_id >= k:
            out = gcn_bias_prop(LAYER-1, None, None, gnn_model.conv1.propagate, _edge_index, x, weights, bias, denoms, ap)
            for (left, right) in all_edges:
                ekey = edge2key(left,right,x.shape[0])
                edge_overall_score[ekey] -= out[-1]/max(1,2*(LAYER-k))/len(all_edges)
    
    # substract offsets -- first classify layer
    for act_id in range(LAYER-1, len(act_p)):
        for a in range(x.shape[0]):
            tmp_act = torch.zeros(off_act_p[act_id-LAYER+1].shape).to(x.device).float()
            tmp_act[a] = copy.deepcopy(off_act_p[act_id-LAYER+1][a])
            ap = copy.deepcopy(act_p)
            ap[LAYER-1] = copy.deepcopy(off_act_p[0])
            ap[LAYER] = copy.deepcopy(off_act_p[1])
            ap[act_id] = tmp_act
            all_edges = []
            for l in range(min(LAYER, act_id+1)):
                all_edges += np.transpose(np.asarray(nhop_edges[l][a].cpu())).tolist()
            for k in range(LAYER-1, len(act_p)):
                if act_id >= k:
                    out = gcn_bias_prop(k, None, None, gnn_model.conv1.propagate, _edge_index, x, weights, bias, denoms, ap)
                    for (left, right) in all_edges:
                        ekey = edge2key(left,right,x.shape[0])
                        edge_overall_score[ekey] -= out[-1]/max(1,2*(LAYER-k))/(len(all_edges))
    # '''

    overall_score = torch.stack([score for score in edge_overall_score.values()])
    
    if args.is_undirected>0:
        undi_edge_overall_score = copy.deepcopy(overall_score)
        for ei in range(_edge_index.shape[1]):
            el, er = _edge_index[:,ei]
            el, er =int(el), int(er)
            rev_e = int((torch.logical_and(_edge_index[0]==er, _edge_index[1]==el)==True).nonzero()[0])
            undi_edge_overall_score[ei]+= overall_score[rev_e]
            undi_edge_overall_score[ei]*=0.5
        overall_score = undi_edge_overall_score
    
    if mutual>0: i_score = torch.norm(overall_score, dim=-1)
    else: i_score = overall_score[:,y.bool()]
    econfi, Hedges = torch.topk(i_score, topk, dim=0)[0].cpu().detach().numpy().T, torch.topk(i_score, topk, dim=0)[1].cpu().detach().numpy().T

    return econfi, Hedges

def gcn_layer_prop(loc, hidden, propagate, edge_index, x, weights, denoms, act_p):
    x = F.normalize(x, p=2.0, dim=-1).float()
    edge_weight = torch.ones(edge_index.shape[1]).to(edge_index.device).float()
    if isinstance(hidden, tuple): 
        if loc == 0:
            (a,b) = hidden
            tmp_h1_bef = torch.zeros(act_p[0].shape).to(x.device).float()
            tmp_h1_bef[a] = F.linear(x[b,:].view(1,-1),weights[0]).float()
            tmp_h1 = tmp_h1_bef*act_p[0].float()/denoms[0].float()
            tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1].float())
            tmp_h2 = tmp_h2_bef.float()*act_p[1]/denoms[1].float()
            tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2.float(), edge_weight=edge_weight),weights[2].float())
            tmp_h3 = tmp_h3_bef*act_p[2]
            tmplin1_bef = F.linear(tmp_h3,weights[3].float())
            tmplin1 = tmplin1_bef*act_p[3].float()
            tmplin2 = F.linear(tmplin1,weights[4])
        elif loc == 1:
            tmp_h1_bef = F.linear(propagate(edge_index, x=x, edge_weight=edge_weight),weights[0])
            tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
            (a,b) = hidden
            tmp_h2_bef = torch.zeros(act_p[1].shape).to(x.device).float()
            tmp_h2_bef[a] = F.linear(tmp_h1[b,:].view(1,-1),weights[1])
            tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
            tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
            tmp_h3 = tmp_h3_bef*act_p[2]
            tmplin1_bef = F.linear(tmp_h3,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin2 = F.linear(tmplin1,weights[4])
        elif loc == 2:
            tmp_h1_bef = F.linear(propagate(edge_index, x=x, edge_weight=edge_weight),weights[0])
            tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
            tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1])
            tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
            (a,b) = hidden
            tmp_h3_bef = torch.zeros(act_p[2].shape).to(x.device).float()
            tmp_h3_bef[a] = F.linear(tmp_h2[b,:].view(1,-1),weights[2])
            tmp_h3 = tmp_h3_bef*act_p[2]
            tmplin1_bef = F.linear(tmp_h3,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin2 = F.linear(tmplin1,weights[4])
    else:
        tmp_h1_bef = F.linear(propagate(edge_index, x=x.float(), edge_weight=edge_weight),weights[0])
        tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
        tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1])
        tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
        tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
        tmp_h3 = tmp_h3_bef*act_p[2]
        tmplin1_bef = F.linear(tmp_h3,weights[3])
        tmplin1 = tmplin1_bef*act_p[3]
        tmplin2 = F.linear(tmplin1,weights[4])
    return [tmp_h1_bef, tmp_h2_bef, tmp_h3_bef, tmplin1_bef, tmplin2]

def gcn_bias_prop(layer, loc, hidden, propagate, edge_index, x, weights, bias, denoms, act_p):
    edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
    if layer==0:
        if isinstance(hidden, tuple): 
            (a,b) = hidden
            tmp_h1_bef = bias[0].repeat(x.shape[0],1).float()
            tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
            if loc==1:
                tmp_h2_bef = torch.zeros(act_p[1].shape).to(x.device).float()
                tmp_h2_bef[a] = F.linear(tmp_h1[b,:].view(1,-1),weights[1])
                tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
                tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
                tmp_h3 = tmp_h3_bef*act_p[2]
            elif loc==2:
                tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1])
                tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
                tmp_h3_bef = torch.zeros(act_p[2].shape).to(x.device).float()
                tmp_h3_bef[a] = F.linear(tmp_h2[b,:].view(1,-1),weights[2])
                tmp_h3 = (tmp_h3_bef*act_p[2][a]).float()
            tmplin1_bef = F.linear(tmp_h3,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin2 = F.linear(tmplin1,weights[4])
        else:
            tmp_h1_bef = bias[0].repeat(x.shape[0],1).float()
            tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
            tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1])
            tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
            tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
            tmp_h3 = tmp_h3_bef*act_p[2]
            tmplin1_bef = F.linear(tmp_h3,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin2 = F.linear(tmplin1,weights[4])
        return [tmp_h1_bef, tmp_h2_bef, tmp_h3_bef, tmplin1_bef, tmplin2]
    elif layer==1:
        if isinstance(hidden, tuple): 
            (a,b) = hidden
            if loc==2:
                tmp_h2_bef = bias[1].repeat(x.shape[0],1).float()
                tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
                tmp_h3_bef = F.linear(tmp_h2[b,:].view(1,-1),weights[2])
                tmp_h3_bef = torch.zeros(act_p[2].shape).to(x.device).float()
                tmp_h3_bef[a] = F.linear(tmp_h2[b,:].view(1,-1),weights[2])
                tmp_h3 = tmp_h3_bef*act_p[2]
                tmplin1_bef = F.linear(tmp_h3,weights[3])
                tmplin1 = tmplin1_bef*act_p[3]
                tmplin2 = F.linear(tmplin1,weights[4])
            else: 
                print("ERROR: not loc==2. ")
                exit(0)
        else:
            tmp_h2_bef = bias[1].repeat(x.shape[0],1).float()
            tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
            tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
            tmp_h3 = tmp_h3_bef*act_p[2]
            tmplin1_bef = F.linear(tmp_h3,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin2 = F.linear(tmplin1,weights[4])
        return [None, tmp_h2_bef, tmp_h3_bef, tmplin1_bef, tmplin2]
    elif layer==2:
        tmp_h3_bef = bias[2].repeat(x.shape[0],1).float()
        tmp_h3 = tmp_h3_bef*act_p[2]
        tmplin1_bef = F.linear(tmp_h3,weights[3])
        tmplin1 = tmplin1_bef*act_p[3]
        tmplin2 = F.linear(tmplin1,weights[4])
        return [None, None, tmp_h3_bef, tmplin1_bef, tmplin2]
    elif layer==3:
        tmplin1_bef = bias[3].view(1,-1).float()
        tmplin1 = tmplin1_bef*act_p[3]
        tmplin2 = F.linear(tmplin1,weights[4])
        return [None, None, None, tmplin1_bef, tmplin2]

def edge2key(a,b,N):
    return int(a*N+b)

def next_hop_neigh(edge_index, all_lefts=None, all_rights=None):
    if all_rights is None:
        # print(sum(torch.nonzero(edge_index[0]==all_lefts[i]).view(-1).tolist() for i in range(len(all_lefts))))
        all_next_hop_edges = []
        for i in range(len(all_lefts)):
            all_next_hop_edges += torch.nonzero(edge_index[0]==all_lefts[i]).view(-1).tolist()
    else:
        print('all_rights is not None')
        exit(0)    
    return edge_index[:, all_next_hop_edges]

# ---------------part of our explainer---------------

def plot(dataset, dataname, i, Hedges):
    NC_vis_graph(edge_index=dataset.edge_index, y=dataset.y, datasetname=dataname, node_idx=i, H_edges=Hedges)
    show()

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ba_shape')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--do_plot', type=int, default=0)
    parser.add_argument('--is_undirected', type=int, default=1)
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--mutual', type=int, default=0)
    
    return parser.parse_args()

if __name__ == "__main__":

    args = build_args()
    main(args)
    print("done")









