import argparse
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import time
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from Utils.utils import check_task, load_model, detect_exp_setting, detect_motif_nodes, NC_vis_graph, GC_vis_graph, NLP_vis_graph, find_thres, show
from Utils.metrics import acc, efidelity, nfidelity

from Utils.datasets import get_dataset, get_graph_data
import torch.nn.functional as F
import copy
from torch_geometric.nn import global_mean_pool

from sklearn.metrics import roc_auc_score
from fast_pytorch_kmeans import KMeans

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataname = args.dataset
    task_type = check_task(dataname)
    dataset = get_dataset(dataname)
    n_fea, n_cls = dataset.num_features, dataset.num_classes
    explain_ids = detect_exp_setting(dataname, dataset)
    motif_nodes_number = detect_motif_nodes(dataname)
    gnn_model = load_model(dataname, args.gnn, n_fea, n_cls)
    gnn_model.eval()
    print(f"GNN Model Loaded. {dataname}, {task_type}. \nsize of Motif: {motif_nodes_number}. num of samples to explain: {len(explain_ids)}")
    
    Fidelities = []
    neg_fids = []
    A_sparsities, Times = [], []

    if task_type == "GC":
        loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

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
            weights = [weight.double() for weight in weights]
            bias = [bb.double() for bb in bias]
            eps = 1e-12
            p: float = 2.0
            dim = -1

        if dataname == 'Mutagenicit':
            # for NO2/NH2 acc
            edge_lists, graph_labels, edge_label_lists, node_label_lists = get_graph_data(dataname)
            selected =  []
            for gid in range(len(dataset)):
                if np.sum(edge_label_lists[gid]) > 0:
                    selected.append(gid)
            print('number of mutagen graphs with NO2 and NH2',len(selected))
            interest_ids = selected
        else: interest_ids = range(len(dataset))

        exp_embs = []
        FLAGS,flag = [], 0
        all_ids = []

        for i, d in enumerate(loader): 
            if i in interest_ids and i in explain_ids: 

                sparsity = args.sparsity
                thres = args.plot_thres
                d = d.to(device)
                logits = gnn_model(d)[0]
                if torch.argmax(logits) != int(d.y): continue

                x = F.normalize(d.x, p=2.0, dim=-1).double()
                if True: 
                    h1 = activation['conv1']
                    h2 = activation['convs.0']
                    h3 = activation['convs.1']
                    lin1out = activation['lin1']
                    lin2out = activation['lin2']
                    h1_a = (activation['conv1']>0).double()
                    h2_a = (activation['convs.0']>0).double()
                    h3_a = (activation['convs.1']>0).double()
                    lin1out_a = (activation['lin1']>0).double()
                    denom_1 = (h1*h1_a).norm(p, dim, keepdim=True).clamp_min(eps).expand_as(h1*h1_a)  
                    denom_2 = (h2*h2_a).norm(p, dim, keepdim=True).clamp_min(eps).expand_as(h2*h2_a)

                    denoms = [denom_1, denom_2]
                    edges = np.transpose(np.asarray(d.edge_index.cpu()))
                    act_p = [h1_a, h2_a, h3_a, lin1out_a]
                    h_logits = [h1, h2, h3, lin1out, lin2out]
                    denoms = [den.double() for den in denoms]
                    h_logits = [hlo.double() for hlo in h_logits]

                    # offset logits
                    off_edge_index = torch.tensor([[],[]]).to(device).long()
                    off_logits = gnn_model.fwd_base(d.x, off_edge_index)[0]
                    h3_a_off = (activation['convs.1']>0).double()
                    lin1out_a_off = (activation['lin1']>0).double()
                    off_act_p = [h3_a_off, lin1out_a_off]

                # second-hop and third-hop edges
                first_hop_edges={a:d.edge_index[:, torch.nonzero(d.edge_index[0]==a).view(-1).tolist()] for a in range(x.shape[0])}
                second_hop_edges={a:0 for a in range(x.shape[0])}
                third_hop_edges={a:0 for a in range(x.shape[0])}
                for a in range(x.shape[0]):
                    all_b = list(set(first_hop_edges[a][1].view(-1).tolist()))
                    second_hop_edges[a]=next_hop_neigh(d.edge_index, all_lefts=all_b, all_rights=None)
                    all_c = list(set(second_hop_edges[a][1].view(-1).tolist()))
                    third_hop_edges[a]=next_hop_neigh(d.edge_index, all_lefts=all_c, all_rights=None)
                nhop_edges = [first_hop_edges, second_hop_edges, third_hop_edges]

                LAYER = 3
                edge_overall_score = {edge2key(a,b,x.shape[0]):0 for (a,b) in edges} 
                for (a,b) in edges:
                    ekey = edge2key(a,b,x.shape[0])
                    for loc in range(LAYER):
                        out = gcn_layer_prop(loc, (a,b), gnn_model.conv1.propagate, d.edge_index, x, weights, denoms, act_p)
                        edge_overall_score[ekey] += out[-1]/(2*LAYER+1)
                        for k in range(LAYER-1):
                            if loc > k:
                                out = gcn_bias_prop(k, loc, (a,b), gnn_model.conv1.propagate, d.edge_index, x, weights, bias, denoms, act_p)
                                edge_overall_score[ekey] += out[-1]/max(1,2*(LAYER-k))

                for act_id in range(LAYER):
                    for a in range(x.shape[0]):
                        tmp_act = torch.zeros(act_p[act_id].shape).to(device).double()
                        tmp_act[a] = copy.deepcopy(act_p[act_id][a])
                        ap = copy.deepcopy(act_p)
                        ap[act_id] = tmp_act
                        out = gcn_layer_prop(None, None, gnn_model.conv1.propagate, d.edge_index, x, weights, denoms, ap)
                        all_edges = []
                        for l in range(act_id+1):
                            all_edges += np.transpose(np.asarray(nhop_edges[l][a].cpu())).tolist()
                        for (left, right) in all_edges:
                            ekey = edge2key(left,right,x.shape[0])
                            edge_overall_score[ekey] += out[-1]/(2*LAYER+1)/len(all_edges)
                        for k in range(LAYER):
                            if act_id >= k:
                                out = gcn_bias_prop(k, None, None, gnn_model.conv1.propagate, d.edge_index, x, weights, bias, denoms, ap)
                                for (left, right) in all_edges:
                                    ekey = edge2key(left,right,x.shape[0])
                                    edge_overall_score[ekey] += out[-1]/max(1,2*(LAYER-k))/len(all_edges)

                for act_id in range(LAYER, len(act_p)):
                    out = gcn_layer_prop(None, None, gnn_model.conv1.propagate, d.edge_index, x, weights, denoms, act_p)
                    for ekey in edge_overall_score.keys():
                        edge_overall_score[ekey] += out[-1]/(2*LAYER+1)/d.edge_index.shape[1]                        
                    for k in range(len(act_p)):
                        if act_id >= k:
                            out = gcn_bias_prop(k, None, None, gnn_model.conv1.propagate, d.edge_index, x, weights, bias, denoms, act_p)
                            for ekey in edge_overall_score.keys():                        
                                edge_overall_score[ekey] += out[-1]/max(1,2*(LAYER-k))/d.edge_index.shape[1]                      

                # '''     
                # substract offsets -- last conv layer
                act_id = 2
                k=LAYER-1
                for a in range(x.shape[0]):
                    tmp_act = torch.zeros(off_act_p[0].shape).to(device).double()
                    tmp_act[a] = copy.deepcopy(off_act_p[0][a])
                    ap = copy.deepcopy(act_p)
                    ap[LAYER-1] = tmp_act
                    ap[LAYER] = copy.deepcopy(off_act_p[1])
                    all_edges = []
                    for l in range(act_id+1):
                        all_edges += np.transpose(np.asarray(nhop_edges[l][a].cpu())).tolist()
                    if act_id >= k:
                        out = gcn_bias_prop(LAYER-1, None, None, gnn_model.conv1.propagate, d.edge_index, x, weights, bias, denoms, ap)
                        for (left, right) in all_edges:
                            ekey = edge2key(left,right,x.shape[0])
                            edge_overall_score[ekey] -= out[-1]/max(1,2*(LAYER-k))/len(all_edges)
                
                # substract offsets -- first classify layer
                for act_id in range(LAYER, len(act_p)):
                    ap = copy.deepcopy(act_p)
                    ap[LAYER-1] = copy.deepcopy(off_act_p[0])
                    ap[LAYER] = copy.deepcopy(off_act_p[1])
                    for k in range(LAYER-1, len(act_p)):
                        if act_id >= k:
                            out = gcn_bias_prop(k, None, None, gnn_model.conv1.propagate, d.edge_index, x, weights, bias, denoms, ap)
                            for ekey in edge_overall_score.keys():
                                edge_overall_score[ekey] -= out[-1]/max(1,2*(LAYER-k))/d.edge_index.shape[1]
                # '''

                overall_score = torch.stack([score.view(-1) for score in edge_overall_score.values()])

                if args.is_undirected>0:
                    undi_edge_overall_score = copy.deepcopy(overall_score)
                    for ei in range(d.edge_index.shape[1]):
                        el, er = d.edge_index[:,ei]
                        el, er =int(el), int(er)
                        rev_e = int((torch.logical_and(d.edge_index[0]==er, d.edge_index[1]==el)==True).nonzero()[0])
                        undi_edge_overall_score[ei]+= overall_score[rev_e]
                        undi_edge_overall_score[ei]*=0.5
                    overall_score = undi_edge_overall_score

                # overall_score = torch.norm(overall_score, dim=-1)
                # overall_score = overall_score[:,int(d.y)]
                num_edges = max(2, int(d.edge_index.shape[1]*(1.0-sparsity)))

                goal_class = int(d.y)
                # goal_class = -1*int(d.y)+1
                if args.detect_not>0:
                    i_score = torch.abs(overall_score[:,goal_class])
                    Hedges = torch.topk(i_score, num_edges, dim=-1)[1].cpu().detach().numpy()
                    econfi = overall_score[:,goal_class][Hedges].cpu().detach().numpy()

                    pos_Hedges = Hedges[econfi>=thres]
                    neg_Hedges = Hedges[econfi<thres]
                    if len(pos_Hedges)<2: flag = 1
                    else: flag = 0
                else:
                    i_score = overall_score[:,goal_class]
                    # i_score = torch.norm(overall_score, dim=-1)
                    econfi, Hedges = torch.topk(i_score, num_edges, dim=-1)[0].cpu().detach().numpy(), torch.topk(i_score, num_edges, dim=-1)[1].cpu().detach().numpy()

                if args.linear_search>0 and num_edges>3:
                    if args.detect_not>0:
                        if flag==0:
                            _Hedges=pos_Hedges
                        else:
                            _Hedges=neg_Hedges
                    else:
                        _Hedges=Hedges
                    diffs = []
                    
                    for l in range(1, len(_Hedges)-2, 2):
                        f_neg, f_pos = efidelity(_Hedges[:l+3], gnn_model, d, device)
                        # if flag==0: diff=f_pos[1]
                        # else: diff=f_neg[1]
                        if flag==0: diff=f_pos[1]-f_neg[1]
                        else: diff = f_neg[1]-f_pos[1]
                        diffs.append(diff)
                        
                        if args.do_plot:
                            print(d.edge_index[:,_Hedges[:l+3]])
                            print(diff,"\n")
                        if diff > 0.999: break

                    best_index = diffs.index(max(diffs))
                    # _Hedges = _Hedges[:best_index+2]
                    _Hedges = _Hedges[:2*(best_index+2)]
                    
                    _econfi = overall_score[:,int(d.y)][_Hedges].cpu().detach().numpy()
                else: _Hedges=Hedges

                if args.do_evaluate>0:
                    # f_neg, f_pos = efidelity(Hedges[:12], gnn_model, d, device)
                    if flag==0:
                        f_neg, f_pos = efidelity(_Hedges, gnn_model, d, device)
                    else:
                        f_pos, f_neg = efidelity(_Hedges, gnn_model, d, device)
                    Fidelities.append(f_pos[1])
                    neg_fids.append(f_neg[1])
                    if flag==0:
                        A_sparsities.append(1.0-float(len(_Hedges)/d.edge_index.shape[1]))
                    else: A_sparsities.append(float(len(_Hedges)/d.edge_index.shape[1]))
                    print(i, sum(neg_fids)/float(len(neg_fids)+1e-13), sum(Fidelities)/float(len(Fidelities)+1e-13), sum(A_sparsities)/float(len(A_sparsities)+1e-13))

                if args.do_plot>0:
                    # if len(exp_embs)<args.clusters: continue
                    print(f'off_logits: {off_logits}')
                    print(f'global bias: {lin2b}')
                    print(f'prediction logits: {lin2out}')
                    print(f'total edge score: {sum(edge_overall_score.values())+off_logits}')
                    print(f'threshold: {thres}')
                    print(econfi)
                    print(d.edge_index[:,Hedges])
                    if args.detect_not>0:
                        GC_vis_graph(d.x, d.edge_index, Hedges=_Hedges, good_nodes=None, datasetname=dataname)
                        GC_vis_graph(d.x, d.edge_index, Hedges=neg_Hedges, good_nodes=None, datasetname=dataname, edge_color='blue')
                    else:
                        GC_vis_graph(d.x, d.edge_index, Hedges=_Hedges, good_nodes=None, datasetname=dataname)
                    show()


    # print(f'Avg time: {sum(Times)/float(len(Times))}')
    print(f"Fidelity-: {sum(neg_fids)/float(len(neg_fids)+1e-13)}")
    print(f"Fidelity+: {sum(Fidelities)/float(len(Fidelities)+1e-13)}")
    print(f"Actual avg sparsity: {sum(A_sparsities)/float(len(A_sparsities)+1e-13)}")

def gcn_layer_prop(loc, hidden, propagate, edge_index, x, weights, denoms, act_p):
    edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device).double()
    if isinstance(hidden, tuple): 
        if loc == 0:
            (a,b) = hidden
            tmp_h1_bef = F.linear(x[b,:].view(1,-1),weights[0])
            tmp_h1 = torch.zeros(act_p[0].shape).to(x.device)
            tmp_h1[a] = tmp_h1_bef.double()*act_p[0][a].double()/denoms[0][a].double()
            tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1].double())
            tmp_h2 = tmp_h2_bef.double()*act_p[1]/denoms[1].double()
            tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2.double(), edge_weight=edge_weight),weights[2].double())
            tmp_h3 = tmp_h3_bef*act_p[2]
            batch = torch.zeros(x.shape[0]).to(x.device).long()
            tmp_pool = global_mean_pool(tmp_h3, batch)
            tmplin1_bef = F.linear(tmp_pool,weights[3].double())
            tmplin1 = tmplin1_bef*act_p[3].double()
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1,weights[4])
        elif loc == 1:
            tmp_h1_bef = F.linear(propagate(edge_index, x=x, edge_weight=edge_weight),weights[0])
            tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
            (a,b) = hidden
            tmp_h2_bef = F.linear(tmp_h1[b,:].view(1,-1),weights[1])
            tmp_h2 = torch.zeros(act_p[1].shape).to(x.device)
            tmp_h2[a] = tmp_h2_bef*act_p[1][a]/denoms[1][a]
            tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
            tmp_h3 = tmp_h3_bef*act_p[2]
            batch = torch.zeros(x.shape[0]).to(x.device).long()
            tmp_pool = global_mean_pool(tmp_h3, batch)
            tmplin1_bef = F.linear(tmp_pool,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1,weights[4])
        elif loc == 2:
            tmp_h1_bef = F.linear(propagate(edge_index, x=x, edge_weight=edge_weight),weights[0])
            tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
            tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1])
            tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
            (a,b) = hidden
            tmp_h3_bef = F.linear(tmp_h2[b,:].view(1,-1),weights[2])
            tmp_h3 = torch.zeros(act_p[2].shape).to(x.device)
            tmp_h3[a] = tmp_h3_bef*act_p[2][a]
            batch = torch.zeros(x.shape[0]).to(x.device).long()
            tmp_pool = global_mean_pool(tmp_h3.double(), batch).double()
            tmplin1_bef = F.linear(tmp_pool,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1,weights[4])
    else:
        tmp_h1_bef = F.linear(propagate(edge_index, x=x.double(), edge_weight=edge_weight),weights[0])
        tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
        tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1])
        tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
        tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
        tmp_h3 = tmp_h3_bef*act_p[2]
        batch = torch.zeros(x.shape[0]).to(x.device).long()
        tmp_pool = global_mean_pool(tmp_h3, batch).double()
        tmplin1_bef = F.linear(tmp_pool,weights[3])
        tmplin1 = tmplin1_bef*act_p[3]
        tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
        tmplin2 = F.linear(tmplin1,weights[4])
    return [tmp_h1_bef, tmp_h2_bef, tmp_h3_bef, tmplin1_bef, tmplin2]

def gcn_bias_prop(layer, loc, hidden, propagate, edge_index, x, weights, bias, denoms, act_p):
    edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device).double()
    if layer==0:
        if isinstance(hidden, tuple): 
            (a,b) = hidden
            tmp_h1_bef = bias[0].repeat(x.shape[0],1).double()
            tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
            if loc==1:
                tmp_h2_bef = F.linear(tmp_h1[b,:].view(1,-1),weights[1])
                tmp_h2 = torch.zeros(act_p[1].shape).to(x.device)
                tmp_h2[a] = tmp_h2_bef*act_p[1][a]/denoms[1][a]
                tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
                tmp_h3 = tmp_h3_bef*act_p[2]
            elif loc==2:
                tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1])
                tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
                tmp_h3_bef = F.linear(tmp_h2[b,:].view(1,-1),weights[2])
                tmp_h3 = torch.zeros(act_p[2].shape).to(x.device)
                tmp_h3[a] = tmp_h3_bef*act_p[2][a]
            batch = torch.zeros(x.shape[0]).to(x.device).long()
            tmp_pool = global_mean_pool(tmp_h3.double(), batch).double()
            tmplin1_bef = F.linear(tmp_pool,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1,weights[4])
        else:
            tmp_h1_bef = bias[0].repeat(x.shape[0],1).double()
            tmp_h1 = tmp_h1_bef*act_p[0]/denoms[0]
            tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight),weights[1])
            tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
            tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
            tmp_h3 = tmp_h3_bef*act_p[2]
            batch = torch.zeros(x.shape[0]).to(x.device).long()
            tmp_pool = global_mean_pool(tmp_h3, batch).double()
            tmplin1_bef = F.linear(tmp_pool,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1,weights[4])
        return [tmp_h1_bef, tmp_h2_bef, tmp_h3_bef, tmplin1_bef, tmplin2]
    elif layer==1:
        if isinstance(hidden, tuple): 
            (a,b) = hidden
            if loc==2:
                tmp_h2_bef = bias[1].repeat(x.shape[0],1).double()
                tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
                tmp_h3_bef = F.linear(tmp_h2[b,:].view(1,-1),weights[2])
                tmp_h3 = torch.zeros(act_p[2].shape).to(x.device)
                tmp_h3[a] = tmp_h3_bef*act_p[2][a]
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_pool = global_mean_pool(tmp_h3, batch).double()
                tmplin1_bef = F.linear(tmp_pool,weights[3])
                tmplin1 = tmplin1_bef*act_p[3]
                tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
                tmplin2 = F.linear(tmplin1,weights[4])
            else: 
                print("ERROR: not loc==2. ")
                exit(0)
        else:
            tmp_h2_bef = bias[1].repeat(x.shape[0],1).double()
            tmp_h2 = tmp_h2_bef*act_p[1]/denoms[1]
            tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight),weights[2])
            tmp_h3 = tmp_h3_bef*act_p[2]
            batch = torch.zeros(x.shape[0]).to(x.device).long()
            tmp_pool = global_mean_pool(tmp_h3, batch)
            tmplin1_bef = F.linear(tmp_pool,weights[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1,weights[4])
        return [None, tmp_h2_bef, tmp_h3_bef, tmplin1_bef, tmplin2]
    elif layer==2:
        tmp_h3_bef = bias[2].repeat(x.shape[0],1).double()
        tmp_h3 = tmp_h3_bef*act_p[2]
        batch = torch.zeros(x.shape[0]).to(x.device).long()
        tmp_pool = global_mean_pool(tmp_h3, batch)
        tmplin1_bef = F.linear(tmp_pool,weights[3])
        tmplin1 = tmplin1_bef*act_p[3]
        tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
        tmplin2 = F.linear(tmplin1,weights[4])
        return [None, None, tmp_h3_bef, tmplin1_bef, tmplin2]
    elif layer==3:
        tmplin1_bef = bias[3].view(1,-1).double()
        tmplin1 = tmplin1_bef*act_p[3]
        tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
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

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ba_2motifs')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--sparsity', type=float, default=0.7)
    parser.add_argument('--do_plot', type=int, default=0)
    parser.add_argument('--do_evaluate', type=int, default=1)
    parser.add_argument('--global_exp', type=int, default=0)
    parser.add_argument('--clusters', type=int, default=15)
    parser.add_argument('--is_undirected', type=int, default=1)
    parser.add_argument('--detect_not', type=int, default=0)

    parser.add_argument('--plot_thres', type=float, default=-00.0)
    parser.add_argument('--linear_search', type=int, default=1)
    
    return parser.parse_args()

if __name__ == "__main__":

    args = build_args()
    main(args)
    print("done")




    