
import argparse
import torch
from torch_geometric.loader import DataLoader
import numpy as np

from Utils.utils import check_task, load_model, detect_exp_setting, detect_motif_nodes, show, GC_vis_graph
from Utils.metrics import efidelity

from Utils.datasets import get_dataset, get_graph_data
import torch.nn.functional as F
import copy
from torch_geometric.nn import global_mean_pool
from itertools import product

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

            h1wl = model_params['conv1.lin_l.weight']
            h1wr = model_params['conv1.lin_r.weight']
            h2wl = model_params['convs.0.lin_l.weight']
            h2wr = model_params['convs.0.lin_r.weight']
            h3wl = model_params['convs.1.lin_l.weight']
            h3wr = model_params['convs.1.lin_r.weight']
            lin1w = model_params['lin1.weight']
            lin2w = model_params['lin2.weight']

            h1b = model_params['conv1.lin_l.bias']
            h2b = model_params['convs.0.lin_l.bias']
            h3b = model_params['convs.1.lin_l.bias']
            lin1b = model_params['lin1.bias']
            lin2b = model_params['lin2.bias']

            weights_l = [h1wl, h2wl, h3wl, lin1w, lin2w]
            weights_r = [h1wr, h2wr, h3wr]
            bias = [h1b, h2b, h3b, lin1b, lin2b]
            weights_l = [weight for weight in weights_l]
            weights_r = [weight for weight in weights_r]
            bias = [bb for bb in bias]

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

        flag = 0
        for i, d in enumerate(loader): 
            if i in interest_ids and i in explain_ids: 
                # if i<500: continue
                sparsity = args.sparsity
                thres = args.plot_thres
                d = d.to(device)
                logits = gnn_model(d)[0]
                if torch.argmax(logits) != int(d.y): continue

                x = d.x

                if True:
                    lin2out = gnn_model.fwd_base(d.x, d.edge_index)[0]
                    h1_a = (activation['conv1']>0).float()
                    h2_a = (activation['convs.0']>0).float()
                    h3_a = (activation['convs.1']>0).float()
                    lin1out_a = (activation['lin1']>0).float()
                    edges = np.transpose(np.asarray(d.edge_index.cpu()))
                    act_p = [h1_a, h2_a, h3_a, lin1out_a]

                    # offset logits
                    off_edge_index = torch.tensor([[],[]]).to(device).long()
                    off_logits = gnn_model.fwd_base(d.x, off_edge_index)[0]
                    # print(f'off_logits: {off_logits}')
                    
                    zerox = torch.zeros(d.x.shape).to(d.x.device)
                    h1_a_off = (activation['conv1']>0).float()
                    h2_a_off = (activation['convs.0']>0).float()
                    h3_a_off = (activation['convs.1']>0).float()
                    lin1out_a_off = (activation['lin1']>0).float()
                    off_act_p = [h1_a_off, h2_a_off, h3_a_off, lin1out_a_off]

                # second-hop and third-hop edges
                first_hop_edges={a:d.edge_index[:, torch.nonzero(d.edge_index[0]==a).view(-1).tolist()] for a in range(x.shape[0])}
                second_hop_edges={a:0 for a in range(d.x.shape[0])}
                third_hop_edges={a:0 for a in range(d.x.shape[0])}
                for a in range(d.x.shape[0]):
                    all_b = list(set(first_hop_edges[a][1].view(-1).tolist()))
                    second_hop_edges[a]=next_hop_neigh(d.edge_index, all_lefts=all_b, all_rights=None)
                    all_c = list(set(second_hop_edges[a][1].view(-1).tolist()))
                    third_hop_edges[a]=next_hop_neigh(d.edge_index, all_lefts=all_c, all_rights=None)
                nhop_edges = [first_hop_edges, second_hop_edges, third_hop_edges]

                LAYER = 3
                edge_overall_score = {edge2key(a,b,d.x.shape[0]):0 for (a,b) in edges}
                for (a,b) in edges:
                    ekey = edge2key(a,b,x.shape[0])
                    for loc in range(LAYER):
                        out = sage_layer_prop(loc, (a,b), gnn_model.conv1.propagate, d.edge_index, x, weights_l, weights_r, act_p)
                        edge_overall_score[ekey] += out
                        for k in range(LAYER-1):
                            if loc > k:
                                out = sage_bias_prop(k, loc, (a,b), gnn_model.conv1.propagate, d.edge_index, x, weights_l, weights_r, bias, act_p)
                                edge_overall_score[ekey] += out
                
                for act_id in range(LAYER):
                    for a in range(x.shape[0]):
                        tmp_act = torch.zeros(act_p[act_id].shape).to(device)
                        tmp_act[a] = copy.deepcopy(act_p[act_id][a])
                        ap = copy.deepcopy(act_p)
                        ap[act_id] = tmp_act
                        out = sage_layer_prop(None, None, gnn_model.conv1.propagate, d.edge_index, x,weights_l, weights_r, ap)
                        all_edges = []
                        for l in range(act_id+1):
                            all_edges += np.transpose(np.asarray(nhop_edges[l][a].cpu())).tolist()
                        for (left, right) in all_edges:
                            ekey = edge2key(left,right,x.shape[0])
                            edge_overall_score[ekey] += out/len(all_edges)
                        for k in range(LAYER):
                            if act_id >= k:
                                out = sage_bias_prop(k, None, None, gnn_model.conv1.propagate, d.edge_index, x, weights_l, weights_r, bias, ap)
                                for (left, right) in all_edges:
                                    ekey = edge2key(left,right,x.shape[0])
                                    edge_overall_score[ekey] += out/len(all_edges)

                for act_id in range(LAYER, len(act_p)):
                    out = sage_layer_prop(None, None, gnn_model.conv1.propagate, d.edge_index, x, weights_l, weights_r, act_p)
                    for ekey in edge_overall_score.keys():
                        edge_overall_score[ekey] += out/d.edge_index.shape[1]                        
                    for k in range(len(act_p)):
                        if act_id >= k:
                            out = sage_bias_prop(k, None, None, gnn_model.conv1.propagate, d.edge_index, x, weights_l, weights_r, bias, act_p)
                            for ekey in edge_overall_score.keys():                        
                                edge_overall_score[ekey] += out/d.edge_index.shape[1]

                # '''
                # substract offsets -- conv layers
                for act_id in range(LAYER):
                    for a in range(x.shape[0]):
                        tmp_act = torch.zeros(off_act_p[act_id].shape).to(device)
                        tmp_act[a] = copy.deepcopy(off_act_p[act_id][a])
                        ap = copy.deepcopy(off_act_p)
                        ap[act_id] = tmp_act
                        all_edges = []
                        for l in range(act_id+1):
                            all_edges += np.transpose(np.asarray(nhop_edges[l][a].cpu())).tolist()
                        out = sage_layer_prop(None, None, gnn_model.conv1.propagate, d.edge_index, x, weights_l, weights_r, ap, 1)
                        for (left, right) in all_edges:
                            ekey = edge2key(left,right,x.shape[0])
                            edge_overall_score[ekey] -= out/len(all_edges)
                        for k in range(act_id+1):
                            if act_id >= k:
                                out = sage_bias_prop(k, None, None, gnn_model.conv1.propagate, d.edge_index, x, weights_l, weights_r, bias, ap, 1)
                                for (left, right) in all_edges:
                                    ekey = edge2key(left,right,x.shape[0])
                                    edge_overall_score[ekey] -= out/len(all_edges)
                
                # substract offsets -- first classify layer
                for act_id in range(LAYER, len(act_p)):
                    out = sage_layer_prop(None, None, gnn_model.conv1.propagate, d.edge_index, x, weights_l, weights_r, off_act_p, 1)
                    for ekey in edge_overall_score.keys():
                        edge_overall_score[ekey] -= out/d.edge_index.shape[1]
                    for k in range(act_id+1):
                        if act_id >= k:
                            out = sage_bias_prop(k, None, None, gnn_model.conv1.propagate, d.edge_index, x, weights_l, weights_r, bias, off_act_p, 1)
                            for ekey in edge_overall_score.keys():
                                edge_overall_score[ekey] -= out/d.edge_index.shape[1]
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
                
                num_edges = max(2, 2*int(d.edge_index.shape[1]//2*(1.0-sparsity)))

                if args.detect_not>0:
                    i_score = torch.abs(overall_score[:,int(d.y)])
                    Hedges = torch.topk(i_score, num_edges, dim=-1)[1].cpu().detach().numpy()
                    econfi = overall_score[:,int(d.y)][Hedges].cpu().detach().numpy()

                    pos_Hedges = Hedges[econfi>=thres]
                    neg_Hedges = Hedges[econfi<thres]
                    if len(pos_Hedges)<2: flag = 1
                    else: flag = 0
                else:
                    i_score = overall_score[:,int(d.y)]
                    econfi, Hedges = torch.topk(i_score, num_edges, dim=-1)[0].cpu().detach().numpy(), torch.topk(i_score, num_edges, dim=-1)[1].cpu().detach().numpy()

                if args.linear_search>0:
                    if args.detect_not>0:
                        if flag==0:
                            _Hedges=pos_Hedges
                        else:
                            _Hedges=neg_Hedges
                    else:
                        _Hedges=Hedges
                    diffs = []
                
                    for l in range(1, len(_Hedges), 2):
                        f_neg, f_pos = efidelity(_Hedges[:l+1], gnn_model, d, device)
                        # if flag==0: diffs.append(f_pos[1])
                        # else: diffs.append(f_neg[1])
                        if flag==0: diffs.append(f_pos[1]-f_neg[1])
                        else: diffs.append(f_neg[1]-f_pos[1])
                    best_index = diffs.index(max(diffs))
                    _Hedges = _Hedges[:2*(best_index+1)]
                    _econfi = overall_score[:,int(d.y)][_Hedges].cpu().detach().numpy()
                else: _Hedges=Hedges

                if args.do_evaluate>0:
                    if flag==0:
                        f_neg, f_pos = efidelity(_Hedges, gnn_model, d, device)
                    else:
                        f_pos, f_neg = efidelity(_Hedges, gnn_model, d, device)
                    Fidelities.append(f_pos[1])
                    neg_fids.append(f_neg[1])
                    if flag==0:
                        A_sparsities.append(1.0-float(len(_Hedges)/d.edge_index.shape[1]))
                    else: A_sparsities.append(float(len(_Hedges)/d.edge_index.shape[1]))
                    # print(i, int(torch.argmax(logits)), int(d.y), f_neg[1],f_pos[1])
                    print(i, sum(neg_fids)/float(len(neg_fids)+1e-13), sum(Fidelities)/float(len(Fidelities)+1e-13), sum(A_sparsities)/float(len(A_sparsities)+1e-13))

                if args.do_plot>0:
                    print(i, int(torch.argmax(logits)), int(d.y), f_neg[1],f_pos[1])
                    print(f'off_logits: {off_logits}')
                    print(f'global bias: {lin2b}')
                    print(f'prediction logits: {lin2out}')
                    print(f'total edge score: {sum(edge_overall_score.values())}')
                    print(f'threshold: {thres}')
                    print(econfi)
                    print(d.edge_index[:,Hedges])
                    if args.detect_not>0:
                        GC_vis_graph(d.x, d.edge_index, Hedges=_Hedges, good_nodes=None, datasetname=dataname)
                        if flag ==0:
                            GC_vis_graph(d.x, d.edge_index, Hedges=neg_Hedges, good_nodes=None, datasetname=dataname, edge_color='blue')
                        else:
                            GC_vis_graph(d.x, d.edge_index, Hedges=_Hedges, good_nodes=None, datasetname=dataname, edge_color='blue')
                    else:
                        GC_vis_graph(d.x, d.edge_index, Hedges=_Hedges, good_nodes=None, datasetname=dataname)
                    show()


def sage_layer_prop(loc, hidden, propagate, edge_index, x, weights_l, weights_r, act_p, offset_flag=0):
    terms = product(range(2),repeat=3)
    if offset_flag==1: terms = [[1, 1, 1]]
    out = 0
    if isinstance(hidden, tuple): 
        (a,b) = hidden
        _edge = torch.tensor(hidden).view(2,1).long().to(x.device)
        if loc == 0:
            for term in terms:
                if term[loc]==1: continue
                if term[0]==0: tmp_h1_bef = F.linear(propagate(_edge, x=x, size=None), weights_l[0])
                else: tmp_h1_bef = F.linear(x, weights_r[0])
                tmp_h1 = tmp_h1_bef*act_p[0]
                if term[1]==0: tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, size=None),weights_l[1])
                else: tmp_h2_bef = F.linear(tmp_h1, weights_r[1])
                tmp_h2 = tmp_h2_bef*act_p[1]
                if term[2]==0: tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, size=None),weights_l[2])
                else: tmp_h3_bef = F.linear(tmp_h2, weights_r[2])
                tmp_h3 = tmp_h3_bef*act_p[2]
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_pool = global_mean_pool(tmp_h3, batch)
                tmplin1_bef = F.linear(tmp_pool,weights_l[3])
                tmplin1 = tmplin1_bef*act_p[3]
                tmplin2 = F.linear(tmplin1,weights_l[4])
                out += tmplin2/(len(act_p)+len(term)-sum(term))
        elif loc ==1:
            for term in terms:
                if term[loc]==1: continue
                if term[0]==0: tmp_h1_bef = F.linear(propagate(edge_index, x=x, size=None), weights_l[0])
                else: tmp_h1_bef = F.linear(x, weights_r[0])
                tmp_h1 = tmp_h1_bef*act_p[0]
                if term[1]==0: tmp_h2_bef = F.linear(propagate(_edge, x=tmp_h1, size=None),weights_l[1])
                else: tmp_h2_bef = F.linear(tmp_h1, weights_r[1])
                tmp_h2 = tmp_h2_bef*act_p[1]
                if term[2]==0: tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, size=None),weights_l[2])
                else: tmp_h3_bef = F.linear(tmp_h2, weights_r[2])
                tmp_h3 = tmp_h3_bef*act_p[2]
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_pool = global_mean_pool(tmp_h3, batch)
                tmplin1_bef = F.linear(tmp_pool,weights_l[3])
                tmplin1 = tmplin1_bef*act_p[3]
                tmplin2 = F.linear(tmplin1,weights_l[4])
                out += tmplin2/(len(act_p)+len(term)-sum(term))
        elif loc == 2:
            for term in terms:
                if term[loc]==1: continue
                if term[0]==0: tmp_h1_bef = F.linear(propagate(edge_index, x=x, size=None), weights_l[0])
                else: tmp_h1_bef = F.linear(x, weights_r[0])
                tmp_h1 = tmp_h1_bef*act_p[0]
                if term[1]==0: tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, size=None),weights_l[1])
                else: tmp_h2_bef = F.linear(tmp_h1, weights_r[1])
                tmp_h2 = tmp_h2_bef*act_p[1]
                if term[2]==0: tmp_h3_bef = F.linear(propagate(_edge, x=tmp_h2, size=None),weights_l[2])
                else: tmp_h3_bef = F.linear(tmp_h2, weights_r[2])
                tmp_h3 = tmp_h3_bef*act_p[2]
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_pool = global_mean_pool(tmp_h3, batch)
                tmplin1_bef = F.linear(tmp_pool,weights_l[3])
                tmplin1 = tmplin1_bef*act_p[3]
                tmplin2 = F.linear(tmplin1,weights_l[4])
                out += tmplin2/(len(act_p)+len(term)-sum(term))
    else:
        for term in terms:
            if term[0]==0: tmp_h1_bef = F.linear(propagate(edge_index, x=x, size=None), weights_l[0])
            else: tmp_h1_bef = F.linear(x, weights_r[0])
            tmp_h1 = tmp_h1_bef*act_p[0]
            if term[1]==0: tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, size=None),weights_l[1])
            else: tmp_h2_bef = F.linear(tmp_h1, weights_r[1])
            tmp_h2 = tmp_h2_bef*act_p[1]
            if term[2]==0: tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, size=None),weights_l[2])
            else: tmp_h3_bef = F.linear(tmp_h2, weights_r[2])
            tmp_h3 = tmp_h3_bef*act_p[2]

            batch = torch.zeros(x.shape[0]).to(x.device).long()
            tmp_pool = global_mean_pool(tmp_h3, batch)
            tmplin1_bef = F.linear(tmp_pool,weights_l[3])
            tmplin1 = tmplin1_bef*act_p[3]
            tmplin2 = F.linear(tmplin1,weights_l[4])
            out += tmplin2/(len(act_p)+len(term)-sum(term))
            # out += tmplin2
    return out

def sage_bias_prop(layer, loc, hidden, propagate, edge_index, x, weights_l, weights_r, bias, act_p, offset_flag=0):
    if layer==0:
        terms = product(range(2),repeat=2)
        if offset_flag==1: terms = [[1, 1]]
        out = 0
        if isinstance(hidden, tuple): 
            (a,b) = hidden
            _edge = torch.tensor(hidden).view(2,1).long().to(x.device)
            for term in terms:
                tmp_h1_bef = bias[0].repeat(x.shape[0],1)
                tmp_h1 = tmp_h1_bef*act_p[0]
                if loc==1:
                    if term[0]==1: continue
                    if term[0]==0: tmp_h2_bef = F.linear(propagate(_edge, x=tmp_h1, size=None),weights_l[1])
                    else: tmp_h2_bef = F.linear(tmp_h1,weights_r[1])
                    tmp_h2 = tmp_h2_bef*act_p[1]
                    if term[1]==0: tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, size=None),weights_l[2])
                    else: tmp_h3_bef = F.linear(tmp_h2,weights_r[2])
                    tmp_h3 = tmp_h3_bef*act_p[2]
                elif loc==2:
                    if term[1]==1: continue
                    if term[0]==0: tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, size=None),weights_l[1])
                    else: tmp_h2_bef = F.linear(tmp_h1,weights_r[1])
                    tmp_h2 = tmp_h2_bef*act_p[1]
                    if term[1]==0: tmp_h3_bef = F.linear(propagate(_edge, x=tmp_h2, size=None),weights_l[2])
                    else: tmp_h3_bef = F.linear(tmp_h2,weights_r[2])
                    tmp_h3 = tmp_h3_bef*act_p[2]
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_pool = global_mean_pool(tmp_h3, batch)
                tmplin1_bef = F.linear(tmp_pool,weights_l[3])
                tmplin1 = tmplin1_bef*act_p[3]
                
                tmplin2 = F.linear(tmplin1,weights_l[4])
                out += tmplin2/(len(act_p)-layer+len(term)-sum(term))
        else:
            for term in terms:
                tmp_h1_bef = bias[0].repeat(x.shape[0],1)
                tmp_h1 = tmp_h1_bef*act_p[0]
                if term[0]==0: tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, size=None),weights_l[1])
                else: tmp_h2_bef = F.linear(tmp_h1,weights_r[1])
                tmp_h2 = tmp_h2_bef*act_p[1]
                if term[1]==0: tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, size=None),weights_l[2])
                else: tmp_h3_bef = F.linear(tmp_h2,weights_r[2])
                tmp_h3 = tmp_h3_bef*act_p[2]
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_pool = global_mean_pool(tmp_h3, batch)
                tmplin1_bef = F.linear(tmp_pool,weights_l[3])
                tmplin1 = tmplin1_bef*act_p[3]
                tmplin2 = F.linear(tmplin1,weights_l[4])
                out += tmplin2/(len(act_p)-layer+len(term)-sum(term))
                # out += tmplin2
        return out 
    elif layer==1:
        terms = product(range(2),repeat=1)
        if offset_flag==1: terms = [[1]]
        out = 0
        if isinstance(hidden, tuple): 
            (a,b) = hidden
            _edge = torch.tensor(hidden).view(2,1).long().to(x.device)
            for term in terms:
                if term[0]==1: continue
                tmp_h2_bef = bias[1].repeat(x.shape[0],1)
                tmp_h2 = tmp_h2_bef*act_p[1]
                if term[0]==0: tmp_h3_bef = F.linear(propagate(_edge, x=tmp_h2, size=None),weights_l[2])
                else: tmp_h3_bef = F.linear(tmp_h2,weights_r[2])
                tmp_h3 = tmp_h3_bef*act_p[2]
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_pool = global_mean_pool(tmp_h3, batch)
                tmplin1_bef = F.linear(tmp_pool,weights_l[3])
                tmplin1 = tmplin1_bef*act_p[3]
                
                tmplin2 = F.linear(tmplin1,weights_l[4])
                out += tmplin2/(len(act_p)-layer+len(term)-sum(term))
        else:
            for term in terms:
                tmp_h2_bef = bias[1].repeat(x.shape[0],1)
                tmp_h2 = tmp_h2_bef*act_p[1]
                if term[0]==0: tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, size=None),weights_l[2])
                else: tmp_h3_bef = F.linear(tmp_h2,weights_r[2])
                tmp_h3 = tmp_h3_bef*act_p[2]
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_pool = global_mean_pool(tmp_h3, batch)
                tmplin1_bef = F.linear(tmp_pool,weights_l[3])
                tmplin1 = tmplin1_bef*act_p[3]
                tmplin2 = F.linear(tmplin1,weights_l[4])
                out += tmplin2/(len(act_p)-layer+len(term)-sum(term))
                # out += tmplin2
        return out
    elif layer==2:
        tmp_h3_bef = bias[2].repeat(x.shape[0],1)
        tmp_h3 = tmp_h3_bef*act_p[2]
        batch = torch.zeros(x.shape[0]).to(x.device).long()
        tmp_pool = global_mean_pool(tmp_h3, batch)
        tmplin1_bef = F.linear(tmp_pool,weights_l[3])
        tmplin1 = tmplin1_bef*act_p[3]        
        tmplin2 = F.linear(tmplin1,weights_l[4])
        return tmplin2/(len(act_p)-layer)
        # return tmplin2
    elif layer==3:
        tmplin1_bef = bias[3].view(1,-1)
        tmplin1 = tmplin1_bef*act_p[3]        
        tmplin2 = F.linear(tmplin1,weights_l[4])
        return tmplin2/(len(act_p)-layer)
        # return tmplin2

def sage_reimp(propagate, edge_index, x, weights_l, weights_r, bias, act_p):
    x1 = F.linear(propagate(edge_index, x=x, size=None), weights_l[0])
    x1 += F.linear(x, weights_r[0]) + bias[0]
    x1 = x1*act_p[0]
    x2 = F.linear(propagate(edge_index, x=x1, size=None), weights_l[1])
    x2 += F.linear(x1, weights_r[1]) + bias[1]
    x2 = x2*act_p[1]
    x3 = F.linear(propagate(edge_index, x=x2, size=None), weights_l[2])
    x3 += F.linear(x2, weights_r[2]) + bias[2]
    x3 = x3*act_p[2]
    batch = torch.zeros(x.shape[0]).to(x.device).long()
    out = global_mean_pool(x3, batch)
    out = (F.linear(out, weights_l[3]) + bias[3])*act_p[3]
    out = F.linear(out, weights_l[4]) + bias[4]
    return out

def sage_reimp_no_bias(propagate, edge_index, x, weights_l, weights_r, act_p):
    x1 = F.linear(propagate(edge_index, x=x, size=None), weights_l[0])
    x1 += F.linear(x, weights_r[0])
    x1 = x1*act_p[0]
    x2 = F.linear(propagate(edge_index, x=x1, size=None), weights_l[1])
    x2 += F.linear(x1, weights_r[1])
    x2 = x2*act_p[1]
    x3 = F.linear(propagate(edge_index, x=x2, size=None), weights_l[2])
    x3 += F.linear(x2, weights_r[2])
    x3 = x3*act_p[2]
    batch = torch.zeros(x.shape[0]).to(x.device).long()
    out = global_mean_pool(x3, batch)
    out = (F.linear(out, weights_l[3]))*act_p[3]
    out = F.linear(out, weights_l[4])
    return out

def sage_reimp_only_bias(propagate, edge_index, x, weights_l, weights_r, bias, act_p):
    x1 = bias[0].repeat(x.shape[0],1)
    x1 = x1*act_p[0]
    x2 = F.linear(propagate(edge_index, x=x1, size=None), weights_l[1])
    x2 += F.linear(x1, weights_r[1]) + bias[1]
    x2 = x2*act_p[1]
    x3 = F.linear(propagate(edge_index, x=x2, size=None), weights_l[2])
    x3 += F.linear(x2, weights_r[2]) + bias[2]
    x3 = x3*act_p[2]
    batch = torch.zeros(x.shape[0]).to(x.device).long()
    out = global_mean_pool(x3, batch)
    out = (F.linear(out, weights_l[3]) + bias[3])*act_p[3]
    out = F.linear(out, weights_l[4]) + bias[4]
    return out

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
    parser.add_argument('--gnn', type=str, default='sage')
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