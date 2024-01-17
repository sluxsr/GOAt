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

eps: float = 1e-5
momentum: float = 0.1

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

            run_means = torch.stack([gnn_model.conv1.nn[4].running_mean, gnn_model.convs[0].nn[4].running_mean, gnn_model.convs[1].nn[4].running_mean])
            run_vars = torch.stack([gnn_model.conv1.nn[4].running_var, gnn_model.convs[0].nn[4].running_var, gnn_model.convs[1].nn[4].running_var])
            bn_weights = torch.stack([model_params['conv1.nn.4.weight'], model_params['convs.0.nn.4.weight'], model_params['convs.1.nn.4.weight']])
            bn_biass = torch.stack([model_params['conv1.nn.4.bias'], model_params['convs.0.nn.4.bias'], model_params['convs.1.nn.4.bias']])
            bn_denoms = torch.sqrt(run_vars+eps)
            equiv_bn_weights = bn_weights/bn_denoms
            equiv_bn_biass = -run_means*equiv_bn_weights+bn_biass
            # equiv_bn_biass = torch.zeros((run_means*equiv_bn_weights+bn_biass).shape).to(device)

            epss = [model_params['conv1.eps'], model_params['convs.0.eps'], model_params['convs.1.eps']]
            weights = [model_params['conv1.nn.0.weight'], model_params['conv1.nn.2.weight'], equiv_bn_weights[0], \
                model_params['convs.0.nn.0.weight'], model_params['convs.0.nn.2.weight'], equiv_bn_weights[1], \
                model_params['convs.1.nn.0.weight'], model_params['convs.1.nn.2.weight'], equiv_bn_weights[2], \
                model_params['lin1.weight'], model_params['lin2.weight']]
            biass = [model_params['conv1.nn.0.bias'], model_params['conv1.nn.2.bias'], equiv_bn_biass[0], \
                model_params['convs.0.nn.0.bias'], model_params['convs.0.nn.2.bias'], equiv_bn_biass[1], \
                model_params['convs.1.nn.0.bias'], model_params['convs.1.nn.2.bias'], equiv_bn_biass[2], \
                    model_params['lin1.bias'], model_params['lin2.bias']]
            
        interest_ids = range(len(dataset))

        exp_embs = {cls:[] for cls in range(n_cls)}
        exp_Hedges = {}

        flag = 0
        for i, d in enumerate(loader): 
            if i in interest_ids and i in explain_ids: 

                sparsity = args.sparsity
                thres = args.plot_thres
                d = d.to(device)
                logits = gnn_model(d)[0]
                if torch.argmax(logits) != int(d.y): continue

                if True:
                    x, edge_index = d.x, d.edge_index
                    edges = np.transpose(np.asarray(d.edge_index.cpu()))
                    
                    lin2out = gnn_model.fwd_base(d.x, d.edge_index)[0]

                    h0_1a = (activation['conv1.nn.1']>0).float()
                    h0_3a = (activation['conv1.nn.3']>0).float()
                    h1_1a = (activation['convs.0.nn.1']>0).float()
                    h1_3a = (activation['convs.0.nn.3']>0).float()
                    h2_1a = (activation['convs.1.nn.1']>0).float()
                    h2_3a = (activation['convs.1.nn.3']>0).float()
                    lin1a = (activation['lin1']>0).float()
                    act_p = [h0_1a, h0_3a, h1_1a, h1_3a, h2_1a, h2_3a, lin1a]

                    # offset logits
                    off_edge_index = torch.tensor([[],[]]).to(device).long()
                    off_logits = gnn_model.fwd_base(d.x, off_edge_index)[0]

                    h0_1a_off = (activation['conv1.nn.1']>0).float()
                    h0_3a_off = (activation['conv1.nn.3']>0).float()
                    h1_1a_off = (activation['convs.0.nn.1']>0).float()
                    h1_3a_off = (activation['convs.0.nn.3']>0).float()
                    h2_1a_off = (activation['convs.1.nn.1']>0).float()
                    h2_3a_off = (activation['convs.1.nn.3']>0).float()
                    lin1a_off = (activation['lin1']>0).float()
                    off_act_p = [h0_1a_off, h0_3a_off, h1_1a_off, h1_3a_off, h2_1a_off, h2_3a_off, lin1a_off]

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
                # contribution of edge itself
                for (a,b) in edges:
                    ekey = edge2key(a,b,x.shape[0])
                    out = 0
                    for loc in range(LAYER):
                        out += gin_layer_prop(loc, (a,b), gnn_model.conv1.propagate, edge_index, x, epss, weights, act_p)
                        for k in range(3*(LAYER-1)):
                            if 3*loc > k:
                                out += gin_bias_prop(k, loc, (a,b), gnn_model.conv1.propagate, edge_index, x, epss, weights, biass, act_p)
                    edge_overall_score[ekey] += out

                # '''
                # contribution of activation pattern in the message-passing layers
                for act_id in range(2*LAYER):
                    for a in range(x.shape[0]):
                        tmp_act = torch.zeros(act_p[act_id].shape).to(device)
                        tmp_act[a] = copy.deepcopy(act_p[act_id][a])
                        ap = copy.deepcopy(act_p)
                        ap[act_id] = tmp_act
                        out = gin_layer_prop(None, None, gnn_model.conv1.propagate, edge_index, x, epss, weights, ap)
                        all_edges = []
                        for l in range(act_id//2+1):
                            all_edges += np.transpose(np.asarray(nhop_edges[l][a].cpu())).tolist()
                        for k in range(3*LAYER+1):
                            if act_id >= k-k//3:
                                out += gin_bias_prop(k, None, None, gnn_model.conv1.propagate, edge_index, x, epss, weights, biass, ap)
                        for (left, right) in all_edges:
                            ekey = edge2key(left,right,x.shape[0])
                            edge_overall_score[ekey] += out/len(all_edges)

                # contribution of activation pattern in the classifier layers
                for act_id in range(2*LAYER, len(act_p)):
                    out = gin_layer_prop(None, None, gnn_model.conv1.propagate, edge_index, x, epss, weights, act_p)    
                    for k in range(3*LAYER+1):
                        if act_id >= k-k//3:
                            out += gin_bias_prop(k, None, None, gnn_model.conv1.propagate, edge_index, x, epss, weights, biass, act_p)
                    for ekey in edge_overall_score.keys():                        
                        edge_overall_score[ekey] += out/d.edge_index.shape[1]
                
                # calibration of activation pattern in the message-passing layers
                for act_id in range(2*LAYER):
                    for a in range(x.shape[0]):
                        tmp_act = torch.zeros(off_act_p[act_id].shape).to(device)
                        tmp_act[a] = copy.deepcopy(off_act_p[act_id][a])
                        ap = copy.deepcopy(off_act_p)
                        ap[act_id] = tmp_act
                        all_edges = []
                        for l in range(act_id//2+1):
                            all_edges += np.transpose(np.asarray(nhop_edges[l][a].cpu())).tolist()
                        out = gin_layer_prop(None, None, gnn_model.conv1.propagate, edge_index, x, epss, weights, ap, 1)
                        for k in range(3*LAYER+1):
                            if act_id >= k-k//3:
                                out += gin_bias_prop(k, None, None, gnn_model.conv1.propagate, edge_index, x, epss, weights, biass, ap, 1)
                        for (left, right) in all_edges:
                            ekey = edge2key(left,right,x.shape[0])
                            edge_overall_score[ekey] -= out/len(all_edges)
                
                # calibration of activation pattern in the classifier layers
                for act_id in range(2*LAYER, len(act_p)):
                    out = gin_layer_prop(None, None, gnn_model.conv1.propagate, edge_index, x, epss, weights, off_act_p, 1)
                    for k in range(3*LAYER+1):
                        if act_id >= k-k//3:
                            out += gin_bias_prop(k, None, None, gnn_model.conv1.propagate, edge_index, x, epss, weights, biass, off_act_p, 1)
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
                    # i_score = torch.abs(overall_score[:,int(d.y)])
                    # i_score = torch.norm(overall_score, dim=-1)
                    Hedges = torch.topk(i_score, d.edge_index.shape[1], dim=-1)[1].cpu().detach().numpy()
                    econfi = i_score[Hedges].cpu().detach().numpy()

                if args.linear_search>0:
                    if args.detect_not>0:
                        if flag==0:
                            _Hedges=pos_Hedges
                        else:
                            _Hedges=neg_Hedges
                    else:
                        _Hedges=Hedges[:num_edges]
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
                else: 
                    _Hedges=Hedges[:num_edges]
                
                exp_embs[int(d.y)].append(gnn_model.get_emb(x,edge_index[:,_Hedges]))
                exp_Hedges[i] = _Hedges

                if args.test_not>0:
                    f_pos, f_neg = efidelity(neg_Hedges, gnn_model, d, device)
                    print("test NOT: ", i, f_neg, f_pos, float(len(neg_Hedges)/d.edge_index.shape[1]))

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
                    print(f'global bias: {biass[-1]}')
                    print(f'prediction logits: {lin2out}')
                    print(f'total edge score: {sum(edge_overall_score.values())+off_logits}')
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
        
        # save embeddings of the explanations and the edge indice of the explantions
        emb_path = 'embeddings/'+args.dataset.lower()+'_'+args.gnn+'_goat_'+str(100*args.sparsity)+'.pkl'
        save_emb = list(exp_embs.values())
        Hedges_path = 'Hedges/'+args.dataset.lower()+'_'+args.gnn+'_goat_'+str(100*args.sparsity)+'.pkl'
        import pickle
        with open(emb_path, "wb") as f:
            pickle.dump(save_emb, f)
        with open(Hedges_path, "wb") as f:
            pickle.dump(exp_Hedges, f)

def gin_layer_prop(loc, hidden, propagate, edge_index, x, epss, weights, act_p, offset_flag=0):
    edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
    terms = product(range(2),repeat=3)
    if offset_flag==1: terms = [[1, 1, 1]]
    out = 0
    if isinstance(hidden, tuple): 
        _edge = torch.tensor(hidden).view(2,1).long().to(x.device)
        for term in terms:
            if term[loc]==1: continue
            if term[0]==1: x1 = (1.0+epss[0])*x
            else: 
                if loc == 0: x1 = propagate(_edge, x=x, edge_weight=edge_weight[0], size=None)
                else: x1 = propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            x1 = (F.linear(x1, weights[0]))*act_p[0]
            x1 = (F.linear(x1, weights[1]))*act_p[1]
            x1 = x1*weights[2]

            if term[1]==1: x2 = (1.0+epss[1])*x1
            else: 
                if loc == 1: x2 = propagate(_edge, x=x1, edge_weight=edge_weight[0], size=None)
                else: x2 = propagate(edge_index, x=x1, edge_weight=edge_weight, size=None)
            x2 = (F.linear(x2, weights[3]))*act_p[2]
            x2 = (F.linear(x2, weights[4]))*act_p[3]
            x2 = x2*weights[5]

            if term[2]==1: x3 = (1.0+epss[2])*x2
            else: 
                if loc == 2: x3 = propagate(_edge, x=x2, edge_weight=edge_weight[0], size=None)
                else: x3 = propagate(edge_index, x=x2, edge_weight=edge_weight, size=None)
            x3 = (F.linear(x3, weights[6]))*act_p[4]
            x3 = (F.linear(x3, weights[7]))*act_p[5]
            x3 = x3*weights[8]

            batch = torch.zeros(x.shape[0]).to(x.device).long()
            tmp_out = global_mean_pool(x3, batch)

            tmp_out = (F.linear(tmp_out, weights[9]))*act_p[6]
            tmp_out = F.linear(tmp_out, weights[10])
            out += tmp_out/(len(act_p)+len(term)-sum(term))
            # out += tmp_out/(len(term)-sum(term))
    else:
        for term in terms:
            if term[0]==1: x1 = (1.0+epss[0])*x
            else: x1 = propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            x1 = (F.linear(x1, weights[0]))*act_p[0]
            x1 = (F.linear(x1, weights[1]))*act_p[1]
            x1 = x1*weights[2]

            if term[1]==1: x2 = (1.0+epss[1])*x1
            else: x2 = propagate(edge_index, x=x1, edge_weight=edge_weight, size=None)
            x2 = (F.linear(x2, weights[3]))*act_p[2]
            x2 = (F.linear(x2, weights[4]))*act_p[3]
            x2 = x2*weights[5]

            if term[2]==1: x3 = (1.0+epss[2])*x2
            else: x3 = propagate(edge_index, x=x2, edge_weight=edge_weight, size=None)
            x3 = (F.linear(x3, weights[6]))*act_p[4]
            x3 = (F.linear(x3, weights[7]))*act_p[5]
            x3 = x3*weights[8]

            batch = torch.zeros(x.shape[0]).to(x.device).long()
            tmp_out = global_mean_pool(x3, batch)

            tmp_out = (F.linear(tmp_out, weights[9]))*act_p[6]
            tmp_out = F.linear(tmp_out, weights[10])
            out += tmp_out/(len(act_p)+len(term)-sum(term))
            # out += tmp_out
    return out

def gin_bias_prop(layer, loc, hidden, propagate, edge_index, x, epss, weights, biass, act_p, offset_flag=0):
    edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
    rep = max(1,len(epss)-1-(layer-(layer+1)//3)//2)
    terms = product(range(2),repeat=rep)
    if offset_flag==1: terms = [[1]*rep]
    if (layer-(layer+1)//3)//2>=2: terms = [[1]]
    out =0
    if isinstance(hidden, tuple): 
        _edge = torch.tensor(hidden).view(2,1).long().to(x.device)
        for term in terms:
            if layer == 0: 
                x1 = (biass[0].repeat(x.shape[0],1))*act_p[0]
            if layer == 1: 
                x1 = (biass[1].repeat(x.shape[0],1))*act_p[1]
            elif layer < 1: 
                x1 = (F.linear(x1, weights[1]))*act_p[1]
            if layer==2:
                x1 = biass[2].repeat(x.shape[0],1)
            elif layer<2:
                x1 = x1* weights[2]
            if layer == 3:
                x2 = (biass[3].repeat(x.shape[0],1))*act_p[2]
            elif layer < 3:
                if term[0]==1: 
                    if loc == 1: continue
                    else: x2 = (1.0+epss[1])*x1
                else: 
                    if loc==1:
                        x2 = propagate(_edge, x=x1, edge_weight=edge_weight[0], size=None)
                    else:
                        x2 = propagate(edge_index, x=x1, edge_weight=edge_weight, size=None)
                x2 = (F.linear(x2, weights[3]))*act_p[2]
            if layer == 4:
                x2 = (biass[4].repeat(x.shape[0],1))*act_p[3]
            elif layer < 4:
                x2 = (F.linear(x2, weights[4]))*act_p[3]
            if layer==5:
                x2 = biass[5].repeat(x.shape[0],1)
            elif layer<5:
                x2=x2*weights[5]
            if layer == 6:
                x3 = (biass[6].repeat(x.shape[0],1))*act_p[4]
            elif layer < 6:
                if term[-1]==1: 
                    if loc == 2: continue
                    else: x3 = (1.0+epss[2])*x2
                else: 
                    if loc == 2:
                        x3 = propagate(_edge, x=x2, edge_weight=edge_weight[0], size=None)
                    else:
                        x3 = propagate(edge_index, x=x2, edge_weight=edge_weight, size=None)
                x3 = (F.linear(x3, weights[6]))*act_p[4]
            if layer == 7:
                x3 = (biass[7].repeat(x.shape[0],1))*act_p[5]
            elif layer < 7:
                x3 = (F.linear(x3, weights[7]))*act_p[5]
            if layer==8:
                x3 = biass[8].repeat(x.shape[0],1)
            elif layer<8:
                x3=x3*weights[8]
            if layer == 9:
                tmp_out = (biass[9].view(1,-1))*act_p[6]
            elif layer < 9:
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_out = global_mean_pool(x3, batch)
                tmp_out = (F.linear(tmp_out, weights[9]))*act_p[6]
            tmp_out = F.linear(tmp_out, weights[10])
            factors = (len(act_p)-(layer-layer//3)+len(term)-sum(term))
            # print(factors,term,layer)
            out+=tmp_out/factors
            # out+=tmp_out/(len(term)-sum(term))
    else:
        for term in terms:
            # print(layer, term)
            if layer == 0: 
                x1 = (biass[0].repeat(x.shape[0],1))*act_p[0]
            if layer == 1: 
                x1 = (biass[1].repeat(x.shape[0],1))*act_p[1]
            elif layer < 1: 
                x1 = (F.linear(x1, weights[1]))*act_p[1]
            if layer==2:
                x1 = (biass[2].repeat(x.shape[0],1))
            elif layer<2:
                x1 = x1*weights[2]            
            if layer == 3:
                x2 = (biass[3].repeat(x.shape[0],1))*act_p[2]
            elif layer < 3:
                if term[0]==1: x2 = (1.0+epss[1])*x1
                else: x2 = propagate(edge_index, x=x1, edge_weight=edge_weight, size=None)
                x2 = (F.linear(x2, weights[3]))*act_p[2]
            if layer == 4:
                x2 = (biass[4].repeat(x.shape[0],1))*act_p[3]
            elif layer < 4:
                x2 = (F.linear(x2, weights[4]))*act_p[3]   
            if layer==5:
                x2 = biass[5].repeat(x.shape[0],1)
            elif layer<5:
                x2 = x2*weights[5]
            if layer == 6:
                x3 = (biass[6].repeat(x.shape[0],1))*act_p[4]
            elif layer < 6:
                if term[-1]==1: x3 = (1.0+epss[2])*x2
                else: x3 = propagate(edge_index, x=x2, edge_weight=edge_weight, size=None)
                x3 = (F.linear(x3, weights[6]))*act_p[4]
            if layer == 7:
                x3 = (biass[7].repeat(x.shape[0],1))*act_p[5]
            elif layer < 7:
                x3 = (F.linear(x3, weights[7]))*act_p[5]
            if layer==8:
                x3 = biass[8].repeat(x.shape[0],1)
            elif layer<8:
                x3=x3*weights[8]
            if layer == 9:
                tmp_out = (biass[9].view(1,-1))*act_p[6]
            elif layer < 9:
                batch = torch.zeros(x.shape[0]).to(x.device).long()
                tmp_out = global_mean_pool(x3, batch)
                tmp_out = (F.linear(tmp_out, weights[9]))*act_p[6]
            tmp_out = F.linear(tmp_out, weights[10])
            factors = (len(act_p)-(layer-layer//3)+len(term)-sum(term))
            # print(factors,term,layer)
            out+=tmp_out/factors
            # out += tmp_out
    return out

def gin_reimp(propagate, edge_index, x, epss, weights, biass, act_p):
    edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
    
    x1 = propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
    x1 += (1.0+epss[0])*x
    x1 = (F.linear(x1, weights[0]) + biass[0])*act_p[0]
    x1 = (F.linear(x1, weights[1]) + biass[1])*act_p[1]
    x1 = x1*weights[2] + biass[2]

    x2 = propagate(edge_index, x=x1, edge_weight=edge_weight, size=None)
    x2 += (1.0+epss[1])*x1
    x2 = (F.linear(x2, weights[3]) + biass[3])*act_p[2]
    x2 = (F.linear(x2, weights[4]) + biass[4])*act_p[3]
    x2 = x2*weights[5] + biass[5]

    x3 = propagate(edge_index, x=x2, edge_weight=edge_weight, size=None)
    x3 += (1.0+epss[2])*x2
    x3 = (F.linear(x3, weights[6]) + biass[6])*act_p[4]
    x3 = (F.linear(x3, weights[7]) + biass[7])*act_p[5]
    x3 = x3*weights[8] + biass[8]

    batch = torch.zeros(x.shape[0]).to(x.device).long()
    out = global_mean_pool(x3, batch)

    out = (F.linear(out, weights[9]) + biass[9])*act_p[6]
    out = F.linear(out, weights[10]) + biass[10]
    return out

def gin_reimp_no_bias(propagate, edge_index, x, epss, weights, act_p):
    edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
    
    x1 = propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
    x1 += (1.0+epss[0])*x
    x1 = (F.linear(x1, weights[0]))*act_p[0]
    x1 = (F.linear(x1, weights[1]))*act_p[1]
    x1 = x1*weights[2]

    x2 = propagate(edge_index, x=x1, edge_weight=edge_weight, size=None)
    x2 += (1.0+epss[1])*x1
    x2 = (F.linear(x2, weights[3]))*act_p[2]
    x2 = (F.linear(x2, weights[4]))*act_p[3]
    x2 = x2*weights[5]

    x3 = propagate(edge_index, x=x2, edge_weight=edge_weight, size=None)
    x3 += (1.0+epss[2])*x2
    x3 = (F.linear(x3, weights[6]))*act_p[4]
    x3 = (F.linear(x3, weights[7]))*act_p[5]
    x3 = x3*weights[8]

    batch = torch.zeros(x.shape[0]).to(x.device).long()
    out = global_mean_pool(x3, batch)

    out = (F.linear(out, weights[9]))*act_p[6]
    out = F.linear(out, weights[10])
    return out

def gin_reimp_only_bias(propagate, edge_index, x, epss, weights, biass, act_p):
    edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
    
    x1 = biass[0].repeat(x.shape[0],1)*act_p[0]
    x1 = (F.linear(x1, weights[1]) + biass[1])*act_p[1]
    x1 = x1*weights[2] + biass[2]

    x2 = propagate(edge_index, x=x1, edge_weight=edge_weight, size=None)
    x2 += (1.0+epss[1])*x1
    x2 = (F.linear(x2, weights[3]) + biass[3])*act_p[2]
    x2 = (F.linear(x2, weights[4]) + biass[4])*act_p[3]
    x2 = x2*weights[5] + biass[5]

    x3 = propagate(edge_index, x=x2, edge_weight=edge_weight, size=None)
    x3 += (1.0+epss[2])*x2
    x3 = (F.linear(x3, weights[6]) + biass[6])*act_p[4]
    x3 = (F.linear(x3, weights[7]) + biass[7])*act_p[5]
    x3 = x3*weights[8] + biass[8]

    batch = torch.zeros(x.shape[0]).to(x.device).long()
    out = global_mean_pool(x3, batch)

    out = (F.linear(out, weights[9]) + biass[9])*act_p[6]
    out = F.linear(out, weights[10]) + biass[10]
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
    parser.add_argument('--gnn', type=str, default='gin')
    parser.add_argument('--sparsity', type=float, default=0.7)
    parser.add_argument('--do_plot', type=int, default=0)
    parser.add_argument('--do_evaluate', type=int, default=1)
    parser.add_argument('--global_exp', type=int, default=0)
    parser.add_argument('--clusters', type=int, default=15)
    parser.add_argument('--is_undirected', type=int, default=1)
    parser.add_argument('--detect_not', type=int, default=0)
    parser.add_argument('--test_not', type=int, default=0)

    parser.add_argument('--plot_thres', type=float, default=-00.0)
    parser.add_argument('--linear_search', type=int, default=1)
    
    return parser.parse_args()

if __name__ == "__main__":

    args = build_args()
    main(args)
    print("done")
