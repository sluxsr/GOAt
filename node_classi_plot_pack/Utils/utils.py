
import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np

def show():
    plt.show()
    plt.clf()

def NC_vis_graph(edge_index, y, node_idx, datasetname=None, un_edge_index=None, nodelist=None, H_edges=None):
    y = torch.argmax(y, dim=-1)
    node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
    node_color = ['#FFA500', '#4970C6', '#FE0000', 'green','orchid','darksalmon','darkslategray','gold','bisque','tan','navy','indigo','lime',]
    colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

    node_idx = int(node_idx)
    edges = np.transpose(np.asarray(edge_index.cpu()))
    if un_edge_index is not None: un_edges = np.transpose(np.asarray(un_edge_index.cpu()))
    if nodelist is not None:
        edgelist = [(n_frm, n_to) for (n_frm, n_to) in edges if
                                n_frm in nodelist and n_to in nodelist]
        nodelist = nodelist.tolist()
    elif H_edges is not None:
        if un_edge_index is not None: edgelist = un_edges[H_edges]
        else: edgelist = edges[H_edges]
        nodelist = list(set(list(edgelist[:,0])+list(edgelist[:,1])))
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    if datasetname == "tree_grid":
        G = nx.ego_graph(G, node_idx, radius=3)
    elif datasetname == "ba_community":
        G = nx.ego_graph(G, node_idx, radius=2)
    else:
        G = nx.ego_graph(G, node_idx, radius=3)

    for n in nodelist:
        if n not in G.nodes:
            nodelist.remove(n)
    def remove_unavail(edgelist):
        for i, tup in enumerate(edgelist):
            if tup[0] not in G.nodes or tup[1] not in G.nodes:
                edgelist = np.delete(edgelist, i, axis=0)
                return edgelist, i
        return edgelist, len(edgelist)
    edgelist, i = remove_unavail(edgelist)
    while i != len(edgelist):
        edgelist, i = remove_unavail(edgelist)

    pos = nx.kamada_kawai_layout(G) # calculate according to graph.nodes()
    pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}
    colors = [colors[pp] for pp in list(G.nodes)]

    nx.draw_networkx_nodes(G, pos,
                            nodelist=list(G.nodes()),
                            node_color=colors,
                            node_size=100)
    if isinstance(colors, list):
        list_indices = int(np.where(np.array(G.nodes()) == node_idx)[0])
        node_idx_color = colors[list_indices]
    else:
        node_idx_color = colors

    nx.draw_networkx_nodes(G, pos=pos,
                            nodelist=[node_idx],
                            node_color=node_idx_color,
                            node_size=400)

    nx.draw_networkx_edges(G, pos, width=1, edge_color='grey', arrows=False)
    if nodelist is not None or H_edges is not None:
        nx.draw_networkx_edges(G, pos=pos_nodelist,
                            edgelist=edgelist, width=2,
                            edge_color='red',
                            arrows=False)

    labels = {o:o for o in list(G.nodes)}
    nx.draw_networkx_labels(G, pos,labels)

    plt.axis('off')