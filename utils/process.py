import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp
import networkx as nx
import pandas as pd

# edited
def prepare_data(path, sc=3, seed=27):
    FILES = [('BivarCopula.edg', 'unweighted', nx.DiGraph), 
             ('BivarCopula_2D.edg', 'unweighted', nx.DiGraph), 
             ('BivarCopula_L1.edg', 'unweighted', nx.DiGraph), 
             ('Granger_11D.edg', 'unweighted', nx.DiGraph),
             ('Granger_6D.edg', 'unweighted', nx.DiGraph), 
             ('Granger_VL.edg', 'unweighted', nx.DiGraph),
             ('Kendall_AbsGreater05.edg', 'unweighted', nx.Graph), 
             ('Kendall_AbsGreater02.edg', 'unweighted', nx.Graph), 
             ('Kendall_AbsGreater01.edg', 'unweighted', nx.Graph), 
             ('Kendall_Positive.edg', 'unweighted', nx.Graph),
             ('Kendall_Weighted.edg', 'weighted', nx.Graph), 
             ('Pearson_AbsGreater05.edg', 'unweighted', nx.Graph),
             ('Pearson_AbsGreater02.edg', 'unweighted', nx.Graph),
             ('Pearson_AbsGreater01.edg', 'unweighted', nx.Graph),
             ('Pearson_Positive.edg', 'unweighted', nx.Graph), 
             ('Pearson_Weighted.edg', 'weighted', nx.Graph),
             ('Spearman_AbsGreater05.edg', 'unweighted', nx.Graph), 
             ('Spearman_AbsGreater02.edg', 'unweighted', nx.Graph), 
             ('Spearman_AbsGreater01.edg', 'unweighted', nx.Graph), 
             ('Spearman_Positive.edg', 'unweighted', nx.Graph),
             ('Spearman_Weighted.edg', 'weighted', nx.Graph)]

    NODE_NAME = [('SET', 'EQ'), 
                 ('MAI', 'EQ'), 
                 ('ZeroShort', 'FI'), 
                 ('ZeroLong', 'FI'), 
                 ('CorpBond', 'FI'), 
                 ('THB', 'FX'),
                 ('EquityFlow', 'EQ'), 
                 ('BondFlow', 'FI'), 
                 ('EMAsiaEquity', 'EQ'), 
                 ('SP500', 'EQ'), 
                 ('EMBond', 'FI'),
                 ('USBond', 'FI'), 
                 ('EMAsiaFX', 'FX'), 
                 ('USD', 'FX')]

    n_node = len(NODE_NAME)

    label = pd.DataFrame(NODE_NAME, columns=['market', 'label'])
    label = pd.get_dummies(label['label']).values

    features = np.ones(n_node).astype(float).reshape(14, 1)
    features = sp.lil_matrix(features)

    graph_data = {}
    nodelist = [i for i in range(1, 15)]
    for file, mode, graph_constructor in FILES:
        name = file.split('.')[0]
        if mode == 'weighted':
            G = nx.read_edgelist(path + file, nodetype=int, data=(('weight', float),), create_using=graph_constructor)
        else:
            G = nx.read_edgelist(path + file, nodetype=int, create_using=graph_constructor)
        G.add_nodes_from(set(nodelist) - set(G.nodes()))
        G = nx.adjacency_matrix(G, nodelist, weight='weight') + np.eye(len(G))*sc
        G = sp.csr_matrix(G)
        graph_data[name] = G
        
    #train, validate, test = np.split(np.arange(n_node)[np.random.RandomState(seed=27).permutation(n_node)], 
    #                                 [int(.7*n_node), int(.8*n_node)])
    train = np.arange(n_node)
    validate = np.arange(n_node)
    test = np.arange(n_node)
    return graph_data, features, label, train, validate, test
    

def pack_exp1a(graph_data):
    return [graph_data['Pearson_Weighted'], graph_data['Spearman_Weighted'], graph_data['Kendall_Weighted']]

def pack_exp1b(graph_data):
    return [graph_data['Pearson_Positive'], graph_data['Spearman_Positive'], graph_data['Kendall_Positive']]

def pack_exp1c(graph_data):
    return [graph_data['Pearson_AbsGreater02'], graph_data['Spearman_AbsGreater02'], graph_data['Kendall_AbsGreater02']]

def pack_exp1d(graph_data):
    return [graph_data['Pearson_AbsGreater05'], graph_data['Spearman_AbsGreater05'], graph_data['Kendall_AbsGreater05']]

def pack_exp2a(graph_data):
    return [graph_data['Pearson_Weighted'], graph_data['Pearson_Positive'], graph_data['Pearson_AbsGreater02'], graph_data['Pearson_AbsGreater05']]

def pack_exp2b(graph_data):
    return [graph_data['Spearman_Weighted'], graph_data['Spearman_Positive'], graph_data['Spearman_AbsGreater02'], graph_data['Spearman_AbsGreater05']]

def pack_exp2c(graph_data):
    return [graph_data['Kendall_Weighted'], graph_data['Kendall_Positive'], graph_data['Kendall_AbsGreater02'], graph_data['Kendall_AbsGreater05']]

def pack_exp3(graph_data):
    return [graph_data['BivarCopula'], graph_data['BivarCopula_L1'], graph_data['BivarCopula_2D']]

def pack_exp4(graph_data):
    return [graph_data['Granger_6D'], graph_data['Granger_11D'], graph_data['Granger_VL']]

def pack_exp5(graph_data):
    return [graph_data['BivarCopula_2D'], graph_data['Granger_VL']]

def pack_exp6(graph_data):
    return [graph_data['Pearson_Weighted'], graph_data['Spearman_Weighted'], graph_data['Kendall_Weighted'], 
            graph_data['BivarCopula_2D'], graph_data['Granger_VL']]

def pack_exp7(graph_data):
    return [graph_data['Pearson_Positive'], graph_data['Spearman_Positive'], graph_data['Kendall_Positive'], 
            graph_data['BivarCopula_2D'], graph_data['Granger_VL']]

def pack_exp8(graph_data):
    return [graph_data['Pearson_AbsGreater02'], graph_data['Spearman_AbsGreater02'], graph_data['Kendall_AbsGreater02'], 
            graph_data['BivarCopula_2D'], graph_data['Granger_VL']]

def pack_exp9(graph_data):
    return [graph_data['Pearson_AbsGreater05'], graph_data['Spearman_AbsGreater05'], graph_data['Kendall_AbsGreater05'], 
            graph_data['BivarCopula_2D'], graph_data['Granger_VL']] 

def pack_PearsonKendall(graph_data):
    return [graph_data['Pearson_AbsGreater01'], graph_data['Kendall_AbsGreater01']] 
    
def pack_PearsonKendallCopulaGranger(graph_data):
    return [graph_data['Pearson_AbsGreater01'], graph_data['Kendall_AbsGreater01'], 
            graph_data['BivarCopula_2D'], graph_data['Granger_VL']] 
            
def pack_PearsonCopula(graph_data):
    return [graph_data['Pearson_AbsGreater01'], graph_data['BivarCopula_2D']] 
    
def pack_PearsonGranger(graph_data):
    return [graph_data['Pearson_AbsGreater01'], graph_data['Granger_VL']] 
    
def pack_KendallCopula(graph_data):
    return [graph_data['Kendall_AbsGreater01'], graph_data['BivarCopula_2D']] 
    
def pack_KendallGranger(graph_data):
    return [graph_data['Kendall_AbsGreater01'], graph_data['Granger_VL']] 
    
def pack_CopulaGranger(graph_data):
    return [graph_data['BivarCopula_2D'], graph_data['Granger_VL']] 
#

def load_acm_mat(sc=3):
    data = sio.loadmat('data/acm.mat')
    label = data['label']

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0])*sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test


def load_dblp(sc=3):
    data = pkl.load(open("data/dblp.pkl", "rb"))
    label = data['label']

    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0])*sc
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0])*sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test


def load_imdb(sc=3):
    data = pkl.load(open("data/imdb.pkl", "rb"))
    label = data['label']

    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*sc
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test


def load_amazon(sc=3):
    data = pkl.load(open("data/amazon.pkl", "rb"))
    label = data['label']

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*sc
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*sc
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
