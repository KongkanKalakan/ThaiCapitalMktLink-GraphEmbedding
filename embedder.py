import torch
from utils import process


class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        #if True:
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")

        if args.dataset == "dblp":
            adj_list, features, labels, idx_train, idx_val, idx_test = process.load_dblp(args.sc)
        elif args.dataset == "acm":
            adj_list, features, labels, idx_train, idx_val, idx_test = process.load_acm_mat()
        elif args.dataset == "imdb":
            adj_list, features, labels, idx_train, idx_val, idx_test = process.load_imdb(args.sc)
        elif args.dataset == "amazon":
            adj_list, features, labels, idx_train, idx_val, idx_test = process.load_amazon(args.sc)
        else:
            path = r'..\..\N2VP_221212\Data\input\\'
            graph_data, features, labels, idx_train, idx_val, idx_test = process.prepare_data(path)
            if args.dataset == "exp1a":
                adj_list = process.pack_exp1a(graph_data)
            if args.dataset == "exp1b":
                adj_list = process.pack_exp1b(graph_data)
            if args.dataset == "exp1c":
                adj_list = process.pack_exp1c(graph_data)
            if args.dataset == "exp1d":
                adj_list = process.pack_exp1d(graph_data)
            if args.dataset == "exp2a":
                adj_list = process.pack_exp2a(graph_data)
            if args.dataset == "exp2b":
                adj_list = process.pack_exp2b(graph_data)
            if args.dataset == "exp2c":
                adj_list = process.pack_exp2c(graph_data)
            if args.dataset == "exp3":
                adj_list = process.pack_exp3(graph_data)
            if args.dataset == "exp4":
                adj_list = process.pack_exp4(graph_data)
            if args.dataset == "exp5":
                adj_list = process.pack_exp5(graph_data)
            if args.dataset == "exp6":
                adj_list = process.pack_exp6(graph_data)
            if args.dataset == "exp7":
                adj_list = process.pack_exp7(graph_data)
            if args.dataset == "exp8":
                adj_list = process.pack_exp8(graph_data)
            if args.dataset == "exp9":
                adj_list = process.pack_exp9(graph_data)
            if args.dataset == "PearsonKendall":
                adj_list = process.pack_PearsonKendall(graph_data)
            if args.dataset == "PearsonKendallCopulaGranger":
                adj_list = process.pack_PearsonKendallCopulaGranger(graph_data)
            if args.dataset == "PearsonCopula":
                adj_list = process.pack_PearsonCopula(graph_data)
            if args.dataset == "PearsonGranger":
                adj_list = process.pack_PearsonGranger(graph_data)
            if args.dataset == "KendallCopula":
                adj_list = process.pack_KendallCopula(graph_data)
            if args.dataset == "KendallGranger":
                adj_list = process.pack_KendallGranger(graph_data)
            if args.dataset == "CopulaGranger":
                adj_list = process.pack_CopulaGranger(graph_data)
        
        features = process.preprocess_features(features)
        
        args.nb_nodes = adj_list[0].shape[0]
        args.ft_size = features[0].shape[1]
        args.nb_classes = labels.shape[1]

        adj_list = [process.normalize_adj(adj) for adj in adj_list]
        self.adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)

        self.args = args
