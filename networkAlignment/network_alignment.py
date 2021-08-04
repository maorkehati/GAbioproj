import os
os.environ['MPLCONFIGDIR'] = '/home/yandex/AMNLP2021/maorkehati/GAbioproj'

from input.dataset import Dataset
from time import time
from algorithms import *
from evaluation.metrics import get_statistics
from evaluation.matcher import greedy_match, get_pairs
import utils.graph_utils as graph_utils
import random
import numpy as np
import torch
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="dataspace/ppi/human/graphsage/")
    parser.add_argument('--target_dataset', default="dataspace/ppi/mouse/graphsage/")
    parser.add_argument('--groundtruth',    default="dataspace/douban/dictionaries/groundtruth")
    parser.add_argument('--alignment_matrix_name', default=None, help="Prefered name of alignment matrix.")
    parser.add_argument('--seed',           default=123,    type=int)

    subparsers = parser.add_subparsers(dest="algorithm", help='Choose 1 of the algorithm from: IsoRank, FINAL, UniAlign, PALE, DeepLink, REGAL, IONE, NAWAL')

    parser_IsoRank = subparsers.add_parser('IsoRank', help='IsoRank algorithm')
    parser_IsoRank.add_argument('--H',                   default=None, help="Priority matrix")
    parser_IsoRank.add_argument('--max_iter',            default=30, type=int, help="Max iteration")
    parser_IsoRank.add_argument('--alpha',               default=0.82, type=float)
    parser_IsoRank.add_argument('--tol',                 default=1e-4, type=float)

    parser_FINAL = subparsers.add_parser('FINAL', help='FINAL algorithm')
    parser_FINAL.add_argument('--H',                   default=None, help="Priority matrix")
    parser_FINAL.add_argument('--max_iter',            default=30, type=int, help="Max iteration")
    parser_FINAL.add_argument('--alpha',               default=0.82, type=float)
    parser_FINAL.add_argument('--tol',                 default=1e-4, type=float)

    parser_BigAlign = subparsers.add_parser('BigAlign', help='BigAlign algorithm')
    parser_BigAlign.add_argument('--lamb', default=0.01, help="Lambda")

    parser_IONE = subparsers.add_parser('IONE', help='IONE algorithm')
    parser_IONE.add_argument('--train_dict', default="groundtruth.train", help="Groundtruth use to train.")
    parser_IONE.add_argument('--epochs', default=100, help="Total iterations.", type=int)
    parser_IONE.add_argument('--dim', default=100, help="Embedding dimension.")


    parser_REGAL = subparsers.add_parser('REGAL', help='REGAL algorithm')
    parser_REGAL.add_argument('--attrvals', type=int, default=2,
                        help='Number of attribute values. Only used if synthetic attributes are generated')

    parser_REGAL.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser_REGAL.add_argument('--k', type=int, default=10,
                        help='Controls of landmarks to sample. Default is 10.')

    parser_REGAL.add_argument('--max_layer', type=int, default=2,
                        help='Calculation until the layer for xNetMF.')
    parser_REGAL.add_argument('--alpha', type=float, default=0.01, help="Discount factor for further layers")
    parser_REGAL.add_argument('--gammastruc', type=float, default=1, help="Weight on structural similarity")
    parser_REGAL.add_argument('--gammaattr', type=float, default=1, help="Weight on attribute similarity")
    parser_REGAL.add_argument('--num_top', type=int, default=10,
                        help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
    parser_REGAL.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")

    parser_PALE = subparsers.add_parser('PALE', help="PALE algorithm")
    parser_PALE.add_argument('--cuda',                action='store_true')

    parser_PALE.add_argument('--learning_rate1',      default=0.01,        type=float)
    parser_PALE.add_argument('--embedding_dim',       default=300,         type=int)
    parser_PALE.add_argument('--batch_size_embedding',default=512,         type=int)
    parser_PALE.add_argument('--embedding_epochs',    default=1000,        type=int)
    parser_PALE.add_argument('--neg_sample_size',     default=10,          type=int)

    parser_PALE.add_argument('--learning_rate2',      default=0.01,       type=float)
    parser_PALE.add_argument('--batch_size_mapping',  default=32,         type=int)
    parser_PALE.add_argument('--mapping_epochs',      default=100,         type=int)
    parser_PALE.add_argument('--mapping_model',       default='linear')
    parser_PALE.add_argument('--activate_function',   default='sigmoid')
    parser_PALE.add_argument('--train_dict',          default='dataspace/douban/dictionaries/node,split=0.2.train.dict')
    parser_PALE.add_argument('--embedding_name',          default='')


    parser_DeepLink = subparsers.add_parser("DeepLink", help="DeepLink algorithm")
    parser_DeepLink.add_argument('--cuda',                action="store_true")

    parser_DeepLink.add_argument('--embedding_dim',       default=800,         type=int)
    parser_DeepLink.add_argument('--embedding_epochs',    default=5,        type=int)

    parser_DeepLink.add_argument('--unsupervised_lr',     default=0.001, type=float)
    parser_DeepLink.add_argument('--supervised_lr',       default=0.001, type=float)
    parser_DeepLink.add_argument('--batch_size_mapping',  default=32,         type=int)
    parser_DeepLink.add_argument('--unsupervised_epochs', default=500, type=int)
    parser_DeepLink.add_argument('--supervised_epochs',   default=2000,         type=int)

    parser_DeepLink.add_argument('--train_dict',          default="dataspace/dictionaries/node,split=0.2.train.dict")
    parser_DeepLink.add_argument('--hidden_dim1',         default=1200, type=int)
    parser_DeepLink.add_argument('--hidden_dim2',         default=1600, type=int)

    parser_DeepLink.add_argument('--number_walks',        default=1000, type=int)
    parser_DeepLink.add_argument('--format',              default="edgelist")
    parser_DeepLink.add_argument('--walk_length',         default=5, type=int)
    parser_DeepLink.add_argument('--window_size',         default=2, type=int)
    parser_DeepLink.add_argument('--top_k',               default=5, type=int)
    parser_DeepLink.add_argument('--alpha',               default=0.8, type=float)
    parser_DeepLink.add_argument('--num_cores',           default=8, type=int)
    
    parser_NAWAL = subparsers.add_parser("NAWAL", help="NAWAL")
    parser_NAWAL.add_argument('--NAWAL_only',               default=True,  type=bool)
    parser_NAWAL.add_argument('--load_emb',                 action="store_true")
    parser_NAWAL.add_argument('--save_emb',                 action="store_true")
    parser_NAWAL.add_argument('--embedding_name',           default="emb",   type=str)
    
    parser_NAWAL.add_argument('--embedding_dim',            default=300,   type=int)
    parser_NAWAL.add_argument('--neg_sample_size',          default=10,    type=int)
    parser_NAWAL.add_argument('--emb_lr',                   default=0.001,    type=float)
    parser_NAWAL.add_argument('--cuda',                     action='store_true')
    parser_NAWAL.add_argument('--batch_size_embedding',     default=512,   type=int)
    parser_NAWAL.add_argument('--embedding_epochs',         default=10,  type=int)
    parser_NAWAL.add_argument('--map_beta',                 default=0.01,  type=float)
    
    
    parser_NAWAL.add_argument('--nawal_mapping_epochs',     default=5,  type=int)
    parser_NAWAL.add_argument('--nawal_mapping_epoch_size', default=1024,  type=int)
    parser_NAWAL.add_argument('--nawal_mapping_batch_size', default=64,  type=int)
    parser_NAWAL.add_argument('--num_keep',                 default=400,  type=int)
    parser_NAWAL.add_argument('--map_optimizer',            default="adam",  type=str)
    
    parser_NAWAL.add_argument('--dis_optimizer',            default="adam",  type=str)
    parser_NAWAL.add_argument('--dis_dropout',              default=0.1,  type=float)
    parser_NAWAL.add_argument('--dis_input_dropout',        default=0.01,  type=float)
    parser_NAWAL.add_argument('--dis_hid_dim',              default=300,  type=int)
    parser_NAWAL.add_argument('--dis_layers',               default=1,  type=int)
    parser_NAWAL.add_argument('--dis_smooth',               default=0.0,  type=float)
    parser_NAWAL.add_argument('--dis_steps',                default=1,  type=int)
    
    parser_NAWAL.add_argument('--min_lr',                   default=0.001,    type=float)
    parser_NAWAL.add_argument('--lr_decay',                 default=0.999,  type=float)
    parser_NAWAL.add_argument('--lr_shrink',                default=1.0,  type=float)
    
    parser_NAWAL.add_argument('--folder',                   default="",  type=str)
    
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    source_nodes = source_dataset.G.nodes()
    target_nodes = target_dataset.G.nodes()
    groundtruth_matrix = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx)

    algorithm = args.algorithm

    if (algorithm == "IsoRank"):
        model = IsoRank(source_dataset, target_dataset, args.H, args.alpha, args.max_iter, args.tol)
    elif (algorithm == "FINAL"):
        model = FINAL(source_dataset, target_dataset, H=args.H, alpha=args.alpha, maxiter=args.max_iter, tol=args.tol)
    elif (algorithm == "REGAL"):
        model = REGAL(source_dataset, target_dataset, max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=args.buckets,
                      gammastruc = args.gammastruc, gammaattr = args.gammaattr, normalize=True, num_top=args.num_top)
    elif algorithm == "BigAlign":
        model = BigAlign(source_dataset, target_dataset, lamb=args.lamb)
    elif algorithm == "IONE":
        model = IONE(source_dataset, target_dataset, gt_train=args.train_dict, epochs=args.epochs, dim=args.dim, seed=args.seed)
    elif algorithm == "PALE":
        model = PALE(source_dataset, target_dataset, args)
    elif algorithm == "DeepLink":
        model = DeepLink(source_dataset, target_dataset, args)
    elif algorithm == "NAWAL":
        model = NAWAL(source_dataset, target_dataset, args)
    else:
        raise Exception("Unsupported algorithm")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(7)

    start_time = time()

    S = model.align()
    
    if algorithm != "NAWAL":
        get_statistics(S, groundtruth_matrix)
             
    print("Full_time: ", time() - start_time)
    
    pairs = get_pairs(greedy_match(S))
    
    path = '/home/yandex/AMNLP2021/maorkehati/GAbioproj/networkAlignment'
    
    with open(f'{path}/{"/".join([i for i in args.source_dataset.split("/") if i][:-1])}/edgelist/dic.pkl','rb') as handle:
        sd = pickle.load(handle)
        
    with open(f'{path}/{"/".join([i for i in args.target_dataset.split("/") if i][:-1])}/edgelist/dic.pkl','rb') as handle:
        td = pickle.load(handle)
        
    A_nodes = [sd[i] for i in range(S.shape[0])]
    B_nodes = [td[i] for i in range(S.shape[1])]
    dS = {'X':S, 'A_nodes': A_nodes, 'B_nodes': B_nodes, }
        
    pairs = [(sd[i[0]], td[i[1]]) for i in pairs]
    
    folder = args.folder
    np.save(f"{path}/outs/{folder}/pairs.npy", pairs)
    with open(f"{path}/outs/{folder}/S.pkl",'wb') as handle:
        pickle.dump(dS, handle)
    
    open(f"{path}/outs/{folder}/args.txt", 'w').write("\r\n".join([f'{i[0]}:{i[1]}' for i in list(vars(args).items())]))
