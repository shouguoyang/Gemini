# embed feature in advance
import json

import tensorflow as tf
import sys

sys.path.append("..")
from utils import *
import os
import argparse
from pathlib2 import Path
from tqdm import tqdm
import pickle as pkl
from extract_feature_multi import read_json, write_json


def write_pickle(content, fname):
    with open(fname, 'wb') as f:
        pkl.dump(content, f)


def read_pickle(fname):
    with open(fname, 'rb') as f:
        return pkl.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
                    help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7,
                    help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64,
                    help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2,
                    help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64,
                    help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
                    help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--epoch', type=int, default=100,
                    help='epoch number')
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch size')
parser.add_argument('--load_path', type=str,
                    default='../saved_model/graphnn-model-DFcon_best',
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--log_path', type=str, default=None,
                    help='path for training log')


def embed_by_binaries(gnn):
    # load features
    select_bins = read_json('select_bin_paths-filter_dataset.json')
    bar = tqdm(select_bins)
    for bin_path in bar:
        bin_path = Path(bin_path.replace('O0', 'O2'))
        if not bin_path.exists():
            continue
        bar.set_description('embedding Gemini features of {}'.format(bin_path.name))
        feature_path = bin_path.parent.joinpath("{}_Gemini_features.json".format(bin_path.name))
        if not feature_path.exists():
            continue
        func_name2features = read_json(feature_path)
        func_name2embedding = {}
        for func_name, features in func_name2features.items():
            feature_list = np.asarray(features['feature_list'])
            feature_list = np.expand_dims(feature_list, axis=0)
            adj_matrix = features['adjacent_matrix']
            adj_matrix = np.expand_dims(adj_matrix, axis=0)
            func_name2embedding[func_name] = gnn.get_embed(feature_list, adj_matrix).reshape(-1)
        write_pickle(func_name2embedding,
                     str(bin_path.parent.joinpath("{}_Gemini_embedding.pkl".format(bin_path.name))))

def embed_by_text(gnn, feature_path, embedding_path):
    print('[embedding for text] from {} to {}'.format(feature_path, embedding_path))
    f_embedding = open(embedding_path, 'w')
    with open(feature_path, 'r') as f:
        for line in tqdm(f, desc='embedding...'):
            gemini_features = json.loads(line.strip())
            if len(gemini_features['features']) != len(gemini_features['succs']):
                # block feature error
                continue
            mat_size = len(gemini_features['succs'])
            adj_matrix = np.zeros((mat_size, mat_size))

            for x, ys in enumerate(gemini_features['succs']):
                for y in ys:
                    adj_matrix[x, y] = 1
            feature_list = np.expand_dims(np.asarray(gemini_features['features']), axis=0)
            adj_matrix = np.expand_dims(adj_matrix, axis=0)
            embedding = gnn.get_embed(feature_list, adj_matrix)
            f_embedding.write(json.dumps({'fid': gemini_features['fid'], 'embedding': embedding.tolist()}) + '\n')

    f_embedding.close()



if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5
    DATA_FILE_NAME = '../data/acfgSSL_{}/'.format(NODE_FEATURE_DIM)
    SOFTWARE = ('openssl-1.0.1f-', 'openssl-1.0.1u-')
    OPTIMIZATION = ('-O0', '-O1', '-O2', '-O3')
    COMPILER = ('armeb-linux', 'i586-linux', 'mips-linux')
    VERSION = ('v54',)

    FUNC_NAME_DICT = {}

    # Process the input graphs
    F_NAME = get_f_name(DATA_FILE_NAME, SOFTWARE, COMPILER,
                        OPTIMIZATION, VERSION)
    FUNC_NAME_DICT = get_f_dict(F_NAME)

    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)
    print "{} graphs, {} functions".format(len(Gs), len(classes))

    if os.path.isfile('../data/class_perm.npy'):
        perm = np.load('../data/class_perm.npy')
    else:
        perm = np.random.permutation(len(classes))
        np.save('../data/class_perm.npy', perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save('../data/class_perm.npy', perm)

    # Model
    gnn = graphnn(
        N_x=NODE_FEATURE_DIM,
        Dtype=Dtype,
        N_embed=EMBED_DIM,
        depth_embed=EMBED_DEPTH,
        N_o=OUTPUT_DIM,
        ITER_LEVEL=ITERATION_LEVEL,
        lr=LEARNING_RATE
    )
    gnn.init(LOAD_PATH, LOG_PATH)

    # embed_by_text(gnn,
    #               feature_path='../data/train_gemini_features.json',
    #               embedding_path='../data/train_gemini_embeddings.json')
    embed_by_text(gnn,
                  feature_path='../data/test_gemini_features.json',
                  embedding_path='../data/test_gemini_embeddings.json')
