import tensorflow as tf
print tf.__version__
from datetime import datetime
from tqdm import tqdm
from pathlib2 import Path
from collections import defaultdict
from utils import *
import os
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='-1',
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
parser.add_argument('--batch_size', type=int, default=20,
        help='batch size')
parser.add_argument('--load_path', type=str, default=None,
        help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--save_path', type=str,
        default='./saved_model/graphnn-model-DFcon', help='path for model saving')
parser.add_argument('--log_path', type=str, default='log/train_DFcon-process.log',
        help='path for training log')


def read_graph2(file_path):
    """
    :param file_path:
    :return: a list of graphs(function), classes list(gp_id -> fid)
    """
    graphs = []
    classes = []
    for f in range(70000):
        classes.append([])

    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError('can not find file:{}'.format(file_path))
    with file_path.open('r') as f:
        cnt = 500000
        for line in tqdm(f, desc='reading graph from {}'.format(file_path)):
            g_info = json.loads(line.strip())
            if len(g_info['features']) != len(g_info['succs']):
                # block feature error
                continue
            label = int(g_info['gp_id']) - 1
            classes[label].append(len(graphs))
            cur_graph = graph(g_info['n_num'], label, file_path.name)
            for u in range(g_info['n_num']):
                cur_graph.features[u] = np.array(g_info['features'][u])
                for v in g_info['succs'][u]:
                    try:
                        cur_graph.add_edge(u, v)
                    except IndexError:
                        print cur_graph

            graphs.append(cur_graph)
            cnt -= 1
            if cnt < 0:
                break

    return graphs, classes

if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    # os.environ["CUDA_VISIBLE_DEVICES"]=args.device
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
    SAVE_PATH = args.save_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5
    # DATA_FILE_NAME = './data/acfgSSL_{}/'.format(NODE_FEATURE_DIM)
    # DATA_FILE_NAME = '/home/cp/code/DF-competation/features/train_gemini_features.json'
    DATA_FILE_NAME = 'data/train_gemini_features.json'

    FUNC_NAME_DICT = {}


    Gs, classes = read_graph2(DATA_FILE_NAME)
    print "{} graphs, {} functions".format(len(Gs), len(classes))

    perm_path = 'data/class_perm_{}.npy'.format(len(classes))
    if os.path.isfile(perm_path):
        print 'perm exist, loading...'
        perm = np.load(perm_path)
    else:
        print 'perm does not exist, generating...'
        perm = np.random.permutation(len(classes))
        np.save(perm_path, perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save(perm_path, perm)

    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test =\
            partition_data(Gs,classes,[0.8,0.1,0.1],perm)

    print "Train: {} graphs, {} functions".format(
            len(Gs_train), len(classes_train))
    print "Dev: {} graphs, {} functions".format(
            len(Gs_dev), len(classes_dev))
    print "Test: {} graphs, {} functions".format(
            len(Gs_test), len(classes_test))

    # Fix the pairs for validation
    valid_path = Path('data/valid_{}.json'.format(len(classes)))
    if False and valid_path.exists():
        print 'valid data exist, loading from {}'.format(valid_path)
        with open(str(valid_path)) as inf:
            valid_ids = json.load(inf)
        valid_epoch = generate_epoch_pair(
                Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    else:
        print 'valid does not exist, generating...'
        valid_epoch, valid_ids = generate_epoch_pair(
                Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
        with open(str(valid_path), 'w') as outf:
            json.dump(valid_ids, outf)

    # Model
    gnn = graphnn(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype, 
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )
    gnn.init(LOAD_PATH, LOG_PATH)

    # Train
    # auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
    #         BATCH_SIZE, load_data=valid_epoch)
    # gnn.say("Initial training auc = {0} @ {1}".format(auc, datetime.now()))
    # auc0, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
    #         BATCH_SIZE, load_data=valid_epoch)
    # gnn.say("Initial validation auc = {0} @ {1}".format(auc0, datetime.now()))

    best_auc = 0
    for i in range(1, MAX_EPOCH+1):
        l = train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE, i)
        gnn.say("EPOCH {3}/{0}, loss = {1} @ {2}".format(
            MAX_EPOCH, l, datetime.now(), i))

        if (i % TEST_FREQ == 0):
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
                    BATCH_SIZE, load_data=valid_epoch)
            gnn.say("Testing model: training auc = {0} @ {1}".format(
                auc, datetime.now()))
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
                    BATCH_SIZE, load_data=valid_epoch)
            gnn.say("Testing model: validation auc = {0} @ {1}".format(
                auc, datetime.now()))

            if auc > best_auc:
                path = gnn.save(SAVE_PATH+'_best')
                best_auc = auc
                gnn.say("Model saved in {}".format(path))

        if (i % SAVE_FREQ == 0):
            path = gnn.save(SAVE_PATH, i)
            gnn.say("Model saved in {}".format(path))
