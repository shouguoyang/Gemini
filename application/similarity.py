import tensorflow as tf
print tf.__version__
import sys
sys.path.append("..")
import numpy as np
from datetime import datetime
from graphnnSiamese import graphnn
from utils import *
import os
import argparse
import json


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
                    default='../saved_model/graphnn-model_best',
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--log_path', type=str, default=None,
                    help='path for training log')

parser.add_argument("cve_function_json", type=str,
                    help="the json file saves the feature list and adjacent matrix for one function")

parser.add_argument("target_function_json", type=str,
                    help="the json file saves the feature lists and adjacent matrix for many function")

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
    cve_json_file = args.cve_function_json
    target_json_file = args.target_function_json

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

    # Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test = \
    #     partition_data(Gs, classes, [0.8, 0.1, 0.1], perm)
    #
    # print "Train: {} graphs, {} functions".format(
    #     len(Gs_train), len(classes_train))
    # print "Dev: {} graphs, {} functions".format(
    #     len(Gs_dev), len(classes_dev))
    # print "Test: {} graphs, {} functions".format(
    #     len(Gs_test), len(classes_test))
    #
    # # Fix the pairs for validation and testing
    # if os.path.isfile('../data/valid.json'):
    #     with open('../data/valid.json') as inf:
    #         valid_ids = json.load(inf)
    #     valid_epoch = generate_epoch_pair(
    #         Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    # else:
    #     valid_epoch, valid_ids = generate_epoch_pair(
    #         Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
    #     with open('../data/valid.json', 'w') as outf:
    #         json.dump(valid_ids, outf)
    #
    # if os.path.isfile('../data/test.json'):
    #     with open('../data/test.json') as inf:
    #         test_ids = json.load(inf)
    #     test_epoch = generate_epoch_pair(
    #         Gs_test, classes_test, BATCH_SIZE, load_id=test_ids)
    # else:
    #     test_epoch, test_ids = generate_epoch_pair(
    #         Gs_test, classes_test, BATCH_SIZE, output_id=True)
    #     with open('../data/test.json', 'w') as outf:
    #         json.dump(test_ids, outf)

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

    cve_dic_list = json.load(open(cve_json_file, 'r'))
    target_json_list = json.load(open(target_json_file,'r'))

    cve_dic = cve_dic_list[0]
    cve_func_name = cve_dic['func_name']
    cve_func_fealis =  np.asarray(cve_dic["feature_list"])
    cve_func_fealis = np.expand_dims(cve_func_fealis, axis=0)
    cve_func_am = np.asarray(cve_dic["adjacent_matrix"])
    cve_func_am = np.expand_dims(cve_func_am, axis=0)
    res = []
    for target in target_json_list:
        name = target['func_name']
        fl = np.asarray(target["feature_list"])
        fl = np.expand_dims(fl, axis=0)
        am = np.asarray(target["adjacent_matrix"])
        am = np.expand_dims(am, axis=0)
        sim = gnn.calc_diff(X1=cve_func_fealis, X2=fl,
                            mask1=cve_func_am,mask2=am)

        res.append((sim.tolist()[0], name))

    res.sort(key=lambda x:x[0])


    # save to json
    saved_file_name = cve_json_file[:-5] + "_"+target_json_file[:-5] + ".res"
    json.dump(res, open(saved_file_name, 'w'))
