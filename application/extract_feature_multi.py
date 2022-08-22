#encoding=utf-8
#无需学会idapython 的使用，直接调用该类下的接口即可获得函数
#系统ida所在的路径
idapath = '/home/cp/Application/idapro-7.5/idat64'
import os,time,commands,json
from pathlib2 import Path
from tqdm import tqdm
import argparse
from collections import OrderedDict

parse = argparse.ArgumentParser()
import sys
pro_path = sys.path[0]

class getFeature:
    def __init__(self, binarypath):
        self._bin = binarypath
        self._tmpfile = pro_path + os.sep + str(binarypath).split('/')[-1] + str(time.time()) + '.json'

    #read json file to get features
    def _ReadFeatures(self):
        with open(self._tmpfile,'r') as f:
            for line in f.readlines():
                # print line
                x = json.loads(unicode(line,errors='ignore'))
                yield x

    def _del_tmpfile(self):
        os.remove(self._tmpfile)

    def get_Feature_all(self):
        return self.get_Feature_Function('')
        pass

    def get_Feature_Function(self, func_name=''):

        cmd = "TVHEADLESS=1 %s -A -S'%s/Feature_Of_Binary.py %s %s' %s" % (idapath, pro_path, self._tmpfile, func_name, self._bin)
        # print cmd
        s,o = commands.getstatusoutput(cmd)

        if s!=0 :
            print 'error occurs when extract Features from ida database file'
            print 'cmd is %s' % cmd
            print s,o
            return None

        features = list(self._ReadFeatures())
        self._del_tmpfile()
        return features

def extract_bin(binary_path):
    # generate ida database file
    binary_path = Path(binary_path)
    gf = getFeature(binary_path)
    feature = gf.get_Feature_Function()
    out_file = Path(binary_path).resolve().parent.joinpath('{}_Gemini_features.json'.format(binary_path.name))

    # func_dics = []
    func_name2features = {}
    for dic in feature:
        nodes_ordered_list = []
        for node_addr in dic.keys():
            if str(node_addr).startswith('0x'):
                nodes_ordered_list.append(node_addr)
        feature_list = [] # the feature list for BBs
        adjacent_matrix = [[0 for i in range(len(nodes_ordered_list))] for j in range(len(nodes_ordered_list))] # adjacent matrix for CFG
        for i, node in enumerate(nodes_ordered_list):
            feature_list.append([
                len(dic[node]["String_Constant"]),
                len(dic[node]["Numberic_Constant"]),
                dic[node]["No_Tran"],
                dic[node]["No_Call"],
                dic[node]["No_Instru"],
                dic[node]["No_Arith"],
                dic[node]["No_offspring"],
            ])
            for presuccessor in dic[node]['pre']:
                p_i = nodes_ordered_list.index(presuccessor)
                adjacent_matrix[p_i][i] = 1
            # new_dic = {"func_name": dic['fun_name'],
            #            'feature_list':feature_list,
            #            'adjacent_matrix': adjacent_matrix}
        # func_dics.append(new_dic)
        func_name2features[dic['fun_name']] = {'feature_list': feature_list,
                                               'adjacent_matrix': adjacent_matrix}

    with open(str(out_file), 'w') as f:
        json.dump(func_name2features, f, indent=4)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def main():
    skip_suffix = {'.idb', '.idb64', '.id1', '.id0', '.id2', '.nam', '.til', '.i64', '.json'}
    select_bins = read_json('select_bin_paths.json')
    bar = tqdm(select_bins)
    for bin_path in bar:
        bin_path = Path(bin_path.replace('O0', 'O2'))
        if not bin_path.exists():
            continue
        bar.set_description('extracting Gemini features of {}'.format(bin_path.name))
        extract_bin(bin_path)

if __name__ == '__main__':
    main()

