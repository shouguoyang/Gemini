#encoding=utf-8
#无需学会idapython 的使用，直接调用该类下的接口即可获得函数
#系统ida所在的路径
idapath = 'idat64'
import os,time,commands,json
import argparse

parse = argparse.ArgumentParser()
import sys
pro_path = sys.path[0]

class getFeature:
    def __init__(self, binarypath):
        self._bin = binarypath
        self._tmpfile = pro_path + os.sep + binarypath.split('/')[-1] + str(time.time()) + '.json'

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

    def get_Feature_Function(self, func_name):

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

def test(args):

    binary_path = args.binaryfile
    # generate ida database file
    func_name = ''
    out_file = ''
    if args.f:
        func_name = args.f
    if args.o:
        out_file = args.o

    gf = getFeature(binary_path)
    feature = gf.get_Feature_Function(func_name)



    if len(out_file) > 0:
        func_dics = []
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
                new_dic = {"func_name": dic['fun_name'],
                           'feature_list':feature_list,
                           'adjacent_matrix': adjacent_matrix}
                func_dics.append(new_dic)

        with open(out_file, 'w') as f:
            json.dump(func_dics, f, indent=4)
    else:
        for x in feature:
            print x


if __name__ == '__main__':
    parse.add_argument('binaryfile', help='file to be analysed')
    parse.add_argument('-f', help='function name to be handled ')
    parse.add_argument('-b', help='file to be analysed is binary file , default is ida database file', action = 'store_true' , default=False)
    parse.add_argument('-o', help='output filename')
    args = parse.parse_args()
    test(args)

