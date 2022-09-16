#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------File Info-----------------------
Name: extract_feature_from_text.py
Description: 从文本形式数据中提取Gemini所需特征
Author: GentleCP
Email: me@gentlecp.com
Create Date: 2022/9/2
-----------------End-----------------------------
"""
from pathlib2 import Path
from collections import OrderedDict, defaultdict
import json
import networkx as nx
import numpy as np

arch2ins2type = {
    # refer: https://zhuanlan.zhihu.com/p/53394807
    'x86': {
        # trans
        'mov': 'No_Tran',
        'push': 'No_Tran',
        'pop': 'No_Tran',
        'xchg': 'No_Tran',
        'in': 'No_Tran',
        'out': 'No_Tran',
        'xlat': 'No_Tran',
        'lea': 'No_Tran',
        'lds': 'No_Tran',
        'les': 'No_Tran',
        'lahf': 'No_Tran',
        'sahf': 'No_Tran',
        'pushf': 'No_Tran',
        'popf': 'No_Tran',
        # No_Arith ins
        'add': 'No_Arith',
        "adc": 'No_Arith',
        "adcx": 'No_Arith',
        "adox": 'No_Arith',
        "sbb": 'No_Arith',
        'sub': 'No_Arith',
        'mul': 'No_Arith',
        'div': 'No_Arith',
        'inc': 'No_Arith',
        'dec': 'No_Arith',
        'imul': 'No_Arith',
        'idiv': 'No_Arith',
        'cmp': 'No_Arith',
        "neg": 'No_Arith',
        "daa": 'No_Arith',
        "das": 'No_Arith',
        "aaa": 'No_Arith',
        "aas": 'No_Arith',
        "aam": 'No_Arith',
        "aad": 'No_Arith',
        # call ins
        "call": 'No_Call',
    },
    'arm': {
        # No_Tranfer ins arm
        "b": 'No_Tran',
        "bal": 'No_Tran',
        "bne": 'No_Tran',
        "beq": 'No_Tran',
        "bpl": 'No_Tran',
        "bmi": 'No_Tran',
        "bcc": 'No_Tran',
        "blo": 'No_Tran',
        "bcs": 'No_Tran',
        "bhs": 'No_Tran',
        "bvc": 'No_Tran',
        "bvs": 'No_Tran',
        "bgt": 'No_Tran',
        "bge": 'No_Tran',
        "blt": 'No_Tran',
        "ble": 'No_Tran',
        "bhi": 'No_Tran',
        "bls": 'No_Tran',
        # No_Arith ins
        "add": 'No_Arith',
        "adc": 'No_Arith',
        "qadd": 'No_Arith',
        "sub": 'No_Arith',
        "sbc": 'No_Arith',
        "rsb": 'No_Arith',
        "qsub": 'No_Arith',
        "mul": 'No_Arith',
        "mla": 'No_Arith',
        "mls": 'No_Arith',
        "umull": 'No_Arith',
        "umlal": 'No_Arith',
        "smull": 'No_Arith',
        "smlal": 'No_Arith',
        "udiv": 'No_Arith',
        "sdiv": 'No_Arith',
        "cmp": 'No_Arith',
        "cmn": 'No_Arith',
        "tst": 'No_Arith',
        # No_Call ins
        "bl": 'No_Call',
    },
    'mips': {
        # No_Tranfer ins mips
        "beqz": 'No_Tran',
        "beq": 'No_Tran',
        "bne": 'No_Tran',
        "bgez": 'No_Tran',
        "b": 'No_Tran',
        "bnez": 'No_Tran',
        "bgtz": 'No_Tran',
        "bltz": 'No_Tran',
        "blez": 'No_Tran',
        "bgt": 'No_Tran',
        "bge": 'No_Tran',
        "blt": 'No_Tran',
        "ble": 'No_Tran',
        "bgtu": 'No_Tran',
        "bgeu": 'No_Tran',
        "bltu": 'No_Tran',
        "bleu": 'No_Tran',
        # No_Arith ins
        "add": 'No_Arith',
        "addu": 'No_Arith',
        "addi": 'No_Arith',
        "addiu": 'No_Arith',
        "and": 'No_Arith',
        "andi": 'No_Arith',
        "div": 'No_Arith',
        "divu": 'No_Arith',
        "mult": 'No_Arith',
        "multu": 'No_Arith',
        "slt": 'No_Arith',
        "sltu": 'No_Arith',
        "slti": 'No_Arith',
        "sltiu": 'No_Arith',
        # call ins
        "jal": "No_Call"
    },
    'ppc': {
        # No_Tran inf ppc
        "b": 'No_Tran',
        "blt": 'No_Tran',
        "beq": 'No_Tran',
        "bge": 'No_Tran',
        "bgt": 'No_Tran',
        "blr": 'No_Tran',
        "bne": 'No_Tran',
        # No_Arith ins
        "add": 'No_Arith',
        "addi": 'No_Arith',
        "addme": 'No_Arith',
        "addze": 'No_Arith',
        "neg": 'No_Arith',
        "subf": 'No_Arith',
        "subfic": 'No_Arith',
        "subfme": 'No_Arith',
        "subze": 'No_Arith',
        "mulhw": 'No_Arith',
        "mulli": 'No_Arith',
        "mullw": 'No_Arith',
        "divw": 'No_Arith',
        "cmp": 'No_Arith',
        "cmpi": 'No_Arith',
        "cmpl": 'No_Arith',
        "cmpli": 'No_Arith',
        # call ins
        "bl": "No_Call",
    }
}

ARCH = ''


def read_json(fname):
    with open(fname, 'r') as f:
        return json.loads(f.read())


def write_json(content, fname):
    with open(fname, 'w') as f:
        json.dump(content, f)


def get_succs_and_graph(cfg, blocks):
    block_size = blocks[-1][0] + 1
    # matrix = np.zeros((block_size, block_size))
    G = nx.DiGraph()
    for edge in cfg:
        # matrix[edge[0], edge[1]] = 1
        G.add_node(edge[0])
        G.add_node(edge[1])
        G.add_edge(edge[0], edge[1])

    succs = []
    for block in blocks:
        block_id = block[0]
        try:
            succs.append(list(G.successors(block_id)))
        except nx.NetworkXError:
            succs.append([])
    return succs, G


def get_off_spring(graph):
    visit = set()

    def dfs(block_id):
        if block_id in visit:
            return 0
        visit.add(block_id)
        off_spring = 0
        for succ_node in graph.successors(block_id):
            if succ_node not in visit:
                off_spring += dfs(succ_node) + 1
        return off_spring

    block_id2off_spring = {}

    for block in graph.nodes():
        block_id2off_spring[block] = dfs(block)

    return block_id2off_spring


def get_betweenness(graph):
    """
    :param graph:
    :return: betweenness: block_id: btns
    """
    return nx.betweenness_centrality(graph)


def _scan_codes(codes, arch):
    """
    提供一段代码片段（list，通常是一个block内的），统计其中的属性信息
    :param codes:
    :param arch:
    :return:
    """
    attrs = defaultdict(float)
    num_constants = []
    ins2type = arch2ins2type[arch]
    for code in codes:
        try:
            ins, operands = code.split('\t')
        except ValueError:
            ins = code
            operands = ''
        if ins in ins2type:
            attrs[ins2type[ins]] += 1
        attrs['No_Instru'] += 1
        data = operands.split(',')
        for e in data:
            e = e.strip()
            try:
                int(e, 16)
            except ValueError:
                continue
            else:
                attrs['Numberic_Constant'] += 1
                # num_constants.append(e)
    # attrs['Numberic_Constant'] = num_constants
    attrs['String_Constant'] = 0
    return attrs


def get_block_attrs(codes, blocks, G, arch):
    block_ptr, code_ptr = 0, 0
    block_id2codes = defaultdict(list)
    while block_ptr < len(blocks):
        try:
            code = codes[code_ptr]
        except IndexError:
            break
        cur_block = blocks[block_ptr]
        block_id, start_ea, end_ea = cur_block[0], cur_block[1], cur_block[2]
        if start_ea <= code[0] < end_ea:
            block_id2codes[block_id].append(code[2])
            code_ptr += 1
        else:
            block_ptr += 1

    block_id2off_spring = get_off_spring(G)
    # block_id2betweenness = get_betweenness(G)

    block_features = []
    for i, (block_id, codes) in enumerate(block_id2codes.items()):
        # codes: [[sd	$ra, 0x28($sp)]]
        block_attrs = _scan_codes(codes, arch=arch)
        block_attrs.update({'No_offspring': block_id2off_spring.get(block_id, 0),
                            # 'Betweenness': block_id2betweenness[block_id]
                            })
        # block_attrs['pre'] = list(G.predecessors(block_id))
        # block_attrs['succ'] = list(G.successors(block_id))
        block_features.append(
            (
                block_attrs['String_Constant'],
                block_attrs['Numberic_Constant'],
                block_attrs['No_Tran'],
                block_attrs['No_Call'],
                block_attrs['No_Instru'],
                block_attrs['No_Arith'],
                block_attrs['No_offspring'],
                # block_attrs['Betweenness'],  # gemini没用到
            )
        )
    while len(block_features) <= blocks[-1][0]:
        block_features.append((0, 0, 0, 0, 0, 0, 0))
    return block_features


def _get_arch(org_arch):
    if org_arch.startswith('x86'):
        return 'x86'
    elif org_arch.startswith('arm'):
        return 'arm'
    elif org_arch.startswith('mips'):
        return 'mips'
    else:
        return 'ppc'


def get_gemini_feature(func_info):
    succs, G = get_succs_and_graph(cfg=func_info['cfg'], blocks=func_info['block'])
    feature_list = get_block_attrs(codes=func_info['code'],
                                   blocks=func_info['block'],
                                   G=G,
                                   arch=_get_arch(func_info['arch']))

    return {
        'fid': func_info['fid'],
        'n_num': func_info['block'][-1][0] + 1,
        'features': feature_list,
        'succs': succs,
    }

def test_gemini_feature_extract():
    func_sample = read_json('tmp/block_size_error.json')
    res = get_gemini_feature(func_sample)
    print(res)

if __name__ == '__main__':
    test_gemini_feature_extract()
