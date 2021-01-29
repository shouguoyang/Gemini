# encoding:utf-8
# time 2017:8:21  15:37
# author ysg
# function : extract features from executable binary file

# revise in 20210129 for supporting IDAPro 7.x

import sys, os

sys.path.append("/usr/local/lib/python2.7/dist-packages")
from idautils import *
from idaapi import *
from idc import *
from idautils import DecodeInstruction

import networkx as nx

OPTYPEOFFSET = 1000
IMM_MASK = 0xffffffff  # 立即数的掩码
# user defined op type
o_string = o_imm + OPTYPEOFFSET
o_calls = OPTYPEOFFSET + 100
o_trans = OPTYPEOFFSET + 101  # Transfer instructions
o_arith = OPTYPEOFFSET + 102  # arithmetic instructions
# 将当前路径添加入搜索路径
sys.path.append(os.getcwd())
#
transfer_instructions = ['MOV', 'PUSH', 'POP', 'XCHG', 'IN', 'OUT', 'XLAT', 'LEA', 'LDS', 'LES', 'LAHF', 'SAHF',
                         'PUSHF', 'POPF']
arithmetic_instructions = ['ADD', 'SUB', 'MUL', 'DIV', 'XOR', 'INC', 'DEC', 'IMUL', 'IDIV', 'OR', 'NOT', 'SLL', 'SRL']
# is_type_arithmetic()
from LogRecorder import CLogRecoder

ymd = time.strftime("%Y-%m-%d", time.localtime())
logger = CLogRecoder(logfile='%s.log' % (ymd))
logger.addStreamHandler()
logger.INFO("\n---------------------\n")
IDA700 = False
logger.INFO("IDA Version {}".format(IDA_SDK_VERSION))
if IDA_SDK_VERSION >= 700:
# IDAPro 6.x To 7.x (https://www.hex-rays.com/products/ida/support/ida74_idapython_no_bc695_porting_guide.shtml)
    logger.INFO("Using IDA7xx API")
    IDA700 = True
    GetOpType = get_operand_type
    GetOperandValue = get_operand_type
    SegName = get_segm_name
    autoWait = auto_wait
    GetFunctionName = get_func_name
    import ida_pro
    Exit = ida_pro.qexit



def wait_for_analysis_to_finish():
    '''
    等待ida将二进制文件分析完毕再执行其他操作
    :return:
    '''
    autoWait()

wait_for_analysis_to_finish()

# 通过file命令获得可执行文件的位数
def get_ELF_bits(filename):
    # logger.INFO('file path and name: %s' % filename)
    import commands
    cmd = 'file -b %s' % filename
    s, o = commands.getstatusoutput(cmd)
    if s != 0:
        print 'error', s, o

    bits = o.strip().split(' ')[1]
    if (int(bits[:1]) == 32):
        return 32

    return 64


if get_ELF_bits(get_input_file_path()) == 64:
    IMM_MASK = 0xffffffffffffffff


class Attributes_BlockLevel(object):
    def __init__(self, func_t):
        self._Blocks = set()
        self._Blocks_list = []
        self._func = func_t
        self._block_boundary = {}
        self._addr_func = func_t.startEA if not IDA700 else func_t.start_ea  # first address of function
        self._name_func = str(GetFunctionName(func_t.startEA if not IDA700 else func_t.start_ea))  # GetFunctionName(startEA) returns the function name
        self._All_Calls = []
        self._G = nx.DiGraph()
        self._pre_nodes = {}
        self._CFG = {}  # key : Block startEA ; value : Block startEA of successors
        self._init_all_nodes()

        self.callee = set()  # 被该函数调用的其他函数
        self.caller = set()  # 调用该函数的其他函数集合

        # compute betweenness
        self._Betweenness = nx.betweenness_centrality(self._G)
        # self._Betweenness = {}
        # self._dijKstra()
        # normalize betweenness
        # n = len(self._Blocks)
        # if n>1:
        #     for x in self._Betweenness:
        #         self._Betweenness[x] /= (n*(n-1)/2)*1.0

        # compute offspring
        self._offspring = {}
        self.visit = set()
        # logger.INFO('computing offspring...')
        for node in self._Blocks:
            self.visit = set()
            self._offspring[node] = self.dfs(node)
            # logger.INFO('node: %s : offspring = %d' % (hex(node) , self._offspring[node]))

        logger.INFO('offspring computed!')
        # print betweenness
        # for key in self._Betweenness:
        #     logger.INFO(hex(key) + str(self._Betweenness[key]))

    # TODO: 返回被该函数调用的其他函数
    def get_callees(self):

        # function body
        return self.callee

    # TODO: 返回调用该函数的其他函数
    def get_callers(self):

        # 给定一个指令地址，返回包含该指令地址的函数的起始地址和函数名
        def get_func_including_addr(addr):
            func_list = list(Functions(addr, addr + 1))  # 得到包含该地址addr和addr+1的函数列表，取出第一个函数即为目标函数
            if len(func_list) < 1:
                return None
            func_startEA = func_list[0]
            return func_startEA, Name(func_startEA)
            pass

        # function body
        addr = self._func.startEA if not IDA700 else self._func.start_ea
        # logger.INFO("function startEA {}".format(hex(addr)))
        for ref in CodeRefsTo(addr, 1):
            ref_func = get_func_including_addr(ref)
            if ref_func:
                self.caller.add(ref_func[0])
        return self.caller

    # dfs to compute node's offspring
    # return node's offspring
    def dfs(self, node_startEA):

        if node_startEA in self.visit:
            return 0

        self.visit.add(node_startEA)
        offspring = 0
        for succ_node in self._CFG[node_startEA]:
            if succ_node not in self.visit:
                offspring += self.dfs(succ_node) + 1
        # logger.INFO('node %s returns offspring %d' % (hex(node_startEA), offspring))
        return offspring

    # returns Oprand from ea
    # inst_t.Operands : ida_ua.operands_array
    # inst_t.Op1 : ida_ua.op_t
    def _get_OpRands(self, ea):
        inst_t = DecodeInstruction(ea)
        return inst_t.Operands

    # #get Betweenness
    # def _djstra(self):
    #     self._Betweenness[self._addr_func] = 0 #首节点加入
    #     added_node_set = set()
    #     added_node_set.add(self._addr_func)
    #     #记录其前驱节点
    #     pre = {}
    #     for node in self._Blocks:
    #         pre[node] = []
    #     max_loop_time = len(pre)
    #     i = 0
    #     logger.INFO('in %s djstra running ...' % GetFunctionName(self._func.startEA))
    #     logger.INFO('first node address %s' % str(self._Betweenness))
    #     while True:
    #         not_add_node = set(self._Blocks_list) - added_node_set
    #         if len(not_add_node) == 0 :
    #             break
    #         if i > max_loop_time:
    #             logger.INFO('function  max loop !!!! please check CFG in ida')
    #             break
    #
    #         logger.INFO('not added node number:  %d' % len(not_add_node))
    #         #待优化
    #         #added_node集合数目比较大时，非常耗费计算资源
    #         for added_node in copy.deepcopy(added_node_set):
    #             for node in not_add_node:
    #                 if node in self._CFG[added_node]:
    #                     # added_node 以及其前驱节点加1
    #                     self._Betweenness[node] = 0
    #                     self._update_betweenness(added_node, pre)
    #                     pre[node].append(added_node)
    #                     added_node_set.add(node)
    #                     # logger.INFO('node %s added' % (hex(node)))
    #                     # logger.INFO('node '+ hex(node) + ' , pre_node ' + hex(added_node))
    #         i += 1
    #     self._Betweenness[self._Blocks_list[0]] = 0
    #     logger.INFO('djstra finished ...')

    # get Betweenness

    def _update_betweenness(self, added_node, pre):
        self._Betweenness[added_node] += 1
        if len(pre[added_node]) == 0:
            return
        queue = []
        queue += pre[added_node]
        while len(queue) > 0:
            node = queue.pop(0)
            queue += pre[node]
            self._Betweenness[node] += 1

        # for pre_node in pre[added_node]:
        # self._update_betweenness(pre_node, pre)

    def _add_predecessors(self, cbs, preb):
        for cb in cbs:
            if cb not in self._pre_nodes:
                self._pre_nodes[cb] = []
            self._pre_nodes[cb].append(preb)

    # initial block_boundary , get every node's range of address
    def _init_all_nodes(self):
        flowchart = FlowChart(self._func)
        for i in range(flowchart.size):
            basicblock = flowchart.__getitem__(i)
            self._Blocks.add(basicblock.startEA if not IDA700 else basicblock.start_ea)
            self._G.add_node(basicblock.startEA if not IDA700 else basicblock.start_ea)
            # 节点的前继节点
            # self._pre_nodes[basicblock.startEA if not IDA700 else basicblock.start_ea] = []
            # logger.INFO(hex(basicblock.startEA if not IDA700 else basicblock.start_ea) + ' prenode: ' + str(self._pre_nodes[basicblock.startEA if not IDA700 else basicblock.start_ea]))
            self._CFG[basicblock.startEA if not IDA700 else basicblock.start_ea] = [b.startEA if not IDA700 else b.start_ea for b in basicblock.succs()]
            for b in basicblock.succs():
                self._G.add_node(b.startEA if not IDA700 else b.start_ea)
                self._G.add_edge(basicblock.startEA if not IDA700 else basicblock.start_ea, b.startEA if not IDA700 else b.start_ea)
            self._add_predecessors([b.startEA if not IDA700 else b.start_ea for b in basicblock.succs()], basicblock.startEA if not IDA700 else basicblock.start_ea)
            self._block_boundary[basicblock.startEA if not IDA700 else basicblock.start_ea] = basicblock.endEA if not IDA700 else basicblock.end_ea
        self._Blocks_list = list(self._Blocks)
        self._Blocks_list.sort()
        # print CFG
        # for key in self._CFG:
        #     succ = [hex(node) for node in self._CFG[key]]
        #     logger.INFO('node : '+hex(key) + ' succ :' + str(succ))

    # # return the n'th operation if it is reference to memory
    # def get_op(self, ea, n, op_type):
    #     # Direct Memory Reference  (DATA)      addr
    #     if (op_type == o_mem):
    #         return GetString(GetOperandValue(ea, n))
    #     elif (op_type == o_phrase):# Memory Ref [Base Reg + Index Reg]    phrase
    #         return GetOperandValue(ea, n)

    '''
    # return the string contained in this instruction
    # if nothing , returns NULL
    # something wrong
    def get_String_in_instruction(self, ea):
        # logger.INFO('ea: ' + hex(ea) + ' inst: '+ GetDisasm(ea))
        All_strings = []
        op = 0
        op_type = GetOpType(ea, op)
        while (op_type != o_void):
            # logger.INFO( 'op: %d, op_type : %d' % (op, op_type))
            if (op_type == o_imm):
                addr = GetOperandValue(ea, op)
                if (not SegName(addr) == '.rodata'):
                    addrx = list(DataRefsFrom(addr))
                    if len(addrx) == 0:
                        op += 1
                        op_type = GetOpType(ea, op)
                        continue
                    addr = addrx[0]
                All_strings.append(GetString(addr))
                # logger.INFO("imm")
                # logger.INFO(GetString(addr))

            op += 1
            try:
                op_type = GetOpType(ea, op)
            except RuntimeError:
                print 'runtime error in', hex(ea), 'op', str(op) ,'OP_TYPE', op_type

        if (len(All_strings) == 0):
            return None

        return All_strings
'''

    # 返回该节点的前继节点的首地址
    def get_PreNodes_of_blocks(self, startEA):

        if startEA not in self._Blocks:
            return
        if startEA not in self._pre_nodes:
            return []

        return self._pre_nodes[startEA]

    # returns all Strings referenced in one block
    # return generator of Strings
    def get_All_Strings_of_Block(self, block_startEA):
        return self.get_OpValue_Block(block_startEA, my_op_type=o_string)

        '''
        All_String = []
        # address is not right
        if (block_startEA not in self._block_boundary):
            return

        strings = []
        endEA = self._block_boundary[block_startEA]
        it_code = func_item_iterator_t(self._func, block_startEA)
        ea = it_code.current()
        while (ea < endEA):
            strings = self.get_String_in_instruction(ea)
            if strings:
                All_String += strings
            # see if arrive end of the blocks
            if (not it_code.next_code()):
                break
            ea = it_code.current()

        return All_String
        '''

    # return a instruction's n'th oprand's reference
    # ea : the address of the instruction
    # n  : order of the operand , 0-the first operand
    def get_reference(self, ea, n):
        if (GetOpType(ea, n) == -1):
            return
        if (GetOpType(ea, n) == 1):
            print
            'General Register'
        if (GetOpType(ea, n) == 2):
            addr = GetOperandValue(ea, n)
            print
            'addr :', hex(Dword(addr))
            print
            ' reference'
            print
            'segment type :', GetSegmentAttr(addr, SEGATTR_TYPE)
            return GetString(Dword(addr))
        elif (GetOpType(ea, n) == 3):
            print
            'base + index'
        elif (GetOpType(ea, n) == 4):
            print
            'B+i+Displacement'
        elif (GetOpType(ea, n) == 5):
            print
            'immediate'
        elif (GetOpType(ea, n) == 6):
            print
            'far address'
        return GetOperandValue(ea, n)

    def get_AdjacencyMatrix(self):
        list = []
        for node in self._Blocks_list:
            newlist = []
            for node2 in self._Blocks_list:
                # if node2 == node:
                #     newlist.append(0)
                # else:
                if node2 in self._CFG[node]:
                    newlist.append(1)
                else:
                    newlist.append(0)
            list.append(newlist)

        return list

    # offspring means children nodes in CFG
    def get_Offspring_of_Block(self, startEA):
        if startEA not in self._Blocks_list:
            return None
        return self._offspring[startEA]

    # there is some error to be solved
    # returns the next address of instruction which are in same basic block
    def get_next_instruction_addr(self, ea):
        return next(ea)

    # get_reference_data_one_block
    def get_reference_data_one_block(self, startEA):

        # address is not right
        if (startEA not in self._block_boundary):
            return

        endEA = self._block_boundary[startEA]
        it_code = func_item_iterator_t(self._func, startEA)
        ea = it_code.current()
        while (ea < endEA):
            yield (''.join(self.get_instruction(ea)))

            # see if arrive end of the blocks
            if (not it_code.next_code()):
                break
            ea = it_code.current()

    # get the whole instruction
    def get_instruction(self, ea):
        return idc.GetDisasm(ea)

    def get_Trans_of_block(self, ea):
        return len(self.get_OpValue_Block(ea, o_trans))

    # startEA:basicblock's start address
    # return all instruction in one block
    # it is replaced by function get_reference_data_one_block
    def get_All_instr_in_one_block(self, startEA):

        return list(self.get_reference_data_one_block(startEA))

        '''
        #
        # instr_list = []
        # if (startEA not in self._block_boundary):
        #     return instr_list
        #
        # endEA = self._block_boundary[startEA]
        # it_code = func_item_iterator_t(self._func, startEA)
        # ea = it_code.current()
        # while ((ea) < endEA):
        #     newlist = []
        #     newlist.append(ua_mnem(ea))
        #     i = 0
        #     op = GetOpnd(ea, i)
        #     while not op == "":
        #         newlist.append(op)
        #         i += 1
        #         op = GetOpnd(ea, i)
        #
        #     instr_list.append(newlist)
        #     if (not it_code.next_code()):
        #         break
        #     ea = it_code.current()
        #
        # return instr_list
        '''

    # return function's name
    def getFuncName(self):
        return self._name_func

    def FrameSize(self):
        return GetFrameSize(self._func.startEA if not IDA700 else self._func.start_ea)  # get full size of function frame

    def getHexAddr(self, addr):
        return hex(addr)

    def FrameArgsSize(self):  # get size of arguments in function frame which are purged upon return
        return GetFrameArgsSize(self._func.startEA if not IDA700 else self._func.start_ea)

    def FrameRegsSize(self):  # get size of
        return GetFrameRegsSize(self._func.startEA if not IDA700 else self._func.start_ea)

    # get operand value in one block
    def get_OpValue_Block(self, startEA, my_op_type):
        OPs = []
        # address is not right
        if (startEA not in self._block_boundary):
            return

        endEA = self._block_boundary[startEA]
        it_code = func_item_iterator_t(self._func, startEA)
        ea = it_code.current()
        while (ea < endEA):
            OPs += self.get_OpValue(ea, my_op_type)
            # see if arrive end of the blocks
            if (not it_code.next_code()):
                break
            ea = it_code.current()

        return OPs

    def get_Arithmetics_Of_Block(self, ea):
        return len(self.get_OpValue_Block(ea, o_arith))

    # return all function or api names called by this function
    def get_Calls_BLock(self, ea):
        return len(self.get_OpValue_Block(ea, o_calls))
        # ref = xrefblk_t()
        #
        # for ea in self._Blocks_list:
        #     it = func_item_iterator_t(self._func, ea)
        #
        #     while it.next_code():
        #         nea = it.current()
        #         logger.INFO('address :' + hex(nea))
        #         frm = [hex(x.frm) for x in XrefsFrom(nea)]
        #         logger.INFO(str(frm))

    # this is an abstract interface
    # it can replace functions like get_Numeric_Constant
    def get_OpValue(self, ea, my_op_type=o_void):
        OV = []

        # instruction level features
        if (my_op_type == o_trans):
            # it's a transfer instruction if data transfered between reg and mem
            # logger.INFO('disasm:' + GetDisasm(ea))
            inst = GetDisasm(ea).split(' ')[0].upper()
            if (inst in transfer_instructions):
                # logger.INFO('in trans')
                OV.append(inst)
            return OV

        elif (my_op_type == o_arith):
            inst = GetDisasm(ea).split(' ')[0].upper()
            # logger.INFO('disasm:' + GetDisasm(ea))
            if (inst in arithmetic_instructions):
                # logger.INFO('in arithmetic')
                OV.append(inst)
            return OV

        op = 0
        op_type = GetOpType(ea, op)
        while (op_type != o_void):

            # o_calls
            if (my_op_type == o_calls):
                # logger.INFO("disasm : " + GetDisasm(ea))
                if (GetDisasm(ea).split(' ')[0].upper() == "CALL"):
                    # logger.INFO('in o_calls : ' + self.get_instruction(ea))
                    OV.append(GetDisasm(ea).split(' ')[-1])
                    break

            if (op_type == my_op_type % OPTYPEOFFSET):
                ov = GetOperandValue(ea, op)
                ov &= 0xffffffff  # 强制转化成32位
                if (my_op_type == o_imm):
                    # if SegName(ov) == "":
                    #     OV.append(ov)
                    logger.INFO(hex(ea) + ' imm : ' + hex(ov))
                    if ov != 0:
                        OV.append(hex(ov))
                elif (my_op_type == o_string):
                    if (not SegName(ov) == '.rodata'):
                        addrx = list(DataRefsFrom(ov))
                        if len(addrx) == 0:
                            op += 1
                            op_type = GetOpType(ea, op)
                            continue
                        ov = addrx[0]
                    OV.append(GetString(ov))

            op += 1
            op_type = GetOpType(ea, op)
        return OV

    '''
    #return the Numeric Constants in the linear address ea
    def get_Numeric_Constants(self, ea):
        # op_enum()
        Con = []
        op = 0
        op_type = GetOpType(ea, op)
        while ( op_type != o_void ):

            if (op_type == o_imm):
                if (SegName(GetOperandValue(ea, op)) == ""):# if the immediate number is not an address
                    Con.append(GetOperandValue(ea, op))

            op += 1
            op_type = GetOpType(ea, op)

        logger.INFO( "get_Numeric_Constants : " + self.get_instruction(ea) +' : '+ str(Con) )
        return Con
    '''

    # get immediate num in blocks
    def get_Numeric_Constants_One_block(self, startEA):
        return self.get_OpValue_Block(startEA, my_op_type=o_imm)

    # get Betweenness of Blocks
    def get_Betweenness_of_Block(self, startEA):
        if startEA not in self._Betweenness:
            return -0
        return self._Betweenness[startEA]

    def get_CFG(self):
        return self._CFG

    '''
    def getCFG_OF_Func(self):
        # get the Control Flow Graph of the function , return a list in the format of [(current_block_startaddr:next_block_startaddr), ......]
        # if a function has only one node , it's cfg may be empty
        # flowchart for a function
        flowchart = FlowChart(self._func)
        list = []
        for i in range(flowchart.size):
            basicblock = flowchart.__getitem__(i)
            suc = basicblock.succs()
            for item in suc:
                list.append(((basicblock.startEA if not IDA700 else basicblock.start_ea), (item.startEA)))
                # print basicblock.id,hex(basicblock.startEA if not IDA700 else basicblock.start_ea),hex(basicblock.endEA if not IDA700 else basicblock.end_ea)
        return list
    '''

    # return all the start address of basicblock in form of set
    def get_All_Nodes_StartAddr(self):
        return self._Blocks_list

    # return a blocks end address
    def get_Block_Endaddr(self, startEA):
        if (startEA in self._block_boundary):
            return self._block_boundary[startEA]
        return -1


# print how to use this script
def print_help():
    help = 'args not enough'
    print(help)


# get block attributes
# return a dic
# which have keys : startEA, String_Constant, Numberic_Constant, No_Tran, No_Call, No_Instru, No_Arith, No_offspring, Betweenness
def get_att_block(blockEA, Attribute_Block):
    AB = Attribute_Block
    dic = {}
    dic['startEA'] = blockEA
    dic['String_Constant'] = AB.get_All_Strings_of_Block(blockEA)
    dic['Numberic_Constant'] = AB.get_Numeric_Constants_One_block(blockEA)
    dic['No_Tran'] = AB.get_Trans_of_block(blockEA)
    dic['No_Call'] = AB.get_Calls_BLock(blockEA)
    dic['No_Instru'] = len(AB.get_All_instr_in_one_block(blockEA))
    dic['No_Arith'] = AB.get_Arithmetics_Of_Block(blockEA)
    dic['No_offspring'] = AB.get_Offspring_of_Block(blockEA)
    dic['Betweenness'] = round(AB.get_Betweenness_of_Block(blockEA), 3)
    dic['pre'] = [hex(ea) for ea in AB.get_PreNodes_of_blocks(blockEA)]
    return dic


def save_Json(filename, func_name):
    for i in range(0, get_func_qty()):
        fun = getn_func(i)
        segname = get_segm_name(fun.startEA if not IDA700 else fun.start_ea)
        if segname[1:3] not in ["OA", "OM", "te"]:
            continue
        if (func_name != '' and GetFunctionName(fun.startEA if not IDA700 else fun.start_ea) != func_name):
            continue
        with open(filename, 'a') as f:
            AB = Attributes_BlockLevel(fun)
            logger.INFO(AB.getFuncName())
            CFG = AB.get_CFG()
            dic = {}
            dic['fun_name'] = AB.getFuncName()
            dic['adjacentmat'] = AB.get_AdjacencyMatrix()
            for ea in CFG:
                dic[hex(ea)] = get_att_block(ea, AB)
                dic[hex(ea)]['succ'] = []
                for succ_ea in CFG[ea]:
                    dic[hex(ea)]['succ'].append(hex(succ_ea))

            # logger.INFO('dic' + str(dic))
            json.dump(dic, f, ensure_ascii=False)

            f.write('\n')


def main():
    if len(idc.ARGV) < 2:
        return
    func_name = ''  # 待提取特征的函数名，''空字符当成提取全部函数的特征
    filename = idc.ARGV[1]
    if len(idc.ARGV) >= 3:
        func_name = idc.ARGV[2]

    # filename = get_root_filename()
    # filename = filename + '.json'
    save_Json(filename, func_name)
    return

    if len(idc.ARGV) < 0:
        print_help()
        return
    set_seg = set()
    for i in range(0, get_func_qty()):
        fun = getn_func(i)  # get_func returns a func_t struct for the function
        segname = get_segm_name(
            fun.startEA if not IDA700 else fun.start_ea)  # get the segment name of the function by address ,x86 arch segment includes (_init _plt _plt_got _text extern _fini)
        if segname[1:3] not in ["OA", "OM", "te"]:
            continue
        # print p_func.getCFG_OF_Func()
        # print p_func.getAll_Nodes_Addr()
        # for item in p_func.getAll_Nodes_Addr():
        # print hex(item),hex(p_func.get_Nodes_Endaddr(item))
        if (GetFunctionName(fun.startEA if not IDA700 else fun.start_ea) != 'main'):
            p_func = Attributes_BlockLevel(fun)
            logger.INFO(p_func.getFuncName())
            # p_func.get_All_Calls()
            allnodes = p_func.get_All_Nodes_StartAddr()
            for ea in allnodes:
                logger.INFO('block start' + hex(ea))
                logger.INFO('block offspring:' + str(p_func.get_Offspring_of_Block(ea)))
                logger.INFO('block betweenness :' + str(p_func.get_Betweenness_of_Block(ea)))
                # logger.INFO('block instructions :' + str(len(p_func.get_All_instr_in_one_block(ea))))
                # logger.INFO(p_func.get_reference_data_one_block(ea).next())
                # logger.INFO('String: ' + str(p_func.get_All_Strings_of_Block(ea)))
                # logger.INFO(p_func.get_Numeric_Constants_One_block(ea))
                # calls = p_func.get_Calls_BLock(ea)
                # logger.INFO(calls)
                # logger.INFO('trans number: ' + str(p_func.get_Trans_of_block(ea)))
                # logger.INFO('arithmetics :' + str(p_func.get_Arithmetics_Of_Block(ea)))


# test for callers
# 2018年12月25日10:19:07 测试通过
def test_caller():
    # 测试使用代码块 上
    for func in Functions():
        AB = Attributes_BlockLevel(func_t(func))
        func_callers = AB.get_callers()
        print "function {} {}".format(hex(func), Name(func))
        for caller_ea in func_callers:
            print "callers {} {}".format(hex(caller_ea), Name(caller_ea))
    # 测试使用代码块 下


# def test_callee():

# do something within one function
import json

if __name__ == '__main__':
    # test_caller()
    try:
        main()
    except Exception as e:
        import traceback
        logger.INFO(traceback.format_exc())
    Exit(0)