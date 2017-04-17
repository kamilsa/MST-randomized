# venv/bin/python

import os
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
from algs.algs import *
from algs.dsa import *
import numpy as np
from matplotlib.legend_handler import HandlerLine2D


def get_graph(filename, with_root=False):
    DG = nx.DiGraph()
    f = open(filename, 'r')
    line = None
    edges = []
    coordinates = []
    terms = []
    if with_root:
        root = None
    while line != 'EOF':
        line = f.readline().strip()
        toks = line.split(' ')
        if toks[0] == 'A':
            t = tuple(int(x) for x in toks[1:])
            edges.append(t)
        if toks[0] == 'T':
            terms.append(int(toks[1]))
        if toks[0] == 'Root':
            if with_root:
                root = int(toks[1])
        if toks[0] == 'DD':
            t = tuple(int(x) for x in toks[1:])
            coordinates.append(t)
    for coord in coordinates:
        DG.add_node(coord[0], pos=(coord[1], coord[2]))
    terms.sort()
    DG.add_weighted_edges_from(edges)
    # print_graph(DG)
    # nx.draw(DG, node_size=50)
    # plt.show()
    # f.close()
    if with_root:
        return DG, terms, root
    else:
        print_graph(DG)
        max_len = 0
        max_node = None
        for node in nx.nodes(DG):
            # print(node, tr_cl.out_edges(node))
            descs = nx.descendants(DG, node)
            # desc_numb = len(descs)
            if len(set(terms) & set(descs)) == len(descs):
                # max_len = desc_numb
                max_node = node
        if max_len == len(nx.nodes(DG)):
            return DG, terms, max_node
        else:
            reachable = set(nx.descendants(DG, max_node)) | {max_node}
            unreachable = set(nx.nodes(DG)) - reachable
            for node in unreachable:
                DG.remove_node(node)
        terms = list(set(terms) & reachable)
        print('terms =', len(terms))
        return DG, terms, max_node


# save times in format: <vertex number> <edge number> <alg> <time> <res>
def save_time(v, e, terms, alg, t, res):
    f = open('vi2time.txt', 'r')
    f_strs = f.readlines()
    f.close()
    row_to_put = -1
    i = 0
    for f_str in f_strs:
        toks = f_str.strip().split(' ')
        if toks[0] == str(v) and toks[1] == str(e) and toks[2] == str(alg) and toks[3] == str(terms):
            row_to_put = i
        i += 1
    if row_to_put == -1:
        f_strs.append(str(v) + " " + str(e) + " " + str(terms) + " " + str(alg) + " " + str(t) + " " + str(res) + '\n')
    else:
        f_strs[row_to_put] = str(v) + " " + str(e) + " " + str(terms) + " " + str(alg) + " " + str(t) + " " + str(
            res) + "\n"
    f = open('vi2time.txt', 'w')
    f.writelines(f_strs)
    f.close()


def induce_graph(DG=nx.DiGraph(), n=0, t=0, root=1):  # induce graph to have n nodes
    bfs_list = list(set(sum(list(nx.algorithms.bfs_tree(DG, root).edges()), ())))
    t = int(t)
    bfs_list = bfs_list[:n]
    DG = DG.subgraph(bfs_list)
    import random
    terms = random.sample(set(nx.nodes(DG)) - {root}, t)
    return DG, terms


def graph_test(filename):
    print('')
    print('Getting graph..')
    DG, terms, root = get_graph(filename, with_root=True)
    print('Getting graph is finished')
    print("")
    terms = list(set(terms) - {root})
    # DG, terms = get_graph('WRP4/wrp4-11.stp')
    print_graph(DG)
    v = nx.number_of_nodes(DG)
    e = nx.number_of_edges(DG)
    print("Number of vertices: ", v)
    print("Number of reachable vertices: ", len(nx.descendants(DG, root)) + 1)
    print("Number of edges: ", e)
    print('')
    print('apsp started')
    start_time = time.time()
    tr_cl = trans_clos(DG)
    elapsed_time = time.time() - start_time
    print('apsp finished in', elapsed_time)
    # print_graph(tr_cl)
    max_len = 0
    max_node = None
    for node in nx.nodes(tr_cl):
        # print(node, tr_cl.out_edges(node))
        if len(tr_cl.out_edges(node)) > max_len:
            max_len = len(tr_cl.out_edges(node))
            max_node = node
    print("max node ", max_node)
    print("intersect", set(v for x, v in tr_cl.out_edges(max_node)) & set(terms))
    i = 1
    print('Alg6 with i = ', i, 'started')
    start_time = time.time()
    set_start_time(start_time)
    terms.sort()
    tree = alg6(tr_cl, i=2, k=len(terms), r=root, x=terms)
    elapsed_time = time.time() - start_time
    print('Elapsed time = ', elapsed_time)
    tot_weight = tree.size(weight='weight')
    print('Weight of MSTw = ', tot_weight)
    print_graph(tree)
    exit()
    prev = dict()
    for i in [1, 2]:
        # try:
        #     if not (('alg3-' + str(i)) not in prev or prev[('alg3-' + str(i))]):
        #         raise Exception('')
        #     raise Exception()
        #     print('alg3-' + str(i), 'started..')
        #     start_time = time.time()
        #     set_start_time(start_time)
        #     tree = alg3(tr_cl, i=i, k=len(terms.copy()), r=root, x=terms.copy())
        #     elapsed_time = time.time() - start_time
        #     tot_weight = tot_weight = tree.size(weight='weight')
        #     print('alg3-' + str(i), 'finished in', elapsed_time, 'with res =', tot_weight)
        #     print('')
        #     save_time(v, e, 'alg3-' + str(i), elapsed_time, tot_weight)
        #     prev['alg3-' + str(i)] = True
        # except:
        #     save_time(v, e, 'alg3-' + str(i), '-', '-')
        #     print('Alg took to long to compute')
        #     prev['alg3-' + str(i)] = False
        # try:
        #     if not (('alg4-' + str(i)) not in prev or prev[('alg3-' + str(i))]):
        #         raise Exception('')
        #     raise Exception()
        #     print('alg4-' + str(i), 'started..')
        #     start_time = time.time()
        #     set_start_time(start_time)
        #     tree = alg4(tr_cl, i=i, k=len(terms.copy()), r=root, x=terms.copy())
        #     elapsed_time = time.time() - start_time
        #     tot_weight = tree.size(weight='weight')
        #     print('alg4-' + str(i), 'finished in', elapsed_time, 'with res =', tot_weight)
        #     print('')
        #     save_time(v, e, 'alg4-' + str(i), elapsed_time, tot_weight)
        #     prev['alg4-' + str(i)] = True
        # except:
        #     save_time(v, e, 'alg4-' + str(i), '-', '-')
        #     print('Alg took to long to compute')
        #     prev['alg4-' + str(i)] = False
        # try:
        if not (('alg6-' + str(i)) not in prev or prev[('alg6-' + str(i))]):
            raise Exception('')
        print('alg6-' + str(i), 'started..')
        start_time = time.time()
        set_start_time(start_time)
        tree = alg6(tr_cl, i=i, k=len(terms.copy()), r=root, x=terms.copy())
        elapsed_time = time.time() - start_time
        tot_weight = tree.size(weight='weight')
        print('alg6-' + str(i), 'finished in', elapsed_time, 'with res =', tot_weight)
        print('')
        save_time(v, e, 'alg6-' + str(i), elapsed_time, tot_weight)
        prev['alg6-' + str(i)] = True
        # except:
        #     save_time(v, e, 'alg6-' + str(i), '-', '-')
        #     print('Alg took to long to compute')
        #     prev['alg6-' + str(i)] = False


def wrp_test(filename=None, g=None, terms=None, root=None):
    global prev
    if g is None:
        # print('')
        # print('Getting graph..')
        DG, terms, root = get_graph(filename, with_root=True)
        # print_graph(DG)

        v = nx.number_of_nodes(DG)
        e = nx.number_of_edges(DG)

        print('root is', root)
        print("Number of vertices: ", v)
        print("Number of reachable vertices: ", len(nx.descendants(DG, root)) + 1)
        print("Number of edges: ", e)
        print('')
        print('apsp started')
        start_time = time.time()
        tr_cl = trans_clos_dense(DG)
        # print_graph(tr_cl)
        elapsed_time = time.time() - start_time
        print('apsp finished in', elapsed_time)

        terms = list(set(terms) - {root})
        terms.sort()

        i = 2
        print('Alg6 with i = ', i, 'started')
        start_time = time.time()
        set_start_time(start_time)
        terms.sort()
        tree = alg3(tr_cl, i=4, k=len(terms), r=root, x=terms)
        elapsed_time = time.time() - start_time
        print('Elapsed time = ', elapsed_time)
        tot_weight = tree.size(weight='weight')
        print('Weight of MSTw = ', tot_weight)
        print_graph(tree)
        exit()
    else:
        DG = g

        v = nx.number_of_nodes(DG)
        e = nx.number_of_edges(DG)

        print('root is', root)
        print("Number of vertices: ", v)
        print("Number of reachable vertices: ", len(nx.descendants(DG, root)) + 1)
        print("Number of edges: ", e)
        print('')
        print('apsp started')
        start_time = time.time()
        tr_cl = trans_clos_dense(DG)
        # print_graph(tr_cl)
        elapsed_time = time.time() - start_time
        print('apsp finished in', elapsed_time)

        terms = list(set(terms) - {root})
        terms.sort()

    for i in [4]:
        try:
            if not (('alg3-' + str(i)) not in prev or prev[('alg3-' + str(i))]):
                raise Exception('')
            print('alg3-' + str(i), 'started..')
            start_time = time.time()
            set_start_time(start_time)
            tree = alg3(tr_cl, i=i, k=len(terms.copy()), r=root, x=terms.copy())
            elapsed_time = time.time() - start_time
            tot_weight = tree.size(weight='weight')
            print('alg3-' + str(i), 'finished in', elapsed_time, 'with res =', tot_weight)
            print('')
            save_time(v, e, len(terms), 'alg3-' + str(i), elapsed_time, tot_weight)
            prev['alg3-' + str(i)] = True
        except:
            save_time(v, e, len(terms), 'alg3-' + str(i), '-', '-')
            print('Alg took to long to compute')
            prev['alg3-' + str(i)] = False
        try:
            if not (('alg4-' + str(i)) not in prev or prev[('alg3-' + str(i))]):
                raise Exception('')
            print('alg4-' + str(i), 'started..')
            start_time = time.time()
            set_start_time(start_time)
            tree = alg4(tr_cl, i=i, k=len(terms.copy()), r=root, x=terms.copy())
            elapsed_time = time.time() - start_time
            tot_weight = tree.size(weight='weight')
            print('alg4-' + str(i), 'finished in', elapsed_time, 'with res =', tot_weight)
            print('')
            save_time(v, e, len(terms), 'alg4-' + str(i), elapsed_time, tot_weight)
            prev['alg4-' + str(i)] = True
        except:
            save_time(v, e, len(terms), 'alg4-' + str(i), '-', '-')
            print('Alg took to long to compute')
            prev['alg4-' + str(i)] = False
        try:
            if not (('alg6-' + str(i)) not in prev or prev[('alg6-' + str(i))]):
                raise Exception('')
            print('alg6-' + str(i), 'started..')
            start_time = time.time()
            set_start_time(start_time)
            tree = alg6(tr_cl, i=i, k=len(terms.copy()), r=root, x=terms.copy())
            elapsed_time = time.time() - start_time
            tot_weight = tree.size(weight='weight')
            print('alg6-' + str(i), 'finished in', elapsed_time, 'with res =', tot_weight)
            print('')
            save_time(v, e, len(terms), 'alg6-' + str(i), elapsed_time, tot_weight)
            prev['alg6-' + str(i)] = True
        except:
            save_time(v, e, len(terms), 'alg6-' + str(i), '-', '-')
            print('Alg took to long to compute')
            prev['alg6-' + str(i)] = False


def plot_vsize2time():
    def get_power_str(n, p):
        return r'$' + str(n) + '^' + str(p) + '$'

    f = open('vsize2time.txt', 'r')
    xx = []
    yy = []
    line = f.readline()
    max_y = -1
    stats = []
    while line:
        toks = line.strip().split(' ')
        if toks[4] != '-':
            t = {'v': int(toks[0]), 'e': int(toks[1]), 'terms': int(toks[2]), 'alg': toks[3], 't': float(toks[4]),
                 'res': float(toks[5])}
            stats.append(t)
        line = f.readline()
    f.close()
    sizes = []  # number of vertices
    alg3_1 = []  # times for alg3-1
    alg3_2 = []
    alg4_2 = []
    alg6_2 = []
    for t in stats:
        if t['v'] not in sizes:
            sizes.append(t['v'])
        if t['alg'] == 'alg3-1':
            alg3_1.append(t['t'])
        if t['alg'] == 'alg3-2':
            alg3_2.append(t['t'])
        if t['alg'] == 'alg4-2':
            alg4_2.append(t['t'])
        if t['alg'] == 'alg6-2':
            alg6_2.append(t['t'])
    yy = sizes
    alg3_1 = np.array(alg3_1)
    alg3_2 = np.array(alg3_2)
    alg4_2 = np.array(alg4_2)
    alg6_2 = np.array(alg6_2)
    fig, ax = plt.subplots()
    # ax.plot(x, y[0], 'r-')
    # ax.plot(x, y2, 'ro')

    ax.plot(yy[:len(alg3_1)], np.log10(alg3_1), 'r-', marker='^')
    ax.plot(yy[:len(alg3_2)], np.log10(alg3_2), 'g-', marker='^')
    ax.plot(yy[:len(alg4_2)], np.log10(alg4_2), 'k-', marker='s')
    ax.plot(yy[:len(alg6_2)], np.log10(alg6_2), 'b-', marker='o')
    # set ticks and tick labels
    ax.set_xlim((0, 6000))
    ax.set_xticks(range(0, 6000, 1000))
    ax.set_xticklabels(range(0, 6000, 1000))
    ax.set_ylim((0, np.log10(100000)))
    # ax.set_ylim((0, max_y))
    # print(np.log10(alg6_2))
    ax.set_yticks(np.log10([1 / 10000, 1 / 100, 1, 100, 10000]))
    # print(range(0,4,0.5))
    labels = [get_power_str(10, x) for x in [-4, -2, 0, 2, 4]]
    ax.set_yticklabels(labels)

    # Only draw spine between the y-ticks
    # ax.spines['left'].set_bounds(-1, 1)
    # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    line31, = plt.plot([], marker='^', label='Alg3-1', color='red')
    line32, = plt.plot([], marker='^', label='Alg3-2', color='green')
    line42, = plt.plot([], marker='s', label='Alg4-2', color='black')
    line62, = plt.plot([], marker='o', label='Alg6-2', color='blue')
    plt.legend(handler_map={line31: HandlerLine2D(numpoints=1), line32: HandlerLine2D(numpoints=1),
                            line42: HandlerLine2D(numpoints=1),
                            line62: HandlerLine2D(numpoints=1)}, loc=4)
    # plt.show()
    plt.savefig('vsize2time2')


def plot_ve2time():
    def get_power_str(n, p):
        return r'$' + str(n) + '^' + str(p) + '$'

    f = open('statistics.txt', 'r')
    xx = []
    yy = []
    line = f.readline()
    max_y = -1
    stats = []
    while line:
        toks = line.strip().split(' ')
        if toks[4] != '-':
            t = {'v': int(toks[0]), 'e': int(toks[1]), 'terms': int(toks[2]), 'alg': toks[3], 't': float(toks[4]),
                 'res': float(toks[5])}
            stats.append(t)
        line = f.readline()
    f.close()
    sizes = []  # number of vertices
    alg6_1 = []  # times for alg6-1
    alg6_2 = []
    for t in stats:
        if t['e'] / t['v'] not in sizes:
            sizes.append(t['e'] / t['v'])
        if t['alg'] == 'alg6-1':
            alg6_1.append(t['t'])
        if t['alg'] == 'alg6-2':
            alg6_2.append(t['t'])
    yy = [int(size) for size in sizes]
    alg6_1 = np.array(alg6_1)
    alg6_2 = np.array(alg6_2)
    fig, ax = plt.subplots()
    # ax.plot(x, y[0], 'r-')
    # ax.plot(x, y2, 'ro')

    ax.plot(yy[:len(alg6_1)], np.log10(alg6_1), 'r-', marker='^')
    ax.plot(yy[:len(alg6_2)], np.log10(alg6_2), 'b-', marker='o')
    # set ticks and tick labels
    ax.set_xlim((8, 50))
    ax.set_xticks(yy)

    ax.set_ylim((0, np.log10(100000)))
    # ax.set_ylim((0, max_y))
    # print(np.log10(alg6_2))
    ax.set_yticks(np.log10([1 / 10000, 1 / 100, 1, 100, 10000]))
    # print(range(0,4,0.5))
    labels = [get_power_str(10, x) for x in [-4, -2, 0, 2, 4]]
    ax.set_yticklabels(labels)

    # Only draw spine between the y-ticks
    # ax.spines['left'].set_bounds(-1, 1)
    # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    line42, = plt.plot([], marker='^', label='Alg6-1', color='red')
    line62, = plt.plot([], marker='o', label='Alg6-2', color='blue')
    plt.legend(handler_map={line42: HandlerLine2D(numpoints=1),
                            line62: HandlerLine2D(numpoints=1)}, loc=1)
    # plt.show()
    plt.savefig('ve2time')


def plot_vk2time():
    def get_power_str(n, p):
        return r'$' + str(n) + '^' + str(p) + '$'

    f = open('vk2time.txt', 'r')
    xx = []
    yy = []
    line = f.readline()
    max_y = -1
    stats = []
    while line:
        toks = line.strip().split(' ')
        if toks[4] != '-':
            t = {'v': int(toks[0]), 'e': int(toks[1]), 'terms': int(toks[2]), 'alg': toks[3], 't': float(toks[4]),
                 'res': float(toks[5])}
            stats.append(t)
        line = f.readline()
    f.close()
    sizes = []  # number of vertices
    alg6_2 = []
    for t in stats:
        if t['terms'] not in sizes:
            sizes.append(t['terms'])
        if t['alg'] == 'alg6-3':
            alg6_2.append(t['t'])
    yy = [int(size) for size in sizes]
    alg6_2 = np.array(alg6_2)
    fig, ax = plt.subplots()
    # ax.plot(x, y[0], 'r-')
    # ax.plot(x, y2, 'ro')

    ax.plot(yy[:len(alg6_2)], alg6_2, 'b-', marker='o')
    # set ticks and tick labels
    ax.set_xlim((0, 50))
    ax.set_xticks(range(0, 50, 5))

    ax.set_ylim((0, 1000))
    # ax.set_ylim((0, max_y))
    # print(np.log10(alg6_2))
    ax.set_yticks(range(0, 1000, 50))
    # print(range(0,4,0.5))
    labels = range(0, 1000, 50)
    ax.set_yticklabels(labels)

    # Only draw spine between the y-ticks
    # ax.spines['left'].set_bounds(-1, 1)
    # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    line62, = plt.plot([], marker='o', label='Alg6-3', color='blue')
    plt.legend(handler_map={line62: HandlerLine2D(numpoints=1)}, loc=4)
    # plt.show()
    plt.savefig('vk2time')


def plot_vi2time():
    def get_power_str(n, p):
        return r'$' + str(n) + '^' + str(p) + '$'

    f = open('vi2time.txt', 'r')
    xx = []
    yy = []
    line = f.readline()
    max_y = -1
    stats = []
    while line:
        toks = line.strip().split(' ')
        if toks[4] != '-':
            t = {'v': int(toks[0]), 'e': int(toks[1]), 'terms': int(toks[2]), 'alg': toks[3], 't': float(toks[4]),
                 'res': float(toks[5])}
            stats.append(t)
        line = f.readline()
    f.close()
    sizes = []  # number of vertices
    alg_3 = []
    alg_4 = []
    alg_6 = []
    for t in stats:
        # if t['terms'] not in sizes:
        #     sizes.append(t['terms'])
        print(t['alg'][:4])
        if t['alg'][:4] == 'alg3':
            alg_3.append(t['t'])
        if t['alg'][:4] == 'alg4':
            alg_4.append(t['t'])
        if t['alg'][:4] == 'alg6':
            alg_6.append(t['t'])
    # yy = [int(size) for size in sizes]
    yy = [1, 2, 3, 4]
    alg_3 = np.array(np.log10(alg_3))
    alg_4 = np.array(np.log10(alg_4))
    alg_6 = np.array(np.log10(alg_6))
    fig, ax = plt.subplots()
    # ax.plot(x, y[0], 'r-')
    # ax.plot(x, y2, 'ro')
    ax.plot(yy[:len(alg_3)], alg_3, 'r-', marker='^')
    ax.plot(yy[:len(alg_4)], alg_4, 'k-', marker='s')
    ax.plot(yy[:len(alg_6)], alg_6, 'b-', marker='o')
    # set ticks and tick labels
    ax.set_xlim((1, 4))
    ax.set_xticks(range(1, 4))

    ax.set_ylim((0, np.log10(100)))
    # ax.set_ylim((0, max_y))
    # print(np.log10(alg6_2))
    ax.set_yticks(np.log10([1 / 10000, 1 / 100, 1, 100]))
    # print(range(0,4,0.5))
    labels = [get_power_str(10, x) for x in [-4, -2, 0, 2]]
    ax.set_yticklabels(labels)

    # Only draw spine between the y-ticks
    # ax.spines['left'].set_bounds(-1, 1)
    # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    line3, = plt.plot([], marker='^', label='Alg3', color='red')
    line4, = plt.plot([], marker='s', label='Alg4', color='black')
    line6, = plt.plot([], marker='o', label='Alg6', color='blue')
    plt.legend(handler_map={line3: HandlerLine2D(numpoints=1), line4: HandlerLine2D(numpoints=1),
                            line6: HandlerLine2D(numpoints=1)}, loc=4)
    # plt.show()
    plt.savefig('vi2time')


# for filename in sorted(os.listdir('statics/'), key=lambda x: int(x.split('_')[0]))[-1:]:
#     graph_test('statics/' + filename)
# graph_test('statics/1465_2184.txt')
# exit()
# graph_test('WRP/wrp4-16.stp')
# wrp_test('statics/35_42.txt')

# prev = dict()
# g, terms, root = get_graph('relay-medium/12509.stp', with_root=True)
# for size in [10, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12509]:
#     dg, terms = induce_graph(g, n=size, t=size/10, root=root)
#     wrp_test(g=dg, terms=terms, root=root)

# for filename in os.listdir('relay-small/'):
#     g, terms, root = get_graph('relay-small/' + filename, with_root=True)
#     dg, terms = induce_graph(g, n=320, t=256, root=root)
#     wrp_test(g=dg, terms=terms, root=root)

# plot_vsize2time()
# plot_ve2time()
plot_vk2time()
# plot_vi2time()
