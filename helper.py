# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 6:41 PM
# @Author  : liuxiyang
import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
import matplotlib.pyplot as plt
import matplotlib as mpl
import optuna
from optuna.trial import TrialState
import seaborn as sns
import networkx as nx
from pyvis.network import Network

np.set_printoptions(precision=4)


def set_gpu(gpus):
    """
    Sets the GPU to be used for the run

    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results['count'])
    count_r1 = float(left_results['count_r1'])
    count_r2 = float(left_results['count_r2'])
    count_r3 = float(left_results['count_r3'])
    count_r4 = float(left_results['count_r4'])

    results['mr_r1'] = round(left_results['mr_r1'] / count_r1, 5)
    results['mr_r2'] = round(left_results['mr_r2'] / count_r2, 5)
    results['mr_r3'] = round(left_results['mr_r3'] / count_r3, 5)
    results['mr_r4'] = round(left_results['mr_r4'] / count_r4, 5)
    results['mrr_r1'] = round(left_results['mrr_r1'] / count_r1, 5)
    results['mrr_r2'] = round(left_results['mrr_r2'] / count_r2, 5)
    results['mrr_r3'] = round(left_results['mrr_r3'] / count_r3, 5)
    results['mrr_r4'] = round(left_results['mrr_r4'] / count_r4, 5)

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round(
            (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
    return results


def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def plot_graph(g,attention,ax, nodes_to_plot=None, nodes_labels=None,
         edges_to_plot=None, nodes_pos=None, nodes_colors=None,
         edge_colormap=plt.cm.Reds):
    """
    Visualize edge attentions by coloring edges on the graph.
    g: nx.DiGraph
        Directed networkx graph
    attention: list
        Attention values corresponding to the order of sorted(g.edges())
    ax: matplotlib.axes._subplots.AxesSubplot
        ax to be used for plot
    nodes_to_plot: list
        List of node ids specifying which nodes to plot. Default to
        be None. If None, all nodes will be plot.
    nodes_labels: list, numpy.array
        nodes_labels[i] specifies the label of the ith node, which will
        decide the node color on the plot. Default to be None. If None,
        all nodes will have the same canonical label. The nodes_labels
        should contain labels for all nodes to be plot.
    edges_to_plot: list of 2-tuples (i, j)
        List of edges represented as (source, destination). Default to
        be None. If None, all edges will be plot.
    nodes_pos: dictionary mapping int to numpy.array of size 2
        Default to be None. Specifies the layout of nodes on the plot.
    nodes_colors: list
        Specifies node color for each node class. Its length should be
        bigger than number of node classes in nodes_labels.
    edge_colormap: plt.cm
        Specifies the colormap to be used for coloring edges.
    """
    if nodes_to_plot is None:
        nodes_to_plot = sorted(g.nodes())
    if edges_to_plot is None:
        assert isinstance(g, nx.DiGraph), 'Expected g to be an networkx.DiGraph' \
                                          'object, got {}.'.format(type(g))
        edges_to_plot = sorted(g.edges())
        print(f"Should not be printed!")
        
    # print(f"Starting elements of edges_to_plot = {edges_to_plot[:10]}")
    print(f"Length of edgelist = {len(edges_to_plot)} and length of att = {len(attention)}")
    print(f"Length of nodelist = {len(nodes_to_plot)}")

    nx.draw_networkx_edges(g, nodes_pos, edgelist=edges_to_plot,
                            edge_color=attention,edge_cmap=edge_colormap,
                           width=2, alpha=0.5, ax=ax, edge_vmin=0,
                           edge_vmax=1)

    # if nodes_colors is None:
    #     print(max(nodes_labels))
    #     nodes_colors = sns.color_palette("deep", max(nodes_labels) + 1)

    nx.draw_networkx_nodes(g, nodes_pos, nodelist=nodes_to_plot, ax=ax, node_size=10,
                         alpha=0.9)
    nx.draw_networkx_labels(g,nodes_pos,labels=nodes_labels,font_size=6)