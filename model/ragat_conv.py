# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 6:45 PM
# @Author  : liuxiyang
from turtle import color
from helper import *
from model.message_passing import MessagePassing
from model.SpecialSpmmFinal import SpecialSpmmFinal
import torch.nn as nn


class RagatConv(MessagePassing):
    def __init__(self, edge_index, edge_type, in_channels, out_channels, num_rels, act=lambda x: x, params=None,
                 head_num=1,nx_g=None):
        super(self.__class__, self).__init__()

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = None
        self.head_num = head_num
        self.nx_g = nx_g

        self.w1_loop = get_param((in_channels, out_channels))
        self.w1_in = get_param((in_channels, out_channels))
        self.w1_out = get_param((in_channels, out_channels))
        self.w_rel = get_param((in_channels, out_channels))

        if self.p.opn == 'concat' or self.p.opn == 'cross_concat':
            self.w1_loop = get_param((2 * in_channels, out_channels))
            self.w1_in = get_param((2 * in_channels, out_channels))
            self.w1_out = get_param((2 * in_channels, out_channels))

        self.loop_rel = get_param((1, in_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))
        self.special_spmm = SpecialSpmmFinal()

        self.w_att_head1 = get_param((out_channels, 1))

        num_edges = self.edge_index.size(1) // 2
        if self.device is None:
            self.device = self.edge_index.device
        self.in_index, self.out_index = self.edge_index[:, :num_edges], self.edge_index[:, num_edges:]
        self.in_type, self.out_type = self.edge_type[:num_edges], self.edge_type[num_edges:]
        self.loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)]).to(self.device)
        self.loop_type = torch.full((self.p.num_ent,), 2 * self.num_rels, dtype=torch.long).to(self.device)
        # E * 1, norm A
        num_ent = self.p.num_ent
        self.in_norm = None if self.p.att else self.compute_norm(self.in_index, num_ent)
        self.out_norm = None if self.p.att else self.compute_norm(self.out_index, num_ent)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.rel_weight1 = get_param((2 * self.num_rels + 1, in_channels))
        if self.head_num == 2 or self.head_num == 3:
            self.w2_loop = get_param((in_channels, out_channels))
            self.w2_in = get_param((in_channels, out_channels))
            self.w2_out = get_param((in_channels, out_channels))

            if self.p.opn == 'concat' or self.p.opn == 'cross_concat':
                self.w2_loop = get_param((2 * in_channels, out_channels))
                self.w2_in = get_param((2 * in_channels, out_channels))
                self.w2_out = get_param((2 * in_channels, out_channels))
            self.w_att_head2 = get_param((out_channels, 1))
            self.rel_weight2 = get_param((2 * self.num_rels + 1, in_channels))

        if self.head_num == 3:
            self.w3_loop = get_param((in_channels, out_channels))
            self.w3_in = get_param((in_channels, out_channels))
            self.w3_out = get_param((in_channels, out_channels))
            if self.p.opn == 'concat' or self.p.opn == 'cross_concat':
                self.w3_loop = get_param((2 * in_channels, out_channels))
                self.w3_in = get_param((2 * in_channels, out_channels))
                self.w3_out = get_param((2 * in_channels, out_channels))
            self.w_att_head3 = get_param((out_channels, 1))
            self.rel_weight3 = get_param((2 * self.num_rels + 1, in_channels))

    def forward(self, x, rel_embed):
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        # 2 * num_ent
        in_res1 = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                 rel_weight=self.rel_weight1, edge_norm=self.in_norm, mode='in', w_str='w1_{}')
        loop_res1 = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                   rel_weight=self.rel_weight1, edge_norm=None, mode='loop', w_str='w1_{}')
        out_res1 = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                  rel_weight=self.rel_weight1, edge_norm=self.out_norm, mode='out', w_str='w1_{}')
        if self.head_num == 2 or self.head_num == 3:
            in_res2 = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                     rel_weight=self.rel_weight2, edge_norm=self.in_norm, mode='in', w_str='w2_{}')
            loop_res2 = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                       rel_weight=self.rel_weight2, edge_norm=None, mode='loop', w_str='w2_{}')
            out_res2 = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                      rel_weight=self.rel_weight2, edge_norm=self.out_norm, mode='out', w_str='w2_{}')
        if self.head_num == 3:
            in_res3 = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                     rel_weight=self.rel_weight3, edge_norm=self.in_norm, mode='in', w_str='w3_{}')
            loop_res3 = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                       rel_weight=self.rel_weight3, edge_norm=None, mode='loop', w_str='w3_{}')
            out_res3 = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                      rel_weight=self.rel_weight3, edge_norm=self.out_norm, mode='out', w_str='w3_{}')
        if self.p.att:
            out1 = self.agg_multi_head(in_res1, out_res1, loop_res1, 1,nx_g=self.nx_g)
            if self.head_num == 2:
                out2 = self.agg_multi_head(in_res2, out_res2, loop_res2, 2,nx_g=self.nx_g)
                out = 1 / 2 * (out1 + out2)
            elif self.head_num == 3:
                out2 = self.agg_multi_head(in_res2, out_res2, loop_res2, 2,nx_g=self.nx_g)
                out3 = self.agg_multi_head(in_res3, out_res3, loop_res3, 3,nx_g=self.nx_g)
                out = 1 / 3 * (out1 + out2 + out3)
            else:
                out = out1
        else:
            out = self.drop(in_res1) * (1 / 3) + self.drop(out_res1) * (1 / 3) + loop_res1 * (1 / 3)
        if self.p.bias:
            out = out + self.bias
        relation1 = rel_embed.mm(self.w_rel)
        out = self.bn(out)
        entity1 = self.act(out)

        return entity1, relation1[:-1]

    def agg_multi_head(self, in_res, out_res, loop_res, head_no,nx_g=None):
        att_weight = getattr(self, 'w_att_head{}'.format(head_no))
        edge_index = torch.cat([self.edge_index, self.loop_index], dim=1)
        # print(f"Edge idxs size = {edge_index.size()}")
        all_message = torch.cat([in_res, out_res, loop_res], dim=0)
        # print(f"Attention weights size = {self.w_att_head1.size()}")
        powers = -self.leakyrelu(all_message.mm(att_weight).squeeze()) #power of each message m(u,r,v)
        # edge_exp: E * 1
        edge_exp = torch.exp(powers).unsqueeze(1)
        
        # if nx_g is not None:
        # self.vis_attention(nx_g, edge_exp)

        weight_rowsum = self.special_spmm(
            edge_index, edge_exp, self.p.num_ent, self.p.num_ent, 1, dim=1)
        # except 0
        weight_rowsum[weight_rowsum == 0.0] = 1.0
        # weight_rowsum: num_nodes x 1
        # info_emb_weighted: E * D
        edge_exp = self.drop(edge_exp)
        info_emb_weighted = edge_exp * all_message
        # assert not torch.isnan(info_emb_weighted).any()
        emb_agg = self.special_spmm(
            edge_index, info_emb_weighted, self.p.num_ent, self.p.num_ent, all_message.shape[1], dim=1)
        # emb_agg: N x D, finish softmax
        emb_agg = emb_agg.div(weight_rowsum)
        assert not torch.isnan(emb_agg).any()
        return emb_agg

    def vis_attention(self, nx_g, edge_exp):
        jsonFilePath1 = r'data/ent_to_id.json'
        jsonFilePath2 = r'data/id_to_ent.json'
        att_to_plot = edge_exp[:(self.edge_index.size(1))//2,:]
        att_to_plot = torch.flatten(att_to_plot).tolist()
        # print(f"Required weights size = {len(att_to_plot)}")
        # pos = nx.spring_layout(nx_g,seed=100)
        # fig, ax = plt.subplots()
        # edge_list_t, att, node_list,labels = self.plot_preds(jsonFilePath1, jsonFilePath2,att_to_plot)
        self.plot_preds(nx_g,jsonFilePath1, jsonFilePath2,att_to_plot)

        # print(f"Len of nodelist = {len(node_list)}")
        # print(f"Starting elements of edge idxs = {edge_list[:10]}") #real eg. [(0, 1), (2, 3)]
        # plot_graph(nx_g, att_to_plot[:1000],ax,nodes_pos=pos, nodes_to_plot=node_list,
        #                         edges_to_plot=edge_list)

        # cmap = plt.cm.plasma
        # nodes = nx.draw_networkx_nodes(nx_g, pos, node_size=10, node_color="indigo",
        #                                 nodelist=node_list,alpha=0.9,ax=ax)
        # edges = nx.draw_networkx_edges(nx_g,pos,arrowstyle="->",edgelist=edge_list_t,
        #                                 arrowsize=5,edge_color=att,
        #                                 edge_cmap=cmap,width=2, alpha=0.5,ax=ax)
        # nx.draw_networkx_labels(nx_g,pos,labels=labels,font_size=6)
        
        # pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        # pc.set_array(att)
        # cbar = plt.colorbar(pc)
        # cbar.set_label('Attention', labelpad=5)

        # ax = plt.gca()
        # plt.show()

        # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
        # sm.set_array([])
        # plt.colorbar(sm, fraction=0.046, pad=0.01)

        # ax.set_axis_off()
        # plt.savefig('plots/graph.png')

    def plot_preds(self, nx_g,jsonFilePath1, jsonFilePath2,att_to_plot):
        nodes_wanted = ['N05BA12']
        # nodes_wanted = ['401.2']
        idxs = []
        with open(jsonFilePath1) as ent_file:
            data = json.load(ent_file)
            idxs = [data[i.lower()] for i in nodes_wanted]
            print(f"The req ids are = {idxs}")
        # edge_list = self.in_index[:,:100]
        edge_list = self.in_index.tolist()
        # edge_list_t = []
        att = []
        wt_edges = []
        for i in idxs:
            for j in range(len(edge_list[0])):
                if i == edge_list[0][j] or i == edge_list[1][j]:
                    # edge_list_t.append((edge_list[0][j],edge_list[1][j]))
                    att.append(att_to_plot[j])
                    nx_g.add_edge(edge_list[0][j],edge_list[1][j],color=att_to_plot[j])
                    # wt_edges.append((edge_list[0][j],edge_list[0][j],att_to_plot[j]))

        # for j in range(len(edge_list[0])):
        #     wt_edges.append((edge_list[0][j],edge_list[1][j],att_to_plot[j]))
        
        # nx_g.add_weighted_edges_from(wt_edges)
        idxs_n = [n for n in nx_g[idxs[0]]] #to get neighbours
        print(f"Neighbour idxs = {idxs_n}")
        # print(nx_g.edges())
        wt_edges_n = []
        pred_obj = 550
        for i in idxs_n:
            for j in range(len(edge_list[0])):
                if (i == edge_list[0][j] and pred_obj == edge_list[1][j]) or \
                        (pred_obj == edge_list[0][j] and i == edge_list[1][j]):
                    # edge_list_t.append((edge_list[0][j],edge_list[1][j]))
                    att.append(att_to_plot[j])
                    nx_g.add_edge(edge_list[0][j],edge_list[1][j],color=att_to_plot[j])
                    # wt_edges_n.append((edge_list[0][j],edge_list[1][j],att_to_plot[j]))
        # nx_g.add_weighted_edges_from(wt_edges_n)
        # # edge_list_tuple = [i for i in zip(edge_list[0],edge_list[1])] #to make [(u,v),(w,x)] original edge list
        # node_list = list(set.union(*map(set,edge_list_t)))
        labels = {}
        with open(jsonFilePath2) as id_file:
            data = json.load(id_file)
            # for i in node_list:
            #     labels[i] = data[str(i)].upper()
            for i in nx_g.nodes():
                labels[i] = data[str(i)].upper()
                nx_g.nodes[i]['label'] = data[str(i)].upper()
            # print(f"The req labels are = {labels}")
        
        # print(f"The edge list = {edge_list_t}")
        # print(f"The att list = {att}")
        # print(f"The node list = {node_list}")
        
        # fig, ax = plt.subplots()
        # nx_g.add_edge(1037,550,color=0.9)
        # att.append(0.9)
        pos = nx.spring_layout(nx_g,seed=100)
        # cmap = plt.cm.plasma
        # nodes = nx.draw_networkx_nodes(nx_g, pos, node_size=10, node_color="indigo",
        #                                 nodelist=nx_g.nodes(),alpha=0.9,ax=ax)
        # edges = nx.draw_networkx_edges(nx_g,pos,arrowstyle="->",
        #                                 arrowsize=5,edgelist=nx_g.edges(),
        #                                 edge_cmap=cmap,width=2, alpha=0.5,ax=ax)
        # # nx.draw_networkx_labels(nx_g,pos,labels=labels,font_size=6)
        
        # pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        # pc.set_array(att_to_plot)
        # cbar = plt.colorbar(pc)
        # cbar.set_label('Attention', labelpad=5)
        # ax.set_axis_off()
        options = {
        "node_color": "#A0CBE2",
        "edge_color": att,
        "width": 4,
        "edge_cmap": plt.cm.Blues,
        }
        # print(f"Att values = {att}")
        nx.draw(nx_g, pos, **options)
        nx.draw_networkx_labels(nx_g,pos,labels=labels,font_size=6)

        plt.savefig('plots/graph.png')

        nt = Network(height='750px', width='100%')
        nt.from_nx(nx_g)
        nt.show('nt.html')
        # return edge_list_t,att,node_list,labels

    def rel_transform(self, ent_embed, rel_embed, rel_weight, opn=None):
        if opn is None:
            opn = self.p.opn
        if opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif opn == 'corr_ra':
            trans_embed = ccorr(ent_embed * rel_weight, rel_embed)
        elif opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif opn == 'es':
            trans_embed = ent_embed
        elif opn == 'sub_ra':
            trans_embed = ent_embed * rel_weight - rel_embed
        elif opn == 'mult':
            trans_embed = ent_embed * rel_embed
        elif opn == 'mult_ra':
            trans_embed = (ent_embed * rel_embed) * rel_weight
        elif opn == 'cross':
            trans_embed = ent_embed * rel_embed * rel_weight + ent_embed * rel_weight
        elif opn == 'cross_wo_rel':
            trans_embed = ent_embed * rel_weight
        elif opn == 'cross_simplfy':
            trans_embed = ent_embed * rel_embed + ent_embed
        elif opn == 'concat':
            trans_embed = torch.cat([ent_embed, rel_embed], dim=1)
        elif opn == 'concat_ra':
            trans_embed = torch.cat([ent_embed, rel_embed], dim=1) * rel_weight
        elif opn == 'ent_ra':
            trans_embed = ent_embed * rel_weight + rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, rel_weight, edge_norm, mode, w_str):
        weight = getattr(self, w_str.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        rel_weight = torch.index_select(rel_weight, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb, rel_weight)
        out = torch.mm(xj_rel, weight)
        assert not torch.isnan(out).any()
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float().unsqueeze(1)
        deg = self.special_spmm((row.cpu().numpy(), col.cpu().numpy()), edge_weight, num_ent, num_ent, 1, dim=1)
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
