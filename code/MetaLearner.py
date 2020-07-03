# coding: utf-8
# author: lu yf
# create date: 2019-12-10 14:25
import torch
from torch.nn import functional as F


class MetaLearner(torch.nn.Module):
    def __init__(self,config, user_emb, item_emb):
        super(MetaLearner, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = 32 + config['item_embedding_dim']
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']
        self.config = config

        self.item_emb = item_emb
        self.user_emb = user_emb

        # prediction parameters
        self.vars = torch.nn.ParameterDict()
        self.vars_bn = torch.nn.ParameterList()

        w1 = torch.nn.Parameter(torch.ones([self.fc2_in_dim,self.fc1_in_dim]))  # 64, 96
        torch.nn.init.xavier_normal_(w1)
        self.vars['ml_fc_w1'] = w1
        self.vars['ml_fc_b1'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w2 = torch.nn.Parameter(torch.ones([self.fc2_out_dim,self.fc2_in_dim]))
        torch.nn.init.xavier_normal_(w2)
        self.vars['ml_fc_w2'] = w2
        self.vars['ml_fc_b2'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w3 = torch.nn.Parameter(torch.ones([1, self.fc2_out_dim]))
        torch.nn.init.xavier_normal_(w3)
        self.vars['ml_fc_w3'] = w3
        self.vars['ml_fc_b3'] = torch.nn.Parameter(torch.zeros(1))

        # # # parameters for [Wu, Wi]
        # user_w = torch.nn.Parameter(torch.ones([32,config['user_embedding_dim']+32]))
        # torch.nn.init.xavier_normal_(user_w)
        # self.vars['user_w'] = user_w
        # self.vars['user_b'] = torch.nn.Parameter(torch.zeros(32))
        # #
        # # item_w = torch.nn.Parameter(torch.ones([32,config['item_embedding_dim']]))
        # # torch.nn.init.xavier_normal_(item_w)
        # # self.vars['item_w'] = item_w
        # # self.vars['item_b'] = torch.nn.Parameter(torch.zeros(32))

    def forward(self, x, user_neigh_emb, vars_dict=None):
        """

        :param x:
        :param user_neigh_emb: (#sample, user_emb_dim), under single mp or fusion of multi-mps.
        :param vars_dict:
        :return:
        """
        if vars_dict is None:
            vars_dict = self.vars
        # user_emb = self.user_emb(x[:, self.config['item_fea_len']:])  # (#sample, #fea_user*emb_dim=128)
        item_emb = self.item_emb(x[:, 0:self.config['item_fea_len']])  # (#sample, #fea_item*emb_dim=64)

        # enhanced_emb = F.leaky_relu(F.linear(torch.cat((user_emb, user_neigh_emb), 1),
        #                                           vars_dict['user_w'], vars_dict['user_b']))

        # w_user_emb = F.linear(user_emb, vars_dict['user_w'], vars_dict['user_b'])
        # w_item_emb = F.linear(item_emb, vars_dict['item_w'], vars_dict['item_b'])
        # cat_self_neigh_emb = F.leaky_relu(w_user_emb * w_item_emb + user_neigh_emb)

        x_i = item_emb
        # x_u = user_emb  # movielens: loss:1.2.. down! mae:0.84, perfect! ; dbook: 2.2... up!, 10epoch,mae: 3.466,
        # x_u = user_enhanced_emb  # add self,  movielens: loss:1.3.. down! mae:0.831 , perfect! ;; dbook(lr=0.005): 1.52... down!, after 5epoch loss up (6.6)!!! then down, 10epoch,mae: 3.552;
        x_u = user_neigh_emb  # movielens: loss:12.14... up! ; dbook 20epoch: user_cold: mae 0.6051;
        # x_u = cat_self_neigh_emb  # movielens: bad!!
        #
        # # [Wu,Wi]
        # x_i = F.linear(item_emb, vars_dict['item_w'], vars_dict['item_b'])
        # x_u = F.linear(user_neigh_emb, vars_dict['user_w'], vars_dict['user_b'])

        x = torch.cat((x_i, x_u), 1)  # ?, item_emb_dim+user_emb_dim+user_emb_dim
        x = F.relu(F.linear(x, vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1']))
        x = F.relu(F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2']))
        x = F.linear(x, vars_dict['ml_fc_w3'], vars_dict['ml_fc_b3'])
        return x.squeeze()

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class MetapathLearner(torch.nn.Module):
    def __init__(self,config, user_emb, item_emb):
        super(MetapathLearner, self).__init__()
        self.config = config

        self.item_emb = item_emb
        self.user_emb = user_emb

        # meta-path parameters
        self.vars = torch.nn.ParameterDict()
        # w = torch.nn.Parameter(torch.ones([32,config['user_embedding_dim']]))  # 128+64, 128
        # torch.nn.init.xavier_normal_(w)
        # self.vars['user_w'] = w
        # self.vars['user_b'] = torch.nn.Parameter(torch.zeros(32))

        neigh_w = torch.nn.Parameter(torch.ones([32,config['item_embedding_dim']]))  # dim=32, movielens 0.81006
        torch.nn.init.xavier_normal_(neigh_w)
        self.vars['neigh_w'] = neigh_w
        self.vars['neigh_b'] = torch.nn.Parameter(torch.zeros(32))

        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def mp_neighbor_agg(self, con_neigh_fea, vars_dict):
        """
        :param con_neigh_fea:
        :param vars_dict:
        :return:
        """
        neigh_emb = self.item_emb(con_neigh_fea)  # (#neigh, #item_emb_dim)
        neigh_emb = F.linear(neigh_emb, vars_dict['neigh_w'], vars_dict['neigh_b'])
        neigh_emb = F.leaky_relu(torch.mean(neigh_emb, dim=0).view(1, -1))  # (1,  #fea * emb_dim)
        return neigh_emb  # (1,  #fea * emb_dim)

    def forward(self, x, mp_neighbors, mp, vars_dict=None):
        """
        :param x: tensor, shape = [#sample, #user_fea+item_fea]
        :param mp_neighbors: list, len = #sample, each element is a Tensor, shape = [#neighbors of user based on mp, #item_fea]
        :param mp: str
        :param vars_dict:
        :param training:
        :return: aggregated embedding of neighbors based on mp, shape = [#sample, #user_emb]
        """
        if vars_dict is None:
            vars_dict = self.vars
        # user_emb = self.user_emb(x[:, self.config['item_fea_len']:])
        # item_emb = self.item_emb(x[:, 0:self.config['item_fea_len']])

        # put all neighbors together, no u,i emb, is better than single (no u i emb), mae at 10epoch 0.60451
        # put all neighbors together, + u i emb, worse than single + u i emb
        neighs_emb = self.item_emb(torch.cat(mp_neighbors))  # (#neighbors in all samples, item_emb_dim)
        agg_neighbor_emb = F.linear(neighs_emb, vars_dict['neigh_w'], vars_dict['neigh_b'])  # (#neighbors, item_emb_dim)
        agg_neighbor_emb = F.leaky_relu(torch.mean(agg_neighbor_emb, 0))  # (user_emb_dim)
        # agg_neighbor_emb = F.leaky_relu(F.linear(user_emb[0], vars_dict['neigh_w'], vars_dict['neigh_b']) +
        #                                 torch.mean(agg_neighbor_emb, 0)).repeat(x.shape[0], 1)  # add self
        # sim = self.cos(user_emb[0], agg_neighbor_emb)
        return agg_neighbor_emb.repeat(x.shape[0], 1)

        # neighs_emb = torch.cat([self.mp_neighbor_agg(i, vars_dict) for i in mp_neighbors],dim=0)  # (#sample in a task, item_emb_dim)
        # w_user_emb = F.linear(user_emb, vars_dict['user_w'], vars_dict['user_b'])
        # w_item_emb = F.linear(item_emb, vars_dict['neigh_w'], vars_dict['neigh_b'])
        # cat_self_neigh_emb = F.leaky_relu(w_user_emb * w_item_emb + neighs_emb)
        # return cat_self_neigh_emb  # (#sample, 64)

        # neighs_emb = torch.cat([self.mp_neighbor_agg(i, vars_dict) for i in mp_neighbors],dim=0)  # (#sample in a task, item_emb_dim)
        # # return F.leaky_relu(F.linear(neighs_emb, vars_dict['mp_fc_w'], vars_dict['mp_fc_b']))  # (#samples, #user_emb_dim)

        # # each mean, then att agg
        # neighs_emb = self.item_emb(torch.cat(mp_neighbors))  # (#neighbors in all samples, item_emb_dim)
        # agg_neighbor_emb = F.linear(neighs_emb, vars_dict['neigh_w'],vars_dict['neigh_b'])  # (#neighbors, item_emb_dim)
        #
        # index_list = map(lambda _: _.shape[0], mp_neighbors)
        # output_emb = []
        # start = 0
        # for idx in index_list:
        #     end = start+idx
        #     output_emb.append(F.leaky_relu(torch.mean(agg_neighbor_emb[start:end],0)))
        #     start = end
        # output_emb = torch.stack(output_emb, 0)  # (#sample, dim)
        # return output_emb

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class AggLearner(torch.nn.Module):
    def __init__(self, config, user_emb, item_emb):
        super(AggLearner, self).__init__()
        self.config = config
        self.mp_num = len(self.config['mp'])
        self.item_emb = item_emb
        self.user_emb = user_emb

        self.vars = torch.nn.ParameterDict()

        # w = torch.nn.Parameter(torch.ones([config['user_embedding_dim'], config['user_embedding_dim']]))  # 128, 64
        # torch.nn.init.xavier_normal_(w)
        # self.vars['agg_w'] = w
        # self.vars['agg_b'] = torch.nn.Parameter(torch.zeros(config['user_embedding_dim']))

    def forward(self, x, mp_enhanced_embs, mp_att, vars_dict=None, training=True):
        """

        :param x:
        :param mp_enhanced_embs: list, (#sample in a task, 64=neigh_emb_dim)
        :param mp_att: dict
        :param vars_dict:
        :param training:
        :return:
        """
        # if vars_dict is None:
        #     vars_dict = self.vars
        # agg_mp_emb = torch.stack(mp_enhanced_embs).mean(0)  # after stack (#mp, #sample, user_emb_dim), after mean (#sample, user_emb_dim)
        # # return F.linear(agg_mp_emb, vars_dict['agg_w'], vars_dict['agg_b'])  # movielens is bad, no leaky_relu is better, must zhushi initialization
        # return agg_mp_emb

        agg_mp_emb = torch.stack(mp_enhanced_embs, 1)  # (#sample, #mp, user_emb_dim)
        return torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)  # (#sample, embedding_dim*3)

        #
        # agg_mp_emb = F.leaky_relu(F.linear(torch.stack(mp_enhanced_embs,1), vars_dict['agg_w'], vars_dict['agg_b']))  # (#sample, #mp, 64)
        # return agg_mp_emb.mean(1)

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars

