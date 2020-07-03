# coding: utf-8
# author: lu yf
# create date: 2019-12-02 11:25

import numpy as np
import torch
from torch.nn import functional as F
from Evaluation import Evaluation
from MetaLearner import MetapathLearner, MetaLearner, AggLearner


class HML(torch.nn.Module):
    def __init__(self, config):
        super(HML, self).__init__()
        self.config = config
        self.use_cuda = self.config['use_cuda']
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")

        if self.config['dataset'] == 'movielens':
            from EmbeddingInitializer import UserEmbeddingML, ItemEmbeddingML
            self.item_emb = ItemEmbeddingML(config)
            self.user_emb = UserEmbeddingML(config)
        elif self.config['dataset'] == 'yelp':
            from EmbeddingInitializer import UserEmbeddingYelp, ItemEmbeddingYelp
            self.item_emb = ItemEmbeddingYelp(config)
            self.user_emb = UserEmbeddingYelp(config)
        elif self.config['dataset'] == 'dbook':
            from EmbeddingInitializer import UserEmbeddingDB, ItemEmbeddingDB
            self.item_emb = ItemEmbeddingDB(config)
            self.user_emb = UserEmbeddingDB(config)

        self.mp_learner = MetapathLearner(config,self.user_emb,self.item_emb)
        self.agg_learner = AggLearner(config,self.user_emb,self.item_emb)
        self.meta_learner = MetaLearner(config,self.user_emb,self.item_emb)

        self.mp_lr = config['mp_lr']
        self.local_lr = config['local_lr']
        self.emb_dim = self.config['embedding_dim']

        self.cal_metrics = Evaluation()

        self.ml_weight_len = len(self.meta_learner.update_parameters())
        self.ml_weight_name = list(self.meta_learner.update_parameters().keys())
        self.agg_weight_len = len(self.agg_learner.update_parameters())
        self.agg_weight_name = list(self.agg_learner.update_parameters().keys())
        self.mp_weight_len = len(self.mp_learner.update_parameters())
        self.mp_weight_name = list(self.mp_learner.update_parameters().keys())

        self.transformer_liners = self.transform_mp2task()

        self.meta_optimizer = torch.optim.Adam(self.parameters(),lr=config['lr'])

    def transform_mp2task(self):
        liners = {}
        ml_parameters = self.meta_learner.update_parameters()
        # output_dim_of_mp = self.config['user_embedding_dim']
        output_dim_of_mp = 32  # movielens: lr=0.001, avg mp, 0.8081
        for w in self.ml_weight_name:
            liners[w.replace('.','-')] = torch.nn.Linear(output_dim_of_mp,
                                                         np.prod(ml_parameters[w].shape))
        return torch.nn.ModuleDict(liners)

    def forward(self, support_set_x, support_set_y, support_mp_user_emb, vars_dict=None):
        """
        """
        if vars_dict is None:
            vars_dict = self.meta_learner.update_parameters()

        support_set_y_pred = self.meta_learner(support_set_x, support_mp_user_emb, vars_dict)
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, vars_dict.values(),create_graph=True)

        fast_weights = {}
        for i, w in enumerate(vars_dict.keys()):
            fast_weights[w] = vars_dict[w] - self.local_lr * grad[i]

        for idx in range(1, self.config['local_update']):  # for the current task, locally update
            support_set_y_pred = self.meta_learner(support_set_x, support_mp_user_emb, vars_dict=fast_weights)
            loss = F.mse_loss(support_set_y_pred, support_set_y)  # calculate loss on support set
            grad = torch.autograd.grad(loss, fast_weights.values(),create_graph=True)  # calculate gradients w.r.t. model parameters

            for i, w in enumerate(fast_weights.keys()):
                fast_weights[w] = fast_weights[w] - self.local_lr * grad[i]

        return fast_weights

    def mp_update(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        Mete-update the parameters of MetaPathLearner, AggLearner and MetaLearner.
        """
        # each mp
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        mp_task_fast_weights_s = {}
        mp_task_loss_s = {}

        mp_initial_weights = self.mp_learner.update_parameters()
        ml_initial_weights = self.meta_learner.update_parameters()

        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp)
            support_set_y_pred = self.meta_learner(support_set_x, support_mp_enhanced_user_emb)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, mp_initial_weights.values(),create_graph=True)

            fast_weights = {}
            for i in range(self.mp_weight_len):
                weight_name = self.mp_weight_name[i]
                fast_weights[weight_name] = mp_initial_weights[weight_name] - self.mp_lr * grad[i]

            for idx in range(1, self.config['mp_update']):
                support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp, vars_dict=fast_weights)
                support_set_y_pred = self.meta_learner(support_set_x, support_mp_enhanced_user_emb)
                loss = F.mse_loss(support_set_y_pred, support_set_y)
                grad = torch.autograd.grad(loss, fast_weights.values(),create_graph=True)

                for i in range(self.mp_weight_len):
                    weight_name = self.mp_weight_name[i]
                    fast_weights[weight_name] = fast_weights[weight_name] - self.mp_lr * grad[i]

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp, vars_dict=fast_weights)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_learner(query_set_x, query_set_mp, mp, vars_dict=fast_weights)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

            f_fast_weights = {}
            for w, liner in self.transformer_liners.items():
                w = w.replace('-','.')
                f_fast_weights[w] = ml_initial_weights[w] * \
                                    torch.sigmoid(liner(support_mp_enhanced_user_emb.mean(0))).\
                                        view(ml_initial_weights[w].shape)

            # # the current mp ---> task update
            mp_task_fast_weights = self.forward(support_set_x, support_set_y, support_mp_enhanced_user_emb, vars_dict=f_fast_weights)
            mp_task_fast_weights_s[mp] = mp_task_fast_weights

            query_set_y_pred = self.meta_learner(query_set_x, query_mp_enhanced_user_emb, vars_dict=mp_task_fast_weights)
            q_loss = F.mse_loss(query_set_y_pred, query_set_y)
            mp_task_loss_s[mp] = q_loss.data  # movielens: 0.8126 dbook 0.6084
            # mp_task_loss_s[mp] = loss.data  # dbook 0.6144
            # mp_task_loss_s[mp] = sim

        # mp_att = torch.FloatTensor([l/sum(mp_task_loss_s.values()) for l in mp_task_loss_s.values()]).to(self.device)  # movielens: 0.81
        mp_att = F.softmax(-torch.stack(list(mp_task_loss_s.values())),dim=0)  # movielens: 0.80781 lr0.001
        # mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)
        # mp_att = F.softmax(torch.stack(list(mp_task_loss_s.values())),dim=0)

        agg_task_fast_weights = self.aggregator(mp_task_fast_weights_s, mp_att)
        support_agg_enhanced_user_emb = self.agg_learner(support_set_x, support_mp_enhanced_user_emb_s, mp_att)
        agg_task_fast_weights = self.forward(support_set_x, support_set_y, support_agg_enhanced_user_emb,vars_dict=agg_task_fast_weights)
        query_agg_enhanced_user_emb = self.agg_learner(query_set_x, query_mp_enhanced_user_emb_s, mp_att)
        query_y_pred = self.meta_learner(query_set_x, query_agg_enhanced_user_emb, vars_dict=agg_task_fast_weights)
        loss = F.mse_loss(query_y_pred, query_set_y)
        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real,query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real,query_y_pred, k=5)
        return loss, mae, rmse, ndcg_5

    def mp_update2(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        Mete-update the parameters of MetaPathLearner, AggLearner and MetaLearner.
        add agg update w.r.t mp_update, a litter worse 10epoch:0.6292
        """
        # each mp
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        mp_task_fast_weights_s = {}
        mp_task_loss_s = {}

        mp_initial_weights = self.mp_learner.update_parameters()
        ml_initial_weights = self.meta_learner.update_parameters()
        agg_initial_weights = self.agg_learner.update_parameters()

        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp)  # (#samples, user_emb_dim)
            support_set_y_pred = self.meta_learner(support_set_x, support_mp_enhanced_user_emb)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, mp_initial_weights.values(), retain_graph=True)  # grad. w.r.t. mp
            mp_fast_weights = {}
            for i in range(self.mp_weight_len):  # update parameters of mp_learner
                weight_name = self.mp_weight_name[i]
                mp_fast_weights[weight_name] = mp_initial_weights[weight_name] - self.mp_lr * grad[i]

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp, vars_dict=mp_fast_weights)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_learner(query_set_x, query_set_mp, mp, vars_dict=mp_fast_weights)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

            # adapt global parameters to mp-level parameters:  f(.)
            f_fast_weights = {}
            for w_name, w_value in ml_initial_weights.items():
                f_fast_weights[w_name] = w_value * \
                                         torch.sigmoid(self.linears[w_name]
                                                       (support_mp_enhanced_user_emb.mean(0))).view(w_value.shape)

            # # the current mp ---> task update
            mp_task_fast_weights = self.forward(support_set_x, support_set_y, support_mp_enhanced_user_emb,
                                                vars_dict=f_fast_weights)
            mp_task_fast_weights_s[mp] = mp_task_fast_weights

            # # test on query data, under meta-path mp
            # query_set_y_pred = self.meta_learner(query_set_x, query_mp_enhanced_user_emb, vars_dict=ml_fast_weights)
            # q_loss = F.mse_loss(query_set_y_pred, query_set_y)
            # mp_task_loss_s[mp] = q_loss

        mp_att = dict(zip(self.config['mp'], [1.0 / len(self.config['mp'])] * len(self.config['mp'])))
        # fuse updated parameters of meta_learner
        agg_task_fast_weights = self.aggregator(mp_task_fast_weights_s, mp_att)

        support_agg_enhanced_user_emb = self.agg_learner(support_set_x, support_mp_enhanced_user_emb_s)
        support_set_y_pred = self.meta_learner(support_set_x, support_agg_enhanced_user_emb)
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, agg_initial_weights.values())
        agg_fast_weights = {}
        for i in range(self.agg_weight_len):
            weight_name = self.agg_weight_name[i]
            agg_fast_weights[weight_name] = agg_initial_weights[weight_name] - self.local_lr * grad[i]

        query_agg_enhanced_user_emb = self.agg_learner(query_set_x, query_mp_enhanced_user_emb_s, vars_dict=agg_fast_weights)
        query_y_pred = self.meta_learner(query_set_x, query_agg_enhanced_user_emb, vars_dict=agg_task_fast_weights)
        loss = F.mse_loss(query_y_pred, query_set_y)

        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real,query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real,query_y_pred, k=5)
        return loss, mae, rmse, ndcg_5

    def mp_update3(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        Mete-update the parameters of MetaPathLearner, AggLearner and MetaLearner.
        dbook: put together, loss up!!! 12.51.. --> agg update is wrong, cause mp_MAML is right, but why mp_update2 is right? (add a local ml update)

        """
        # each mp
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []

        mp_initial_weights = self.mp_learner.update_parameters()
        ml_initial_weights = self.meta_learner.update_parameters()
        agg_initial_weights = self.agg_learner.update_parameters()

        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp)
            support_set_y_pred = self.meta_learner(support_set_x, support_mp_enhanced_user_emb)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, mp_initial_weights.values())

            mp_fast_weights = {}
            for i in range(self.mp_weight_len):
                weight_name = self.mp_weight_name[i]
                mp_fast_weights[weight_name] = mp_initial_weights[weight_name] - self.mp_lr * grad[i]

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp, vars_dict=mp_fast_weights)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_learner(query_set_x, query_set_mp, mp, vars_dict=mp_fast_weights)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

        # get agg embedding for the task on support set
        support_agg_enhanced_user_emb = self.agg_learner(support_set_x, support_mp_enhanced_user_emb_s)
        # initialize meta-learner parameters into mp-specific
        f_fast_weights = {}
        for w_name, w_value in ml_initial_weights.items():
            f_fast_weights[w_name] = w_value * \
                                     torch.sigmoid(self.linears[w_name]
                                                   (support_agg_enhanced_user_emb.mean(0))).view(w_value.shape)
        support_set_y_pred = self.meta_learner(support_set_x, support_agg_enhanced_user_emb, vars_dict=f_fast_weights)
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, agg_initial_weights.values(), retain_graph=True)  # grad. w.r.t. agg parameters
        agg_fast_weights = {}
        for i in range(self.agg_weight_len):
            weight_name = self.agg_weight_name[i]
            agg_fast_weights[weight_name] = agg_initial_weights[weight_name] - self.local_lr * grad[i]
        # after meta-update, test on query data, i.e., get agg embedding for task on query data
        query_agg_enhanced_user_emb = self.agg_learner(query_set_x, query_mp_enhanced_user_emb_s,
                                                       vars_dict=agg_fast_weights)

        # loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss,f_fast_weights.values())  # grad. w.r.t. meta-learner parameters, now is f_fast_weight
        ml_fast_weights = {}
        for i in range(self.ml_weight_len):
            weight_name = self.ml_weight_name[i]
            ml_fast_weights[weight_name] = f_fast_weights[weight_name] - self.local_lr * grad[i]
        # test on query data
        query_y_pred = self.meta_learner(query_set_x, query_agg_enhanced_user_emb, vars_dict=ml_fast_weights)
        loss = F.mse_loss(query_y_pred, query_set_y)

        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real,query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real,query_y_pred, k=5)
        return loss, mae, rmse, ndcg_5

    def mp_update_no_f(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        Mete-update the parameters of MetaPathLearner, AggLearner and MetaLearner.
        """
        # each mp
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        mp_task_fast_weights_s = {}
        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])

            fast_weights = {}
            weights_for_update = self.mp_learner.update_parameters()
            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp)
            support_set_y_pred = self.meta_learner(support_set_x, support_mp_enhanced_user_emb)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, weights_for_update.values())
            for i in range(self.mp_weight_len):
                weight_name = self.mp_weight_name[i]
                fast_weights[weight_name] = weights_for_update[weight_name] - self.mp_lr * grad[i]

            for idx in range(1, self.config['mp_update']):
                support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp, vars_dict=fast_weights)
                support_set_y_pred = self.meta_learner(support_set_x, support_mp_enhanced_user_emb)
                loss = F.mse_loss(support_set_y_pred, support_set_y)
                grad = torch.autograd.grad(loss, fast_weights.values())
                for i in range(self.mp_weight_len):
                    weight_name = self.mp_weight_name[i]
                    fast_weights[weight_name] = fast_weights[weight_name] - self.mp_lr * grad[i]

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp, vars_dict=fast_weights)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_learner(query_set_x, query_set_mp, mp, vars_dict=fast_weights)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

            # # the current mp ---> task update, but no f function!!!
            mp_task_fast_weights = self.forward(support_set_x, support_set_y, support_mp_enhanced_user_emb)  # no f !!
            mp_task_fast_weights_s[mp] = mp_task_fast_weights

        mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)  # mean

        query_agg_enhanced_user_emb = self.agg_learner(query_set_x, query_mp_enhanced_user_emb_s, mp_att)
        agg_task_fast_weights = self.aggregator(mp_task_fast_weights_s, query_mp_enhanced_user_emb_s, mp_att)
        query_y_pred = self.meta_learner(query_set_x, query_agg_enhanced_user_emb, vars_dict=agg_task_fast_weights)

        loss = F.mse_loss(query_y_pred, query_set_y)
        mae, rmse = self.cal_metrics.prediction(query_set_y.data.cpu().numpy(),
                                                query_y_pred.data.cpu().numpy())
        ndcg_5 = self.cal_metrics.ranking(query_set_y.data.cpu().numpy(),
                                          query_y_pred.data.cpu().numpy(), 5)

        return loss, mae, rmse, ndcg_5

    def mp_update_mp_MAML(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        MeLU + multiple meta-paths aggregation
        """
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp)
            query_mp_enhanced_user_emb = self.mp_learner(query_set_x, query_set_mp, mp)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

        mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)  # mean
        support_agg_enhanced_user_emb = self.agg_learner(support_set_x, support_mp_enhanced_user_emb_s,mp_att)
        query_agg_enhanced_user_emb = self.agg_learner(query_set_x, query_mp_enhanced_user_emb_s,mp_att)

        task_fast_weights = self.forward(support_set_x, support_set_y, support_agg_enhanced_user_emb)
        query_y_pred = self.meta_learner(query_set_x,query_agg_enhanced_user_emb,vars_dict=task_fast_weights)
        loss = F.mse_loss(query_y_pred, query_set_y)
        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real, query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=5)

        return loss, mae, rmse, ndcg_5

    def mp_update_multi_MAML(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        multiple MAML for multiple meta-paths
        """
        loss_s = []
        mae_s, rmse_s = [], []
        ndcg_at_5 = []

        for mp in self.config['mp']:
            support_set_mp = support_set_mps[mp]
            query_set_mp = query_set_mps[mp]

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp)
            query_mp_enhanced_user_emb = self.mp_learner(query_set_x, query_set_mp, mp)

            task_fast_weights = self.forward(support_set_x, support_set_y, support_mp_enhanced_user_emb)
            query_y_pred = self.meta_learner(query_set_x, query_mp_enhanced_user_emb, vars_dict=task_fast_weights)
            loss = F.mse_loss(query_y_pred, query_set_y)
            mae, rmse = self.cal_metrics.prediction(query_set_y.data.cpu().numpy(),
                                                    query_y_pred.data.cpu().numpy())
            ndcg_5 = self.cal_metrics.ranking(query_set_y.data.cpu().numpy(),
                                              query_y_pred.data.cpu().numpy(), 5)

            loss_s.append(loss)
            mae_s.append(mae)
            rmse_s.append(rmse)
            ndcg_at_5.append(ndcg_5)

        return torch.stack(loss_s).mean(0), np.mean(mae_s), np.mean(rmse_s), np.mean(ndcg_5)

    def no_MAML(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        # each mp
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = {}, {}
        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])

            support_mp_enhanced_user_emb = self.mp_learner(support_set_x, support_set_mp, mp)
            support_mp_enhanced_user_emb_s[mp] = support_mp_enhanced_user_emb

            query_mp_enhanced_user_emb = self.mp_learner(query_set_x, query_set_mp, mp)
            query_mp_enhanced_user_emb_s[mp] = query_mp_enhanced_user_emb

        support_agg_enhanced_user_emb, _ = self.agg_learner(support_set_x, support_mp_enhanced_user_emb_s)
        support_y_pred = self.meta_learner(support_set_x, support_agg_enhanced_user_emb)
        support_loss = F.mse_loss(support_y_pred, support_set_y.view(-1, 1))
        support_mae, support_rmse = self.cal_metrics.prediction(support_set_y.data.cpu().numpy(),
                                                                support_y_pred.squeeze(1).data.cpu().numpy())
        support_ndcg_5 = self.cal_metrics.ranking(support_set_y.data.cpu().numpy(),
                                                  support_y_pred.squeeze(1).data.cpu().numpy(), k_list=[1, 3, 5])

        query_agg_enhanced_user_emb, _ = self.agg_learner(query_set_x, query_mp_enhanced_user_emb_s)
        query_y_pred = self.meta_learner(query_set_x, query_agg_enhanced_user_emb)
        query_loss = F.mse_loss(query_y_pred, query_set_y.view(-1, 1))
        query_mae, query_rmse = self.cal_metrics.prediction(query_set_y.data.cpu().numpy(),
                                                            query_y_pred.squeeze(1).data.cpu().numpy())
        query_ndcg_5 = self.cal_metrics.ranking(query_set_y.data.cpu().numpy(),
                                                query_y_pred.squeeze(1).data.cpu().numpy(), k_list=[1, 3, 5])

        return (support_loss + query_loss) / 2.0, (support_mae + query_mae) / 2.0, (support_rmse + query_rmse) / 2.0, \
               (support_ndcg_5 + query_ndcg_5) / 2.0

    def global_update(self,support_xs,support_ys,support_mps,query_xs,query_ys,query_mps,device='cpu'):
        """
        """
        batch_sz = len(support_xs)
        loss_s = []
        mae_s = []
        rmse_s = []
        ndcg_at_5_s = []

        for i in range(batch_sz):   # each task in a batch
            support_mp = dict(support_mps[i])  # must be dict!!!
            query_mp = dict(query_mps[i])

            for mp in self.config['mp']:
                support_mp[mp] = map(lambda x: x.to(device), support_mp[mp])
                query_mp[mp] = map(lambda x: x.to(device), query_mp[mp])
            _loss, _mae, _rmse, _ndcg_5 = self.mp_update(support_xs[i].to(device), support_ys[i].to(device), support_mp,
                                                         query_xs[i].to(device), query_ys[i].to(device), query_mp)
            # _loss, _mae, _rmse, _ndcg_5 = self.mp_update_mp_MAML(support_xs[i].to(device), support_ys[i].to(device), support_mp,
            #                                                      query_xs[i].to(device), query_ys[i].to(device), query_mp)
            # _loss, _mae, _rmse, _ndcg_5 = self.mp_update_multi_MAML(support_xs[i].to(device), support_ys[i].to(device), support_mp,
            #                                                         query_xs[i].to(device), query_ys[i].to(device), query_mp)
            # _loss, _mae, _rmse, _ndcg_5 = self.mp_update_no_f(support_xs[i].to(device), support_ys[i].to(device), support_mp,
            #                                                   query_xs[i].to(device), query_ys[i].to(device), query_mp)
            # _loss, _mae, _rmse, _ndcg_5 = self.no_MAML(support_xs[i].to(device), support_ys[i].to(device), support_mp,
            #                                            query_xs[i].to(device), query_ys[i].to(device), query_mp)
            loss_s.append(_loss)
            mae_s.append(_mae)
            rmse_s.append(_rmse)
            ndcg_at_5_s.append(_ndcg_5)

        loss = torch.stack(loss_s).mean(0)
        mae = np.mean(mae_s)
        rmse = np.mean(rmse_s)
        ndcg_at_5 = np.mean(ndcg_at_5_s)

        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

        return loss.cpu().data.numpy(), mae, rmse, ndcg_at_5

    def evaluation(self,support_x,support_y,support_mp,query_x,query_y,query_mp,device='cpu'):
        """
        """
        support_mp = dict(support_mp)  # must be dict!!!
        query_mp = dict(query_mp)
        for mp in self.config['mp']:
            support_mp[mp] = map(lambda x: x.to(device), support_mp[mp])
            query_mp[mp] = map(lambda x: x.to(device), query_mp[mp])

        _, mae, rmse, ndcg_5 = self.mp_update(support_x.to(device), support_y.to(device), support_mp,
                                              query_x.to(device), query_y.to(device), query_mp)
        # _, mae, rmse, ndcg_5 = self.mp_update_mp_MAML(support_x.to(device), support_y.to(device), support_mp,
        #                                               query_x.to(device), query_y.to(device), query_mp)
        # _, mae, rmse, ndcg_5 = self.mp_update_multi_MAML(support_x.to(device), support_y.to(device), support_mp,
        #                                                  query_x.to(device), query_y.to(device), query_mp)
        # _, mae, rmse, ndcg_5 = self.mp_update_no_f(support_x.to(device), support_y.to(device), support_mp,
        #                                            query_x.to(device), query_y.to(device), query_mp)
        # mae, rmse, ndcg_5 = self.eval_no_MAML(query_x.to(device), query_y.to(device), query_mp)

        return mae, rmse, ndcg_5

    def fine_tune(self, support_x,support_y,support_mp):
        if self.cuda():
            support_x = support_x.cuda()
            support_y = support_y.cuda()
            support_mp = dict(support_mp)  # must be dict!!!

            for mp, mp_data in support_mp.items():
                support_mp[mp] = list(map(lambda x: x.cuda(), mp_data))
        support_mp_enhanced_user_emb_s = {}
        for mp in self.config['mp']:
            support_set_mp = support_mp[mp]
            support_mp_enhanced_user_emb = self.mp_learner(support_x, support_set_mp, mp)
            support_mp_enhanced_user_emb_s[mp] = support_mp_enhanced_user_emb

        support_agg_enhanced_user_emb, _ = self.agg_learner(support_x, support_mp_enhanced_user_emb_s)
        support_y_pred = self.meta_learner(support_x, support_agg_enhanced_user_emb)
        support_loss = F.mse_loss(support_y_pred, support_y.view(-1, 1))

        # fine-tune
        self.meta_optimizer.zero_grad()
        support_loss.backward()
        self.meta_optimizer.step()

    def eval_no_MAML(self, query_set_x, query_set_y, query_set_mps):
        # each mp
        query_mp_enhanced_user_emb_s = {}
        for mp in self.config['mp']:
            query_set_mp = list(query_set_mps[mp])
            query_mp_enhanced_user_emb = self.mp_learner(query_set_x, query_set_mp, mp)
            query_mp_enhanced_user_emb_s[mp] = query_mp_enhanced_user_emb

        query_agg_enhanced_user_emb, _ = self.agg_learner(query_set_x, query_mp_enhanced_user_emb_s)
        query_y_pred = self.meta_learner(query_set_x, query_agg_enhanced_user_emb)
        query_mae, query_rmse = self.cal_metrics.prediction(query_set_y.data.cpu().numpy(),
                                                            query_y_pred.squeeze(1).data.cpu().numpy())
        query_ndcg_s = self.cal_metrics.ranking(query_set_y.data.cpu().numpy(),
                                                query_y_pred.squeeze(1).data.cpu().numpy(), k_list=[1, 3, 5])

        return query_mae, query_rmse, {1: query_ndcg_s[1], 3: query_ndcg_s[3], 5: query_ndcg_s[5]}

    def aggregator(self, task_weights_s, att):
        for idx, mp in enumerate(self.config['mp']):
            if idx == 0:
                att_task_weights = dict({k:v*att[idx] for k,v in task_weights_s[mp].items()})
                continue
            tmp_att_task_weights = dict({k:v*att[idx] for k,v in task_weights_s[mp].items()})
            att_task_weights = dict(zip(att_task_weights.keys(),
                                               list(map(lambda x: x[0]+x[1],
                                                        zip(att_task_weights.values(),tmp_att_task_weights.values())))))

        return att_task_weights