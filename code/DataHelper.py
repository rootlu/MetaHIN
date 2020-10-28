# coding: utf-8
# author: lu yf
# create date: 2019-11-24 13:16
import gc
import glob
import os
import pickle
# from DataProcessor import Movielens
from tqdm import tqdm
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import torch


class DataHelper:
    def __init__(self, input_dir, output_dir, config):
        self.input_dir = input_dir  # ../data/movielens_1m/original/
        self.output_dir = output_dir  # ../data/movielens_1m
        self.config = config
        self.mp_list = self.config['mp']

    def load_data(self, data_set, state, load_from_file=True):
        data_dir = os.path.join(self.output_dir,data_set)
        supp_xs_s = []
        supp_ys_s = []
        supp_mps_s = []
        query_xs_s = []
        query_ys_s = []
        query_mps_s = []

        if data_set == 'yelp':
            training_set_size = int(
                len(glob.glob("{}/{}/*.npy".format(data_dir, state))) / self.config['file_num'])  # support, query

            # load all data
            for idx in tqdm(range(training_set_size)):
                supp_xs_s.append(torch.from_numpy(np.load("{}/{}/support_x_{}.npy".format(data_dir, state, idx))))
                supp_ys_s.append(torch.from_numpy(np.load("{}/{}/support_y_{}.npy".format(data_dir, state, idx))))
                query_xs_s.append(torch.from_numpy(np.load("{}/{}/query_x_{}.npy".format(data_dir, state, idx))))
                query_ys_s.append(torch.from_numpy(np.load("{}/{}/query_y_{}.npy".format(data_dir, state, idx))))

                supp_mp_data, query_mp_data = {}, {}
                for mp in self.mp_list:
                    _cur_data = np.load("{}/{}/support_{}_{}.npy".format(data_dir, state, mp, idx), encoding='latin1')
                    supp_mp_data[mp] = [torch.from_numpy(x) for x in _cur_data]
                    _cur_data = np.load("{}/{}/query_{}_{}.npy".format(data_dir, state, mp, idx), encoding='latin1')
                    query_mp_data[mp] = [torch.from_numpy(x) for x in _cur_data]
                supp_mps_s.append(supp_mp_data)
                query_mps_s.append(query_mp_data)
        else:
#             if not load_from_file:
#                 ml = Movielens(os.path.join(self.input_dir,data_set), os.path.join(self.output_dir,data_set))
#                 ml.support_query_data()

            training_set_size = int(len(glob.glob("{}/{}/*.pkl".format(data_dir,state))) / self.config['file_num'])  # support, query

            # load all data
            for idx in tqdm(range(training_set_size)):
                support_x = pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb"))
                if support_x.shape[0] > 5:
                    continue
                del support_x
                supp_xs_s.append(pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb")))
                supp_ys_s.append(pickle.load(open("{}/{}/support_y_{}.pkl".format(data_dir, state, idx), "rb")))
                query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(data_dir, state, idx), "rb")))
                query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(data_dir, state, idx), "rb")))

                supp_mp_data, query_mp_data = {}, {}
                for mp in self.mp_list:
                    supp_mp_data[mp] = pickle.load(open("{}/{}/support_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))
                    query_mp_data[mp] = pickle.load(open("{}/{}/query_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))
                supp_mps_s.append(supp_mp_data)
                query_mps_s.append(query_mp_data)

        print('#support set: {}, #query set: {}'.format(len(supp_xs_s), len(query_xs_s)))
        total_data = list(zip(supp_xs_s, supp_ys_s, supp_mps_s,
                              query_xs_s, query_ys_s, query_mps_s))  # all training tasks
        del (supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s)
        gc.collect()
        return total_data

    def load_batch_data(self, data_set, state, batch_indices, load_from_file=True):
        data_dir = os.path.join(self.output_dir,data_set)

        supp_xs_s = []
        supp_ys_s = []
        supp_mps_s = []
        query_xs_s = []
        query_ys_s = []
        query_mps_s = []

        for idx in batch_indices:
            supp_xs_s.append(pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/{}/support_y_{}.pkl".format(data_dir, state, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(data_dir, state, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(data_dir, state, idx), "rb")))

            supp_mp_data, query_mp_data = {}, {}
            for mp in self.mp_list:
                supp_mp_data[mp] = pickle.load(open("{}/{}/support_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))
                query_mp_data[mp] = pickle.load(open("{}/{}/query_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))

            supp_mps_s.append(supp_mp_data)
            query_mps_s.append(query_mp_data)

        return supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s

    def load_data_multiprocess(self, data_set, state, batch_indices, load_from_file=True):
        data_dir = os.path.join(self.output_dir, data_set)
        global cur_state
        cur_state = state

        supp_xs_s = []
        supp_ys_s = []
        supp_mps_s = []
        query_xs_s = []
        query_ys_s = []
        query_mps_s = []

        pool = ThreadPool(processes=20)
        res = pool.map(self.load_single_data, batch_indices)
        for r in res:
            supp_xs_s.append(r[0])
            supp_ys_s.append(r[1])
            supp_mps_s.append(r[2])
            query_xs_s.append(r[3])
            query_ys_s.append(r[4])
            query_mps_s.append(r[5])
        return supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s

    def load_single_data(self, idx):
        data_dir = os.path.join(self.output_dir, self.config['dataset'])

        supp_xs = pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, cur_state, idx), "rb"))
        supp_ys = pickle.load(open("{}/{}/support_y_{}.pkl".format(data_dir, cur_state, idx), "rb"))
        query_xs = pickle.load(open("{}/{}/query_x_{}.pkl".format(data_dir, cur_state, idx), "rb"))
        query_ys = pickle.load(open("{}/{}/query_y_{}.pkl".format(data_dir, cur_state, idx), "rb"))
        supp_mp_data = {}
        query_mp_data = {}
        for mp in self.config['mp']:
            supp_mp_data[mp] = pickle.load(open("{}/{}/support_{}_{}.pkl".format(data_dir, cur_state, mp, idx), "rb"))
            query_mp_data[mp] = pickle.load(open("{}/{}/query_{}_{}.pkl".format(data_dir, cur_state, mp, idx), "rb"))

        return supp_xs, supp_ys, supp_mp_data, query_xs, query_ys, query_mp_data


# if __name__ == "__main__":
    # from Config import config_ml
    # data_set = 'movielens_1m'
    # input_dir = '../data/'
    # output_dir = '../data/'
    #
    # data_helper = DataHelper(input_dir, output_dir, config_ml)
    #
    # training_set_size = int(len(glob.glob("../data/{}/{}/*.pkl".format(data_set, 'meta_training'))) / config_ml['file_num'])
    # indices = list(range(training_set_size))
    # random.shuffle(indices)
    # num_batch = int(training_set_size / 32)
    # start_time = time.time()
    # for idx, i in tqdm(enumerate(range(num_batch))):
    #     cur_indices = indices[32*i:32*(i+1)]
    #     support_xs, support_ys, support_mps, query_xs, query_ys, query_mps = \
    #         data_helper.load_data_multiprocess(data_set, 'meta_training', cur_indices)
    #
    # print(time.time()-start_time)
