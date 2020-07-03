# coding: utf-8
# author: lu yf
# create date: 2019-11-19 20:48
import collections
import multiprocessing
from collections import defaultdict
import datetime
import os
import random
import re

import json
import pandas as pd
import pickle
import numpy as np
import torch
from Config import states
from tqdm import tqdm

random.seed(13)


class Movielens:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir  # ../data/movielens_1m/original/
        self.output_dir = output_dir  # ../data/movielens_1m
        self.movie_directors, self.movie_actors = dict(), dict()
        self.user_data, self.item_data, self.score_data = self.load()
        self.hashmap_data()
        self.metapath_data()

    def load(self):
        print('loading data...')
        path = self.input_dir
        profile_data_path = "{}/original/users.dat".format(path)
        score_data_path = "{}/original/ratings.dat".format(path)
        item_data_path = "{}/original/movies_extrainfos.dat".format(path)

        profile_data = pd.read_csv(
            profile_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
            sep="::", engine='python'
        )
        item_data = pd.read_csv(
            item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
            sep="::", engine='python', encoding="utf-8"
        )
        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        score_data = score_data.drop(["timestamp"], axis=1)
        return profile_data, item_data, score_data

    def load_list(self, fname):
        list_ = []
        with open(fname, encoding="utf-8") as f:
            for line in f.readlines():
                list_.append(line.strip())
        return list_

    def reverse_dict(self, d):
        # {1:[a,b,c], 2:[a,f,g],...}
        re_d = collections.defaultdict(list)
        for k, v_list in d.items():
            for v in v_list:
                re_d[v].append(k)
        return re_d

    def item_converting(self, row, rate_list, genre_list, director_list, actor_list):
        """

        :param row:
        :param rate_list:
        :param genre_list:
        :param director_list:
        :param actor_list:
        :return:
        """
        rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
        genre_idx = torch.zeros(1, 25).long()
        for genre in str(row['genre']).split(", "):
            idx = genre_list.index(genre)
            genre_idx[0, idx] = 1  # one-hot vector
        director_id = []
        for director in str(row['director']).split(", "):
            idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
            director_id.append(str(idx+1))  # id starts from 1, not index
        actor_id = []
        for actor in str(row['actors']).split(", "):
            idx = actor_list.index(actor)
            actor_id.append(str(idx+1))
        return torch.cat((rate_idx, genre_idx), 1), director_id, actor_id

    def user_converting(self, row, gender_list, age_list, occupation_list, zipcode_list):
        """

        :param row:
        :param gender_list:
        :param age_list:
        :param occupation_list:
        :param zipcode_list:
        :return:
        """
        gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
        age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
        occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
        zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
        return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)  # (1, 4)

    def hashmap_data(self):
        """

        :return:
        """
        print('hashing data...')
        rate_list = self.load_list("{}/original/m_rate.txt".format(self.input_dir))
        genre_list = self.load_list("{}/original/m_genre.txt".format(self.input_dir))
        actor_list = self.load_list("{}/original/m_actor.txt".format(self.input_dir))
        director_list = self.load_list("{}/original/m_director.txt".format(self.input_dir))
        gender_list = self.load_list("{}/original/m_gender.txt".format(self.input_dir))
        age_list = self.load_list("{}/original/m_age.txt".format(self.input_dir))
        occupation_list = self.load_list("{}/original/m_occupation.txt".format(self.input_dir))
        zipcode_list = self.load_list("{}/original/m_zipcode.txt".format(self.input_dir))

        if not os.path.exists("{}/meta_training/".format(self.output_dir)):
            for state in states:
                os.mkdir("{}/{}/".format(self.output_dir, state))

        # hash map for item
        movie_dict = {}
        for idx, row in self.item_data.iterrows():
            m_info = self.item_converting(row, rate_list, genre_list, director_list, actor_list)
            movie_dict[str(row['movie_id'])] = m_info[0]
            self.movie_directors[str(row['movie_id'])] = m_info[1]
            self.movie_actors[str(row['movie_id'])] = m_info[2]
        pickle.dump(movie_dict, open("{}/movie_feature_dict.pkl".format(self.output_dir), "wb"))

        # hash map for user
        user_dict = {}
        for idx, row in self.user_data.iterrows():
            u_info = self.user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            user_dict[str(row['user_id'])] = u_info
        pickle.dump(user_dict, open("{}/user_feature_dict.pkl".format(self.output_dir), "wb"))

        # hash map for actor
        json.dump(self.movie_actors, open("{}/movie_actor_dict.json".format(self.output_dir), 'w'))
        actor_dict = {}
        for m, actors in self.movie_actors.items():
            for a in actors:
                one_hot = torch.zeros(1, 8030).long()
                one_hot[0, int(a) - 1] = 1
                actor_dict[a] = one_hot
        pickle.dump(actor_dict, open("{}/actor_feature_dict.pkl".format(self.output_dir), "wb"))

        # hash map for director
        json.dump(self.movie_directors, open("{}/movie_director_dict.json".format(self.output_dir), 'w'))
        director_dict = {}
        for m, directors in self.movie_directors.items():
            for d in directors:
                one_hot = torch.zeros(1, 2186).long()
                one_hot[0, int(d) - 1] = 1
                director_dict[d] = one_hot
        pickle.dump(director_dict, open("{}/director_feature_dict.pkl".format(self.output_dir), "wb"))

    def metapath_data(self):
        """

        :return:
        """
        if os.path.exists("{}/{}/u_m_a_users.json".format(self.output_dir,'meta_training')):
            print('data based on meta-paths existing!')
        else:
            print('processing data based on meta-paths...')
            # reverse dict MD to DM, MA to AM, UM to MU
            director_movies = self.reverse_dict(self.movie_directors)
            actor_movies = self.reverse_dict(self.movie_actors)
            for state in states:
                print('current state: {}..'.format(state))
                with open("{}/{}.json".format(self.input_dir, state), encoding="utf-8") as f:
                    user_movies = json.loads(f.read())
                movie_users = self.reverse_dict(user_movies)
                u_m_directors, u_m_actors, u_m_users, u_m_d_movies, u_m_a_movies, u_m_u_movies \
                    = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
                for user in tqdm(user_movies):
                    for movie in user_movies[user]:
                        u_m_directors[(user,movie)] += list(set(self.movie_directors[movie]))
                        u_m_actors[(user,movie)] += list(set(self.movie_actors[movie]))
                        u_m_users[(user,movie)] += list(set(movie_users[movie]))
                        for director in self.movie_directors[movie]:
                            u_m_d_movies[(user,movie)] += director_movies[director]
                        for actor in self.movie_actors[movie]:
                            u_m_a_movies[(user,movie)] += actor_movies[actor]
                        for u in movie_users[movie]:
                            u_m_u_movies[(user,movie)] += user_movies[u]
                        # u_m_d_movies[(user,movie)] += list(set(reduce(lambda x,y: x+y, map(lambda d: director_movies[d], self.movie_directors[movie]))))
                        # u_m_a_movies[(user,movie)] += list(set(reduce(lambda x,y: x+y, map(lambda a: actor_movies[a], self.movie_actors[movie]))))
                        # u_m_u_movies[(user,movie)] += list(set(reduce(lambda x,y: x+y, map(lambda u: user_movies[u], movie_users[movie]))))

                        u_m_d_movies[(user, movie)] = list(set(u_m_d_movies[(user, movie)]))
                        u_m_a_movies[(user, movie)] = list(set(u_m_a_movies[(user, movie)]))
                        u_m_u_movies[(user, movie)] = list(set(u_m_u_movies[(user, movie)]))

                # pickle.dump(user_movies, open("{}/{}/u_movies.json".format(self.output_dir,state), "wb+"))
                pickle.dump(u_m_directors, open("{}/{}/u_m_directors.json".format(self.output_dir,state), "wb+"))
                pickle.dump(u_m_actors, open("{}/{}/u_m_actors.json".format(self.output_dir,state), "wb+"))
                pickle.dump(u_m_users, open("{}/{}/u_m_users.json".format(self.output_dir,state), "wb+"))
                pickle.dump(u_m_d_movies, open("{}/{}/u_m_d_movies.json".format(self.output_dir,state), "wb+"))
                pickle.dump(u_m_a_movies, open("{}/{}/u_m_a_movies.json".format(self.output_dir,state), "wb+"))
                pickle.dump(u_m_u_movies, open("{}/{}/u_m_u_movies.json".format(self.output_dir,state), "wb+"))

    def support_query_data(self):
        print('generating data...')
        user_feature = pickle.load(open("{}/user_feature_dict.pkl".format(self.input_dir), "rb"))
        movie_feature = pickle.load(open("{}/movie_feature_dict.pkl".format(self.input_dir), "rb"))
        # director_feature_data = pickle.load(open("{}/director_feature_dict.pkl".format(self.input_dir), "rb"))
        # actor_feature_data = pickle.load(open("{}/actor_feature_dict.pkl".format(self.input_dir), "rb"))

        if not os.path.exists("{}/meta_training/".format(self.output_dir)):
            for state in states:
                os.mkdir("{}/{}/".format(self.output_dir, state))
        if not os.path.exists("{}/log/".format(self.output_dir)):
            os.mkdir("{}/log/".format(self.output_dir))
        if not os.path.exists("{}/log/".format(self.output_dir)):
            os.mkdir("{}/log/".format(self.output_dir))

        for state in states:
            print(state+'...')
            idx = 0
            if not os.path.exists("{}/{}/{}".format(self.output_dir, "log", state)):
                os.mkdir("{}/{}/{}".format(self.output_dir, "log", state))
            with open("{}/{}.json".format(self.input_dir, state), encoding="utf-8") as f:
                user_movies = json.loads(f.read())
            with open("{}/{}_y.json".format(self.input_dir, state), encoding="utf-8") as f:
                user_movies_y = json.loads(f.read())
            # u_m_directors = json.load(open("{}/{}/u_m_directors.json".format(self.input_dir,state), "r"))
            # u_m_actors = json.load(open("{}/{}/u_m_actors.json".format(self.input_dir,state), "r"))
            u_m_d_movies = dict(pickle.load(open("{}/{}/u_m_d_movies.json".format(self.input_dir,state), "rb")))
            u_m_a_movies = dict(pickle.load(open("{}/{}/u_m_a_movies.json".format(self.input_dir,state), "rb")))
            u_m_users = dict(pickle.load(open("{}/{}/u_m_users.json".format(self.input_dir,state), "rb")))
            u_m_u_movies = dict(pickle.load(open("{}/{}/u_m_u_movies.json".format(self.input_dir,state), "rb")))

            for _, u_id in tqdm(enumerate(user_movies.keys())):  # each task contains support set and query set
                seen_movie_len = len(user_movies[u_id])
                indices = list(range(seen_movie_len))

                if seen_movie_len < 13 or seen_movie_len > 100:
                    continue

                random.shuffle(indices)
                tmp_movies = np.array(user_movies[u_id])
                tmp_y = np.array(user_movies_y[u_id])

                support_x_app = None
                support_um_app = []
                support_umu_app = []
                support_umum_app = []
                support_umam_app = []
                support_umdm_app = []
                for m_id in tmp_movies[indices[:-10]]:
                    tmp_x_converted = torch.cat((movie_feature[m_id], user_feature[u_id]), 1)
                    try:
                        support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                    except:
                        support_x_app = tmp_x_converted

                    # meta-paths
                    # UM
                    support_um_app.append(torch.cat(list(map(lambda x: movie_feature[x], user_movies[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
                    # UMU
                    support_umu_app.append(torch.cat(list(map(lambda x: user_feature[x], u_m_users[(u_id,m_id)])), dim=0))  # each element: (#neighbor, 4)
                    # UMUM
                    support_umum_app.append(torch.cat(list(map(lambda x: movie_feature[x], u_m_u_movies[(u_id,m_id)])), dim=0))
                    # UMAM
                    support_umam_app.append(torch.cat(list(map(lambda x: movie_feature[x], u_m_a_movies[(u_id,m_id)])), dim=0))
                    # UMDM
                    support_umdm_app.append(torch.cat(list(map(lambda x: movie_feature[x], u_m_d_movies[(u_id,m_id)])), dim=0))

                query_x_app = None
                query_um_app = []
                query_umu_app = []
                query_umum_app = []
                query_umam_app = []
                query_umdm_app = []
                for m_id in tmp_movies[indices[-10:]]:
                    tmp_x_converted = torch.cat((movie_feature[m_id], user_feature[u_id]), 1)
                    try:
                        query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                    except:
                        query_x_app = tmp_x_converted

                    # meta-paths
                    # UM
                    query_um_app.append(torch.cat(list(map(lambda x: movie_feature[x], user_movies[u_id])), dim=0))
                    # UMU
                    query_umu_app.append(torch.cat(list(map(lambda x: user_feature[x], u_m_users[(u_id, m_id)])), dim=0))
                    # UMUM
                    query_umum_app.append(torch.cat(list(map(lambda x: movie_feature[x], u_m_u_movies[(u_id, m_id)])), dim=0))
                    # UMAM
                    query_umam_app.append(torch.cat(list(map(lambda x: movie_feature[x], u_m_a_movies[(u_id, m_id)])), dim=0))
                    # UMDM
                    query_umdm_app.append(torch.cat(list(map(lambda x: movie_feature[x], u_m_d_movies[(u_id, m_id)])), dim=0))

                support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
                query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])

                pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(self.output_dir, state, idx), "wb"))

                pickle.dump(support_um_app, open("{}/{}/support_um_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(support_umu_app, open("{}/{}/support_umu_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(support_umum_app, open("{}/{}/support_umum_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(support_umam_app, open("{}/{}/support_umam_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(support_umdm_app, open("{}/{}/support_umdm_{}.pkl".format(self.output_dir, state, idx), "wb"))

                pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(self.output_dir, state, idx), "wb"))

                pickle.dump(query_um_app, open("{}/{}/query_um_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(query_umu_app, open("{}/{}/query_umu_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(query_umum_app,open("{}/{}/query_umum_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(query_umam_app,open("{}/{}/query_umam_{}.pkl".format(self.output_dir, state, idx), "wb"))
                pickle.dump(query_umdm_app,open("{}/{}/query_umdm_{}.pkl".format(self.output_dir, state, idx), "wb"))

                with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(self.output_dir, state, idx), "w") as f:
                    for m_id in tmp_movies[indices[:-10]]:
                        f.write("{}\t{}\n".format(u_id, m_id))
                with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(self.output_dir, state, idx), "w") as f:
                    for m_id in tmp_movies[indices[-10:]]:
                        f.write("{}\t{}\n".format(u_id, m_id))
                # TODO: y label!!!!
                idx += 1


if __name__ == "__main__":
    input_dir = '../data/'
    output_dir = '../data/'
    ml = Movielens(os.path.join(input_dir, 'movielens_1m'), os.path.join(output_dir, 'movielens_1m'))
    ml.support_query_data()