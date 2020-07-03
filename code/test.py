# coding: utf-8
# author: lu yf
# create date: 2019-12-25 11:23
import math
import os
import pickle

import numpy as np
import multiprocessing as mp


# def dcg_at_k(scores):
#     # assert scores
#     return scores[0] + sum(sc / math.log(ind+1, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))  # ind+1!!!
#
#
# def ndcg_at_k(real_scores, predicted_scores):
#     assert len(predicted_scores) == len(real_scores)
#     idcg = dcg_at_k(sorted(real_scores, reverse=True))
#     return (dcg_at_k(predicted_scores) / idcg) if idcg > 0.0 else 0.0
#
#
# def ranking(real_score, pred_score, k_list):
#     # ndcg@k
#     ndcg = {}
#     for k in k_list:
#         sorted_idx = sorted(np.argsort(real_score)[::-1][:k])
#         r_s_at_k = real_score[sorted_idx]
#         p_s_at_k = pred_score[sorted_idx]
#
#         ndcg[k] = ndcg_at_k(r_s_at_k, p_s_at_k)
#     return ndcg
#
#
# predicted1 = [.4, .1, .8]
# predicted2 = [.0, .1, .4]
# predicted3 = [.4, .1, .0]
# actual = [.8, .4, .1, .0]
#
# print(ranking(np.array(actual), np.array(predicted1), [1,3]))
# print(ranking(np.array(actual), np.array(predicted2), [1,3]))
# print(ranking(np.array(actual), np.array(predicted3), [1,3]))
#
# print(dcg_at_k([3,2,3,0,1,2]))
# print(ranking(np.array([3,3,2,2,1,0]), np.array([3,2,3,0,1,2]), [6]))


def job(x):
    return x*x, x+x


def multicore():
    l = []

    pool = mp.Pool()
    res = pool.map(job, range(10))
    for r in res:
        l.append(r[0])
    print(res)
    print(l)


if __name__ == '__main__':
    # multicore()
    data_dir = os.path.join('../data', 'yelp')
    supp_xs =pickle.load(open("{}/{}/support_ubtb_0.pkl".format(data_dir, 'meta_training')))
    print(supp_xs)