# MetaHIN
Source code for KDD 2020 paper "[Meta-learning on Heterogeneous Information Networks for Cold-start Recommendation](https://yuanfulu.github.io/publication/KDD-MetaHIN.pdf)"


# Requirements

- Python 3.6.9
- PyTorch 1.4.0
- My operating system is Ubuntu 16.04.1 with one GPU (GeForce RTX) and CPU (Intel Xeon W-2133)
- Detailed [requirements](https://github.com/rootlu/MetaHIN/blob/master/requirements.txt)

# Datasets
We have uploaded the original data of DBook, Movielens and Yelp in `data/` folder.
The processed data of DBook and Movielens can be downloaded from Google Drive and [BaiduYun](https://pan.baidu.com/s/15xD5WSdcG4m0oXgfABAGkQ) (Extraction code: ened).
The processed data of Yelp can be generate by the code [data/yelp/YelpProcessor.ipynb](https://github.com/rootlu/MetaHIN/blob/master/data/yelp/YelpProcessor.ipynb).

# Description

```
MetaHIN/
├── code
│   ├── main.py：the main funtion of model
│   ├── Config.py：configs for model
│   ├── Evaluation.py: evaluate the performance of learned embeddings w.r.t clustering and classification
│   ├── DataHelper.py: load data
│   ├── EmbeddingInitializer.py: map feature and inilitize embedding tables
│   ├── HeteML_new.py: update paramerters in meta-learning paradigm 
│   ├── MetaLeaner_new.py: the base model 
├── data
│   └── dbook
│       ├── original/: the original data without any preprocess
│       ├── DBookProcessor.ipynb: preprocess data 
│   └── movielens
│       ├── original/: the original data without any preprocess
│       ├── MovielensProcessor.ipynb: preprocess data 
│   └── yelp
│       ├── original/: the original data without any preprocess
│       ├── YelpProcessor.ipynb: preprocess data 
├── README.md
```

# Reference

```
@inproceedings{lu2020meta,
  title={Meta-learning on Heterogeneous Information Networks for Cold-start Recommendation},
  author={Lu, Yuanfu and Fang, Yuan and Shi, Chuan},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1563--1573},
  year={2020}
}

```

