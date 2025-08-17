'''
***********************************************************************
Context-aware Sequential Bundle Recommendation via User-specific Representations

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: utils.py
- Utils for data preparation and evaluation including reranking.

Version: 1.0
***********************************************************************
'''


import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm
import pandas as pd


def setup_seed(seed):
    """
    Set random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def get_neg(train_bundle, ts, size): 
    neg = []
    p = np.array([1] * len(train_bundle), dtype=np.float64)
    if ts:
        p[np.array(ts, dtype=int)-1] = 0
    p /= p.sum()
    neg = np.random.choice(train_bundle, size, p=p).tolist()
    return neg


def sample_function(dataset, user_train, user_num, bundle_num, batch_size, maxlen, result_queue, SEED):
    
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    """
    Get instances
    """
    def sample():

        user = np.random.randint(1, user_num+1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, user_num+1)
        
        ts = user_train[user]
        
        if len(ts) <= maxlen+1:
            zeros = [0] * (maxlen-len(ts)+1)
            bundle_seq = ts[:-1]
            pos = ts[1:]
            size = len(pos)
            bundle_seq = zeros + bundle_seq
            pos = zeros + pos
            neg = get_neg(range(1, bundle_num+1), False, size)
            neg = zeros + neg
        else:
            k = random.choice(range(maxlen+1, len(ts)))
            bundle_seq = ts[k-maxlen-1:k-1]
            pos = ts[k-maxlen:k]
            size = len(pos)
            neg = get_neg(range(1, bundle_num+1), False, size)
        return (user, bundle_seq, pos, neg)

    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """
    Sampler for training
    """
    def __init__(self, dataset, user_train, user_num, bundle_num, batch_size=64, maxlen=10, n_workers=1, seed=2025):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(dataset,
                                                      user_train,
                                                      user_num,
                                                      bundle_num,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      seed + i
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def data_partition(fname):
    """
    Train, valid, test split
    """
    user_num = 0
    bundle_num = 0
    User = defaultdict(list)
    user_train = {} 
    user_valid = {}
    user_test = {}
    user_bundle = pd.read_csv(f'datasets/{fname}/user-bundle.txt', sep='\t', header=None, names=['user', 'bundle', 'timestamp'])
    user_num = user_bundle['user'].max()
    bundle_num = user_bundle['bundle'].max()
    User = user_bundle.groupby('user')['bundle'].agg(list).to_dict()
    
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    
    
    bundle_item = pd.read_csv('datasets/'+fname+'/bundle-item_mat.txt', sep='\t', header=None).to_numpy()
    item_num = bundle_item.flatten().max()
    item_pop = pd.read_csv('datasets/'+fname+'/item_pop.txt', sep='\t', header=None).to_numpy()
        
    return [user_train, user_valid, user_test, bundle_item, item_pop, user_num, bundle_num, item_num] 


def evaluate(model, dataset, args, test_f=False):
    """
    Evaluation
    """
    train, valid, test, bundle_item, item_pop, user_num, bundle_num, item_num = dataset
    
    test_user = 0.0
    HT = np.zeros(2, dtype=float)
    nDCG = np.zeros(2, dtype=float)
    all_rank = 0
    
    if not test_f and user_num > 1000:
        users = random.sample(range(1, user_num+1), 1000)
    else:
        users = range(1, user_num+1)
            
    for u in tqdm(users):
        if valid[u] == [] or test[u] == []:
            continue
        test_user += 1
        
        if test_f:
            train[u] = train[u] + valid[u]
            
        bundle_seq = train[u][-args.maxlen:]
        if len(bundle_seq) != args.maxlen:
            zeros = np.zeros(args.maxlen-len(bundle_seq))
            bundle_seq = np.concatenate((zeros, bundle_seq))

        predictions = model.predict(np.array([bundle_seq]), np.array([u]))
        predictions[0] = -np.inf
        if args.dataset in ['ml-10m', 'ml-20m', 'ml-30m', 'allrecipes']:
            predictions[train[u]] = -np.inf
        
        if test_f:
                
            for i, k in enumerate([5, 20]):
                _, topk = torch.topk(predictions, k)
                if test[u][0] in topk:
                    rank = np.where(topk.cpu() == test[u][0])[0]
                    HT[i] += 1
                    nDCG[i] += 1.0 / np.log2(rank+2)
                
        else:
            _, topk = torch.topk(predictions, 5)
            
            if valid[u][0] in topk:
                rank = np.where(topk.cpu() == valid[u][0])[0]
                all_rank += rank
                HT[0] += 1
                nDCG[0] += 1.0 / np.log2(rank+2)
        
    return {'recall': HT / test_user, 'nDCG': nDCG / test_user}