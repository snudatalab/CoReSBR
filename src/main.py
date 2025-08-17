'''
***********************************************************************
Context-aware Sequential Bundle Recommendation via User-specific Representations

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: main.py
- A main class for training and evaluation of CoReSBR.

Version: 1.0
***********************************************************************
'''

import os
import time
import torch
import argparse
from tqdm import tqdm

from model import SASRec
from utils import *

from collections import defaultdict

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="foods", type=str)
parser.add_argument('--name', default="run1", type=str)
parser.add_argument('--seed', default=2025, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=256, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--T', default=0.1, type=float)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--test', default=False, type=bool)
parser.add_argument('--test_epoch', default=1, type=int)
parser.add_argument('--lmb', default=10, type=float)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--num_strategies', default=2, type=int)


args = parser.parse_args()

# maxlen setting
if args.dataset == 'foods':
    args.maxlen = 50
elif args.dataset == 'allrecipes':
    args.maxlen = 50
elif args.dataset == 'ml-10m':
    args.maxlen = 30

if not os.path.isdir('runs/'+args.dataset + '/' + args.name):
    os.makedirs('runs/'+args.dataset + '/' + args.name)
with open(os.path.join('runs/'+args.dataset + '/' + args.name, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()


if __name__ == '__main__':
    setup_seed(args.seed)

    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, bundle_item, item_pop, user_num, bundle_num, item_num] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])

    # prints data information
    print('\ndataset:', args.dataset)
    print('name:', args.name)
    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('user num:', user_num)
    print('bundle num:', bundle_num)
    print('item num:', item_num)
    

    sampler = WarpSampler(args.dataset, user_train, user_num, bundle_num, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(user_num, bundle_num, item_num, bundle_item, item_pop, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass


    if args.test:            
        model.load_state_dict(torch.load('runs/'+args.dataset + '/' + args.name+'/model.pth', map_location=torch.device(args.device)))
    f = open(os.path.join('runs/'+args.dataset + '/' + args.name, 'log.txt'), 'a')
        
    model.train()
    epoch_start_idx = 1
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    
    T = 0.0
    t0 = time.time()
    best = 0
    
    folder = 'runs/'+args.dataset + '/' + args.name
    fname = 'model.pth'
    fname = fname.format(args.num_epochs)              


    # training
    p = 0
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.test: break 
        model.train()
        for step in tqdm(range(num_batch)): 
            u, bundle_seq, pos, neg = sampler.next_batch() 
            u, bundle_seq, pos, neg = np.array(u), np.array(bundle_seq), np.array(pos), np.array(neg)
            
            optimizer.zero_grad()
            pos_logits, neg_logits = model(bundle_seq, pos, neg, u)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.bundle_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            
            
            loss.backward()
            optimizer.step()   
        
        if epoch % args.test_epoch == 0:
            model.eval()
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            print(f'epoch: {epoch}, loss: {round(loss.item(), 6)}, ', ', '.join([key + '@20: ' + str(round(value[0], 4)) for key, value in t_test.items()]))
            f.write(f'epoch: {epoch}, loss: {round(loss.item(), 6)}, '+', '.join([key + '@20: ' + str(round(value[0], 4)) for key, value in t_test.items()])+'\n')
            f.flush()
            t0 = time.time()
            model.train()
            
            if t_test['nDCG'][0] > best:
                p = 0
                best = t_test['nDCG'][0]
                torch.save(model.state_dict(), os.path.join(folder, fname))
            else:
                p += 1
                if p > 20:
                    f.write('Done')
                    break 
            
    
    # inference
    model.load_state_dict(torch.load(os.path.join(folder, fname), map_location=torch.device(args.device)))
    model.eval()
    t_test = evaluate(model, dataset, args, test_f=True)

    print(f'\n----- original (k=5, 20) ----- \n')
    f.write(f'\n----- original (k=5, 20) ----- \n')
    print('\n'.join([key + ':\t' + str([round(i, 4) for i in value]) for key, value in t_test.items()]))
    f.write('\n'.join([key + ':\t' + str([round(i, 4) for i in value]) for key, value in t_test.items()]))
    f.close()
    sampler.close()