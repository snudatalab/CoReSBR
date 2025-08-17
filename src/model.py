'''
***********************************************************************
Context-aware Sequential Bundle Recommendation via User-specific Representations

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: model.py
- SASRec with ideas of CoReBSR.
- This code is based on the implementation of https://github.com/pmixer/SASRec.pytorch.

Version: 1.0
***********************************************************************
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Point-wise Feed Forward Network for SASRec

input:
    * hidden_units: size of hidden layers
    * dropout_rate: rate of dropout
'''
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs
    

'''
I3. Multi-bundling Strategy
'''
class QueryWeightedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, T):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.T = T

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.head_dropout = nn.Dropout(p=0.2)

        self.head_weight_gen = nn.Sequential(
                                    nn.Linear(embed_dim, embed_dim),
                                    nn.ReLU(),
                                    nn.Linear(embed_dim, num_heads)
                            )

    def forward(self, items_emb, query):
        """
        items_emb: (B, L, I, D)  - item embeddings per bundle
        query:     (B, D)        - user embedding per batch
        return:    (B, L, D)     - attended bundle embeddings
        """
        B, L, I, D = items_emb.shape
        H = self.num_heads
        d_k = self.head_dim

        x = items_emb.view(B * L, I, D)  # (B*L, I, D)

        Q = self.q_proj(x).view(B * L, I, H, d_k).transpose(1, 2)  # (B*L, H, I, d_k)
        K = self.k_proj(x).view(B * L, I, H, d_k).transpose(1, 2)  # (B*L, H, I, d_k)
        V = self.v_proj(x).view(B * L, I, H, d_k).transpose(1, 2)  # (B*L, H, I, d_k)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B*L, H, I, I)
        attn_weights = F.softmax(attn_scores, dim=-1)                      # (B*L, H, I, I)
        attn_out = torch.matmul(attn_weights, V)                           # (B*L, H, I, d_k)
        
        head_outputs = attn_out.mean(dim=2)  # (B*L, H, d_k)

        # query: (B, L, D) → head_scales: (B, L, H)
        head_scales = self.head_dropout(F.softmax(self.head_weight_gen(query)/self.T, dim=-1))                   # (B, L, H)
        
        head_scales = head_scales.unsqueeze(-1)                        # (B, L, H, 1)
        head_scales = head_scales.expand(B, L, H, d_k)                 # (B, L, H, d_k)
        

        head_outputs = head_outputs.view(B, L, H, d_k) * head_scales      # (B, L, H, d_k)
        return head_outputs.reshape(B, L, D)   
  
    
'''
Framework of SASRec

input:
    * user_num: number of users
    * bundle_num: number of bundles
    * item_num: number of items
    hidden_units: size of hidden layers
    * dropout_rate: rate of dropout
'''
class SASRec(torch.nn.Module):
    def __init__(self, user_num, bundle_num, item_num, bundle_item, item_pop, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.bundle_num = bundle_num
        
        self.dev = args.device
        self.maxlen = args.maxlen
        self.hidden_units = args.hidden_units
        self.batch_size = args.batch_size
        self.register_buffer('bundle_item', torch.tensor(bundle_item, dtype=torch.int64))
        self.register_buffer('item_pop', torch.log(torch.tensor(item_pop, dtype=torch.int64)) + 1 + 1e-8)
        self.num_strategies = args.num_strategies

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.bundle_emb = torch.nn.Embedding(self.bundle_num+1, args.hidden_units, padding_idx=0)
        self.item_agg = nn.MultiheadAttention(embed_dim=args.hidden_units, num_heads=1, batch_first=True)
        self.strategies = QueryWeightedSelfAttention(embed_dim=args.hidden_units, num_heads=self.num_strategies, T=args.T)
        self.combine_proj = nn.Linear(3 * args.hidden_units, args.hidden_units)
        self.lmb = args.lmb
        self.gamma = args.gamma
        
        self.pos_emb = torch.nn.Embedding(args.maxlen, self.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(self.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
                
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    '''
    I1. User Representation
    
    input:
        * bundle_seqs: user's interaction history
    output:
        * user_emb: user representation 
    '''
    def get_user_emb(self, bundle_seqs, predict=False):
        
        B = bundle_seqs.shape[0]
        item_ids = self.bundle_item[bundle_seqs]
        if predict:
            item_ids = item_ids.unsqueeze(0)

        # Compute item frequency
        item_ids_flat = item_ids.view(B, -1) # (B, I)
        src = torch.ones_like(item_ids_flat, dtype=torch.int64, device=self.dev)
        item_freq = torch.zeros(B, self.item_num+1, dtype=torch.int64, device=self.dev)
        item_freq.scatter_add_(1, item_ids_flat, src)
        item_freq[:, 0] = 0
        
        # Compute preference score
        f_uj = torch.gather(item_freq.unsqueeze(1).expand(-1, self.maxlen, -1), 2, item_ids)  # (B, L, I)
        p_j = self.item_pop[item_ids].squeeze(-1) + 1e-8                                        # (B, L, I)
        pref_score = (f_uj / (p_j + 1e-8)) ** self.gamma    # (B, L, I)

        # Compute weighted sum of item embeddings
        item_vectors = self.emb_dropout(self.item_emb(item_ids))  # (B, L, I, D)
        weighted_vectors = pref_score.unsqueeze(-1) * item_vectors  # (B, L, I, D)
        bundle_embs = weighted_vectors.sum(dim=2)  # (B, L, D)

        # Recency weight only applied here
        position = torch.arange(self.maxlen, device=self.dev).view(1, self.maxlen, 1)
        recency_weight = torch.exp(-self.lmb * (self.maxlen - position))  # (1, L, 1)
        bundle_weights = recency_weight.squeeze(-1) # (1, L)
        norm = bundle_weights.sum(dim=1, keepdim=True)  # (1, 1)

        bundle_embs += self.pos_emb(position).squeeze(2)

        user_emb = (bundle_embs * bundle_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        user_emb = torch.where(norm > 0, user_emb / norm, torch.zeros_like(user_emb))
        return user_emb
    

    def log2feats(self, bundle_seqs, query=None, user_seqs=None):
        bundle_seqs = torch.LongTensor(bundle_seqs).to(self.dev)
        seqs = self.get_emb(bundle_seqs, query, user_seqs)
        # query = seqs[:, -1, :]
        seqs *= self.bundle_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(bundle_seqs.shape[1])), [bundle_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.tensor(positions, dtype=torch.int, device=self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (bundle_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask,)
        
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    '''
    Final bundle representation
    '''
    def get_emb(self, bundle_seqs=None, query=None, user_seqs=None):
        if bundle_seqs is None:
            item_embs = self.emb_dropout(self.item_emb(self.bundle_item)).unsqueeze(0)  # (1, L, I, D)
            query = query[:, None, :].expand(1, self.bundle_item.size(0), -1)  # (1, L, D)

            base = self.bundle_emb.weight.unsqueeze(0)  # (1, L, D)
            scaled_attn = self.attention_bundle_items(item_embs, query)  # (1, L, D)
            strategy = self.strategies(item_embs, query)  # (1, L, D)

            concat = torch.cat([base, scaled_attn, strategy], dim=-1)  # (1, L, 3D)
            return self.combine_proj(concat).squeeze(0).to(self.dev)   # (L, D)
        else:
            item_embs = self.emb_dropout(self.item_emb(self.bundle_item[bundle_seqs]))  # (B, L, I, D)
            query = query[:, None, :].expand(query.size(0), self.maxlen, query.size(1))  # (B, L, D)

            base = self.bundle_emb(bundle_seqs)  # (B, L, D)
            scaled_attn = self.attention_bundle_items(item_embs, query, bundle_seqs=bundle_seqs)
            strategy = self.strategies(item_embs, query)  # (B, L, D)

            concat = torch.cat([base, scaled_attn, strategy], dim=-1)  # (B, L, 3D)
            return self.combine_proj(concat).to(self.dev)  # (B, L, D)
    
    '''
    I2. Adaptive Bundle Aggregation
    
    input:
        * items_emb: embedding of items within bundles in user's interaction history
        * query: user representation
        * bundle_seqs: 
    output: user's interaction history
        * aggregated bundle representation
    '''
    def attention_bundle_items(self, items_emb, query, bundle_seqs=None):
        """
        items_emb: (B, L, I, D)
        query:     (B, L, D)
        return:    (B, L, D)
        """
        B, L, I, D = items_emb.size()

        items = items_emb.view(B * L, I, D)        # (B*L, I, D)
        query = query.reshape(B * L, 1, D)         # (B*L, 1, D)

        if bundle_seqs is None:
            mask = (self.bundle_item != 0).reshape(-1, I)  # padding 여부 마스크
        else:
            mask = (self.bundle_item[bundle_seqs] != 0).reshape(-1, I)  # padding 여부 마스크
        invalid_rows = mask.sum(dim=1) == 0
        mask[invalid_rows, 0] = True  # inplace 수정
        attn_output, _ = self.item_agg(query=F.normalize(query, dim=-1), key=F.normalize(items, dim=-1), value=items, key_padding_mask=~mask)
        

        return attn_output.squeeze(1).view(B, L, D)  # (B, L, D)

            
    def update_two_view(self):
        with torch.no_grad():
            self.two_view = (self.bundle_emb.weight + self.item_emb(self.bundle_item).sum(dim=1)).to(self.dev)  # [num_bundle, hidden]
        
    def forward(self, bundle_seqs, pos_seqs, neg_seqs, user_seqs=None):
        query = self.get_user_emb(bundle_seqs)
        log_feats = self.log2feats(bundle_seqs, query=query, user_seqs=user_seqs) 

        pos_embs = self.get_emb(torch.tensor(pos_seqs, device=self.dev), query=query, user_seqs=user_seqs)
        neg_embs = self.get_emb(torch.tensor(neg_seqs, device=self.dev), query=query, user_seqs=user_seqs)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits 


    def predict(self, bundle_seqs, user_seqs):
        query = self.get_user_emb(bundle_seqs, predict=True)
        log_feats = self.log2feats(bundle_seqs, query, user_seqs)

        final_feat = log_feats[:, -1, :]

        bundle_embs = self.get_emb(query=query, user_seqs=user_seqs)
        logits = bundle_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits.squeeze()
    