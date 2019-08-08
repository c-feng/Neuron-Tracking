import torch 
def pairwise_distances_tensors(x, y):
    xs = torch.sum(x**2 , dim = 1)[:,None]
    ys = torch.sum(y**2 , dim = 1)[None]
    l2_dist = xs + ys - 2 * torch.matmul(x, y.transpose(0,1))
    return l2_dist

def emb_match(cur_emb, prev_emb, prev_pred):
    #cur_emb e * h * w * d
    #prev_emb e * h * w * d
    #prev_pred h * w * d
    e,h,w,d = cur_emb.size()
    prev_pred_mask = prev_pred > 0
    y = prev_emb.masked_select(prev_pred_mask).view(e, -1).transpose(0, 1)
    x = cur_emb.view(e, -1).transpose(0, 1)
    l2_dist = pairwise_distances_tensors(x, y)

    dist_sel, _ = torch.min(l2_dist, dim = 1)
    #_, inds_sel = torch.min(l2_dist, dim = 1)
    #dist_sel = l2_dist[range(inds_sel.size(0)), inds_sel]

    #print("emb", cur_emb.view(-1))
    #print("prev emb", prev_emb.view(-1))
    
    #print("dist",dist_sel) 
    match_map = (1 - 2 /(1 + torch.exp(dist_sel))).view(h,w,d)
    #print("match",match_map.view(-1))
    return match_map

def batch_emb_match(batch_cur_emb, batch_prev_emb, batch_prev_pred):
    batch_match_map = []
    for cur_emb, prev_emb, prev_pred in zip(batch_cur_emb, batch_prev_emb, batch_prev_pred):
        batch_match_map.append(emb_match(cur_emb, prev_emb, prev_pred)[None])
    batch_match_map = torch.cat(batch_match_map, dim = 0)
    return batch_match_map

if __name__  == "__main__":
    a = torch.randint(0,3,(2,2))
    b = torch.randint(0,3,(1,2))
    l2_dist = pairwise_distances_tensors(a, b)
    print(a)
    print(b)
    print(l2_dist, l2_dist.size())
