'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import psutil
import os


CORES = multiprocessing.cpu_count() // 2

# Global metrics tracking
EPOCH_METRICS = {
    'epoch_times': [],
    'sample_times': [],
    'batch_times': [],
    'memory_usage': [],
    'cpu_usage': []
}


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    # Track epoch start time
    epoch_start = time()
    
    # Get initial CPU and memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    cpu_before = process.cpu_percent(interval=0.1)
    
    with timer(name="Sample"):
        sample_start = time()
        S = utils.UniformSample_original(dataset)
        sample_time = time() - sample_start
        EPOCH_METRICS['sample_times'].append(sample_time)
        
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    
    # Track batch timing for energy efficiency
    batch_times = []
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        batch_start = time()
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        batch_time = time() - batch_start
        batch_times.append(batch_time)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalar(f'Performance/batch_time', batch_time, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    
    # Calculate metrics
    aver_loss = aver_loss / total_batch
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    total_epoch_time = time() - epoch_start
    
    # Get final CPU and memory usage
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    cpu_after = process.cpu_percent(interval=0.1)
    mem_used = mem_after - mem_before
    cpu_avg = (cpu_before + cpu_after) / 2
    
    # Store metrics
    EPOCH_METRICS['epoch_times'].append(total_epoch_time)
    EPOCH_METRICS['batch_times'].append(avg_batch_time)
    EPOCH_METRICS['memory_usage'].append(mem_after)
    EPOCH_METRICS['cpu_usage'].append(cpu_avg)
    
    # Log to tensorboard
    if world.tensorboard:
        w.add_scalar(f'Performance/epoch_time', total_epoch_time, epoch)
        w.add_scalar(f'Performance/sample_time', sample_time, epoch)
        w.add_scalar(f'Performance/avg_batch_time', avg_batch_time, epoch)
        w.add_scalar(f'Performance/memory_MB', mem_after, epoch)
        w.add_scalar(f'Performance/cpu_percent', cpu_avg, epoch)
    
    time_info = timer.dict()
    timer.zero()
    
    quant_info = "QUANT" if world.config.get('quantization', False) else "FLOAT"
    return f"loss{aver_loss:.3f}-{time_info}-batch:{avg_batch_time:.4f}s-epoch:{total_epoch_time:.2f}s-mem:{mem_after:.1f}MB-cpu:{cpu_avg:.1f}%-{quant_info}"

    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
