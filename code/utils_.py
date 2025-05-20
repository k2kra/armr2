#%%
import torch;
import numpy as np;
from torch.optim import Adam;
import torch.nn.functional as F;
import torch.nn as nn;
import torch;
import json;
from torch.utils.data import DataLoader;
import time;
from sklearn.metrics import f1_score, average_precision_score;
import random;
import os;
import sys;
#%%

records_final_override = None

def create_log_file(folder_name='default'):
    file_path='../logs/{}/{}.txt'.format(folder_name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return open(file_path, 'w')

def log(file, s):
    print(s)
    if file:
        file.write(s+'\n')
        file.flush()

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def multi_label_metric(y_gt, y_pred, y_prob):
    # Jaccard系数
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if len(union)==0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)
    # 平均精确率
    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score
    # 平均召回率
    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score
    # 平均F1
    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score
    # Macro F1
    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)
    # 平均精确率
    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)
    # Macro F1
    # f1 = f1(y_gt, y_pred)
    # 平均精确率
    prauc = precision_auc(y_gt, y_prob)
    # Jaccard系数
    ja = jaccard(y_gt, y_pred)
    # Precision, Recall, F1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)
    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(med_records, ddi_A):
    all_pairs_cnt=0
    ddi_pairs_cnt=0
    for record in med_records:
        for i in range(len(record)):
            for j in range(i+1, len(record)):
                mi=record[i]; mj=record[j]
                all_pairs_cnt += 1
                if ddi_A[mi][mj]==1 or ddi_A[mj][mi]==1:
                    ddi_pairs_cnt += 1
    if all_pairs_cnt==0:
        return 0
    return ddi_pairs_cnt/all_pairs_cnt

def get_data(dataset_path):
    if records_final_override:
        data = records_final_override
    else:
        data = json.load(open(dataset_path + 'records_final.json', 'r'))
    voc = json.load(open(dataset_path + 'voc_final.json', 'r'))
    ddi_A=json.load(open(dataset_path + 'ddi_A_final.json', 'r'))
    ehr_A=json.load(open(dataset_path + 'ehr.json', 'r'))
    return data,voc,ddi_A,ehr_A

'''
输入：
    items: 只采样部分数据，-1表示使用全部数据（主要用来debug)
    visit_num: 每个病人的采样历史长度

返回的字典格式如下：
{
    'train': {
        'diag_seq': torch.tensor, # [batch, visit_num, size_diag_voc]
        'proc_seq': torch.tensor, # [batch, visit_num, size_pro_voc]
        'med_seq': torch.tensor, # [batch, visit_num, size_med_voc]
        'med_old': torch.tensor, # [batch, size_med_voc]
        'med_new': torch.tensor, # [batch, size_med_voc]
        'med_full': torch.tensor, # [batch, size_med_voc]
        'med_history': torch.tensor, # [batch, size_med
    },
    'eval': 参考train的形式,
    'eval_len': list, # 每个病人的访问次数按顺序排列,
    'tests': list, # 每个元素参考train的形式,
    'tests_len': list, # 每个元素参考eval_len的形式,
    'ddi_A': list, # DDI矩阵
    'size_diag_voc': int, # 诊断词典大小
    'size_pro_voc': int, # 检查词典大小
    'size_med_voc': int, # 药物词典大小
}
'''
def get_tensor_data(dataset_path, visit_num, items=-1):
    data,voc,ddi_A,ehr_A = get_data(dataset_path)
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    size_diag_voc, size_pro_voc, size_med_voc = len(diag_voc['idx2word']), len(pro_voc['idx2word']), len(med_voc['idx2word'])
    size_all_voc = size_diag_voc + size_pro_voc + size_med_voc

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将一个病人的N次入院数据转换成N个长度为num_adm的子序列，不足用0填充
    # 子序列中，下标0...num_adm-1，0为当前的入院数据，后面的为历史数据，也就是时间由近到远
    # 每一次入院数据中的d,p,m由list集合表示变成tensor多热向量表示
    def pat2seq(pat, num_adm):
        results=[]
        for start_idx in range(len(pat)-1, -1, -1):
            patseq=[]
            for i in range(num_adm):
                idx = start_idx - i
                if idx in range(0,len(pat)):
                    adm=pat[idx]
                    diag=torch.zeros(size_diag_voc);diag[torch.tensor(adm[0])]=1
                    proc=torch.zeros(size_pro_voc);proc[torch.tensor(adm[1])]=1
                    med=torch.zeros(size_med_voc);med[torch.tensor(adm[2])]=1
                    patseq += [diag, proc, med]
                else:
                    patseq.append(torch.zeros(size_all_voc))
            results.append(torch.cat(patseq))
        return results

    # return: tensor(samples, (size_diag_voc + size_pro_voc + size_med_voc) * visit_num)
    def data2tensor_with_history(data,num_adm):
        data_samples=[]
        for pat in data:
            data_samples += pat2seq(pat,num_adm)
        result = torch.stack(data_samples)
        return result
 
    def split_last_med_into_history_and_new_med_from_data_tensor(d, max_visit_num):
        # 所有的历史药物
        b=d.shape[0]
        med_history=torch.zeros((b,size_med_voc), dtype=torch.float32).to(d.device)
        for i in range(1, max_visit_num):
            med_history += d[:,size_all_voc * i + size_diag_voc+size_pro_voc:size_all_voc * (i+1)]
        med_history=(med_history>0)*1.0
        med_last=d[:,size_diag_voc+size_pro_voc:size_all_voc]
        # 最后一次处方中包含的历史药物
        med_last_history=med_last*med_history
        # 最后一次处方中的新开的药物
        med_last_new=med_last-med_last_history
        return med_last,med_last_history,med_last_new,med_history

    def data2tensor_with_history_and_drugsplit(data, num_adm):
        d=data2tensor_with_history(data,num_adm=num_adm).to(device)
        _,d_med_old,d_med_new,d_med_history=split_last_med_into_history_and_new_med_from_data_tensor(d,max_visit_num=num_adm)
        # print(d.shape,d_med_old.shape,d_med_new.shape,d_med_history.shape)
        return torch.concat((d,d_med_old,d_med_new,d_med_history),dim=1)
    
    def convert_to_multiple_tensors(t):
        split_sections=[]
        for i in range(visit_num):
            split_sections.append(size_diag_voc)
            split_sections.append(size_pro_voc)
            split_sections.append(size_med_voc)
        split_sections += [size_med_voc] * 3
        result = torch.split(t, split_sections, dim=-1)
        diags=[]
        procs=[]
        meds=[]
        for i in range(visit_num):
            diags.append(result[i*3])
            procs.append(result[i*3+1])
            meds.append(result[i*3+2])
        meds[0]=torch.zeros_like(meds[0], device=meds[0].device)
        return {
            'diag_seq': torch.stack(diags,dim=1),
            'proc_seq': torch.stack(procs,dim=1),
            'med_seq': torch.stack(meds,dim=1),
            'med_old': result[visit_num*3],
            'med_new': result[visit_num*3+1],
            'med_full': result[visit_num*3]+result[visit_num*3+1],
            'med_history': result[visit_num*3+2],
        }

    d_train=data2tensor_with_history_and_drugsplit(data_train,num_adm=visit_num).to(device)
    print('d_train.shape: ', d_train.shape, '| data_train: ', len(data_train))
    d_eval_len=[len(p) for p in data_eval]
    d_eval=data2tensor_with_history_and_drugsplit(data_eval,num_adm=visit_num).to(device)
    print('d_eval.shape: ', d_eval.shape, '| data_eval: ', len(data_eval))
    # 由于使用bootstraping sampling，所以test需要采样10次，每次80%
    d_tests=[]
    d_tests_len=[]
    for i in range(10):
        data_test_i=random.sample(data_test, int(len(data_test)*0.8))
        d_test=data2tensor_with_history_and_drugsplit(data_test_i,num_adm=visit_num).to(device)
        d_tests.append(d_test)
        d_tests_len.append([len(p) for p in data_test_i])
    print('10x d_test.shape: ', d_test.shape, '| data_test: ', len(data_test))

    return {
        'train': convert_to_multiple_tensors(d_train),
        'eval': convert_to_multiple_tensors(d_eval),
        'eval_len': d_eval_len,
        'tests': [convert_to_multiple_tensors(d_test) for d_test in d_tests],
        'tests_len': d_tests_len,
        'ddi_A': ddi_A,
        'size_diag_voc': size_diag_voc,
        'size_pro_voc': size_pro_voc,
        'size_med_voc': size_med_voc,
    }

def get_tensor_data_iii(visit_num=3):
    return get_tensor_data('../data/mimic-iii/', visit_num=visit_num)

def get_tensor_data_iv(visit_num=3):
    return get_tensor_data('../data/mimic-iv/', visit_num=visit_num)

def create_tensor_dataset(data):
    pass
            

'''
输入: 形式参考get_tensor_data的返回值中的'train'
输出：参考格式如下
[   # 1st patient
    {
        'visits': [
            # 1st visit
            {
                'd': [2, 3, 4, 5, 6, 7, 8, 9], # diagnosis
                'p': [2, 3, 4], # procedures
                'm': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] # medications
            }, 
            # 2nd visit
            {
                'd': [10, 11, 12, 9], 
                'p': [5, 6, 3], 
                'm': [2, 3, 4, 5, 7, 6, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19]
            },
            # last visit
            {
                ...
            },
        ]
        'med_old': [],
        'med_new': [],
        'med_full': [],
        'med_history': []
    }, 
    # 2nd patient
    # ...
]
'''
def get_list_from_tensor_data_dict(t):
    p_num=t['diag_seq'].shape[0]
    v_num=t['diag_seq'].shape[1]
    result=[{} for i in range(p_num)]
    for i in range(p_num):
        result[i]['visits']=[]
        for j in range(v_num ):
            result[i]['visits'].append({
                'd': torch.where(t['diag_seq'][i][j]==1)[0].tolist(),
                'p': torch.where(t['proc_seq'][i][j]==1)[0].tolist(),
                'm': torch.where(t['med_seq'][i][j]==1)[0].tolist()
            })
        result[i]['med_old']=torch.where(t['med_old'][i]==1)[0].tolist()
        result[i]['med_new']=torch.where(t['med_new'][i]==1)[0].tolist()
        result[i]['med_full']=torch.where(t['med_full'][i]==1)[0].tolist()
        result[i]['med_history']=torch.where(t['med_history'][i]==1)[0].tolist()
    return result

'''
输入: dim0相同的tensor数组
输出：按照dim0切分的数组

用于Mini-batch训练，输出的每个batch形状都一样。
'''
def tensors_to_chunks(tensors, chunk_size):
    result = []
    for t in tensors:
        result.append(torch.split(t, split_size_or_sections=chunk_size, dim=0))
    result2 = []
    for j in range(len(result[0])):
        b=[]
        for i in range(len(result)):
            b.append(result[i][j])
        result2.append(b)
    return result2

'''
打包batch无非这几种形式：全局padding，全局截断，动态padding，packing等。
这里使用动态padding的方式，即每个batch的长度不一样，但是每个batch内部的长度是一样的。

输入:list装的tensor,每个tensor的shape为[dim0, ...]，dim0可以不同，其他维度必须相同。

sort可以进一步提高空间利用效率，但是会破坏原有的顺序，所以在使用的时候要注意。

输入：
    1. batches: list形式的tensor，每个tensor的shape为[batch_size, max_dim0, ...]，max_dim0是这个batch中最大的dim0，用0填充max_dim0维度。
    2. batches_seq_lens: list[list[int]],shape为[batch_size,o_dim0]，对应padding之前每个tensor的dim0的值，也就是序列长度。
    3. sorted_indexes: 如果sort=True，这个参数是排序后的索引，用于后续的还原。
'''
#%%
def tensors_to_dynamic_batches(tensors, batch_size, sort=True):
    tensors_=tensors
    sorted_indexes=[i for i in range(len(tensors))]
    if sort:
        all_seq_lens=[(t.shape[0],i) for i,t in enumerate(tensors_)]
        sorted_indexes=[i for l,i in sorted(all_seq_lens)]
        tensors_=[tensors[i] for i in sorted_indexes]

    batches_seq_lens=[]
    batches=[]
    for i in range(0, len(tensors), batch_size):
        seq_lens=[t.shape[0] for t in tensors_[i:i+batch_size]]
        batches_seq_lens.append(seq_lens)
        batch=tensors_[i:i+batch_size]
        max_len=max([t.shape[0] for t in batch])
        # padding
        for j in range(len(batch)):
            t=batch[j]
            if t.shape[0]<max_len:
                padding=torch.zeros((max_len-t.shape[0],*t.shape[1:]), dtype=t.dtype, device=t.device)
                batch[j]=torch.cat((t,padding),dim=0)
        batches.append(torch.stack(batch,dim=0))
    return batches, batches_seq_lens,sorted_indexes

#%% test
# import random
# tensors=[]
# for i in range(100):
#     tensors.append(torch.ones(64))
# batches, batches_seq_lens=tensors_to_dynamic_batches(tensors, 16)
#%%

def print_patient_in_data(data,i):
    for j,visit in enumerate(data[i]):
        print('visit ',j,': ')
        d=visit[0]
        p=visit[1]
        m=visit[2]
        print('d: ',sorted(d))
        print('p: ',sorted(p))
        print('m: ',sorted(m))

'''
输入:
data: list[list[list[list[]]]]形式的数据.
    list[0]：第1个病人
    list[0][0]：第1个病人的第1次入院
    list[0][0][0]：第1个病人的第1次入院的d集合
    list[0][0][1]：第1个病人的第1次入院的p集合
    list[0][0][2]：第1个病人的第1次入院的m集合
    list[0][0][0][0]:int, 第1个病人的第1次入院的d集合中第一个代码
voc_size[0]: d的词典大小
voc_size[1]: p的词典大小
voc_size[2]: m的词典大小
max_visits: 采样的最大历史访问次数，-1代表不限制序列长度

输出: 
s_diags = [
    tensor[visit_num_1, d_voc_size],
    tensor[visit_num_2, d_voc_size],
    ...
]
s_procs = [
    tensor[visit_num_1, p_voc_size],
    tensor[visit_num_2, p_voc_size],
    ...
]
s_meds = [
    tensor[visit_num_1, m_voc_size],
    tensor[visit_num_2, m_voc_size],
    ...
]
s_meds dim0=-1的元素为0，用于占位。
labels=[
    tensor[m_voc_size],
    tensor[m_voc_size],
    ...
]
# ids记录每个tensor样本对应的data中病人的id，因为一个病人可能产生多个评估样本，后续会打乱样本的顺序，加上id方便后续按照病人对评测结果进行聚合，以及case study等。
ids=[
    0,
    0,
    1,
    ...
]

其中 dim0按照时间由近到远排列
'''
#%%
def create_dataset(data, voc_size, device, max_visits=-1):
    d_voc_size, p_voc_size, m_voc_size = voc_size
    diags=[]; procs=[]; meds=[]
    for patient in data:
        v_diags=[]; v_procs=[]; v_meds=[]
        for visit in patient:
            d_tensor=torch.zeros(d_voc_size, dtype=torch.float32)
            d_tensor[torch.tensor(visit[0])]=1
            v_diags.append(d_tensor.to(device))
            p_tensor=torch.zeros(p_voc_size, dtype=torch.float32)
            p_tensor[torch.tensor(visit[1])]=1
            v_procs.append(p_tensor.to(device))
            m_tensor=torch.zeros(m_voc_size, dtype=torch.float32)
            m_tensor[torch.tensor(visit[2])]=1
            v_meds.append(m_tensor.to(device))
        diags.append(torch.stack(v_diags,dim=0))
        procs.append(torch.stack(v_procs,dim=0))
        meds.append(torch.stack(v_meds,dim=0))
    # 对于一个长度为N的病人，采样N个样本，长度为1到N
    s_diags=[]; s_procs=[]; s_meds=[]; labels=[]; ids=[]
    for p_i in range(len(data)):
        for v_i in range(len(data[p_i])):
            # 注意，tensor切片是共享内存的，所以这里需要clone一下
            s_diags.append(diags[p_i][:v_i+1].clone())
            s_procs.append(procs[p_i][:v_i+1].clone())
            s_meds.append(meds[p_i][:v_i+1].clone())
            ids.append(p_i)
    # 屏蔽掉feature中的最后一个时间点的med，因为这个时间点是用来预测的，放到label中
    for m in s_meds:
        labels.append(m[-1].clone())
        m[-1]=0
    
    if max_visits > 0:
        s_diags = [d[-max_visits:] for d in s_diags]
        s_procs = [p[-max_visits:] for p in s_procs]
        s_meds = [m[-max_visits:] for m in s_meds]

    # 序列取个反向，时间顺序变成由近到远，这样方便后续模型对padding过的序列截断.
    s_diags = [d.flip(0) for d in s_diags]
    s_procs = [p.flip(0) for p in s_procs]
    s_meds = [m.flip(0) for m in s_meds]

    return s_diags,s_procs,s_meds,labels,ids

# 从原始数据中构建药物共作用矩阵
def create_ehr_adj(data,med_voc_size):
    ehr_adj=torch.zeros((med_voc_size,med_voc_size))
    for p in data:
        for v in p:
            m=v[2]
            for i in range(len(m)):
                for j in range(i+1,len(m)):
                    ehr_adj[m[i],m[j]]=1
                    ehr_adj[m[j],m[i]]=1
    return ehr_adj

def get_med_freq(data,med_voc_size):
    med_freq=torch.zeros(med_voc_size)
    for p in data:
        for v in p:
            m=v[2]
            for i in range(len(m)):
                med_freq[m[i]]+=1
    return med_freq/med_freq.sum()

def create_batches(diags, procs, meds, labels, ids, batch_size, sort=True):
    diag_batch,batch_seq_len,sorted_indexes=tensors_to_dynamic_batches(diags, batch_size, sort)
    proc_batch,_,_=tensors_to_dynamic_batches(procs, batch_size, sort)
    med_batch,_,_=tensors_to_dynamic_batches(meds, batch_size, sort)
    labels_sorted=[labels[i] for i in sorted_indexes]
    label_batch,_,_=tensors_to_dynamic_batches(labels_sorted, batch_size, False)
    ids_sorted=[ids[i] for i in sorted_indexes]
    ids_batch=[]
    for i in range(0, len(ids_sorted), batch_size):
        ids_batch.append(ids_sorted[i:i+batch_size])
    return diag_batch,proc_batch,med_batch,label_batch,ids_batch,batch_seq_len

def create_batches_dict(data, batch_size,voc_size, device, sort=True):
    tensors=create_batches(*create_dataset(data,voc_size,device=device), batch_size=batch_size, sort=sort)
    return {
        'diag_batches':tensors[0],  # list [batch, dynamic_max_seq_len, size_diag_voc]
        'proc_batches':tensors[1],  # list [batch, dynamic_max_seq_len, size_pro_voc]
        'med_batches':tensors[2],   # list [batch, dynamic_max_seq_len, size_med_voc]
        'label_batches':tensors[3], # list [batch, size_med_voc]
        'ids_batches':tensors[4],   # list [batch, 1], 每个样本对应到data的id
        'batch_seq_len':tensors[5], # list list[], batch中每个样本的实际长度
        'n_batches': len(tensors[0])
    }

def view_batch_item(batch_id, item_id, data,diag_batch,proc_batch,med_batch,label_batch,ids_batch,batch_seq_len):
    pid=ids_batch[batch_id][item_id]
    print_patient_in_data(data,pid)
    print('batch:')
    for j in range(batch_seq_len[batch_id][item_id]):
        print('d: ', torch.where(
            diag_batch[batch_id][item_id][j] == 1
        ))
        print('p: ', torch.where(
            proc_batch[batch_id][item_id][j] == 1
        ))
        print('m: ', torch.where(
            med_batch[batch_id][item_id][j] == 1
        ))
    print('l: ', torch.where(
        label_batch[batch_id][item_id] == 1
    ))

'''
#%%
d3=get_tensor_data('../data/mimic-iii/', visit_num=3, items=100)
l3=get_list_from_tensor_data_dict(d3['train'])
i = 55
print("\n\n\n")
print(l3[i])
print(torch.where(d3['train']['diag_seq'][i][0]==1))
print(torch.where(d3['train']['proc_seq'][i][0]==1))
print(torch.where(d3['train']['med_seq'][i][0]==1))

#%%
data,voc,ddi_A = get_data('../data/mimic-iii/', items=100)
voc_size=(len(voc['diag_voc']['idx2word']), len(voc['pro_voc']['idx2word']), len(voc['med_voc']['idx2word']))
diags, procs, meds,labels,ids = create_dataset(data, voc_size, 'cuda:0')
# %%

#%% batch验证
diag_batch,proc_batch,med_batch,label_batch,ids_batch,batch_seq_len = create_batches(diags, procs, meds, labels, ids, batch_size=32)

view_batch_item(6,2,data, diag_batch,proc_batch,med_batch,label_batch,ids_batch,batch_seq_len)
#%%

#%%
print(len(diags)==int(np.array([len(d) for d in data]).sum()))
print(len(diags))
#%%
print(batch_seq_len)
#%%
'''

'''
[[[8, 9, 10, 7],
    [3, 4, 1],
    [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17]],
[[0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]]
'''

# %%
