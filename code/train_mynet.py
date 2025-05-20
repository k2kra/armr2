#%%
from utils_ import *;
from net_mynet import MyNet
import argparse
import copy;

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=3, help='3(mimic-iii) or 4(mimic-iv)')
parser.add_argument('--dim', type=int, default=256, help='embedding dimension')
parser.add_argument('--batch', type=int, default=32, help='mini-batch size')
parser.add_argument('--visit', type=int, default=3, help='history visit length')
parser.add_argument('--seed', type=int, default=1203, help='seed')
parser.add_argument('--test', type=int, default=0, help='test mode')
parser.add_argument('--save_model', type=int, default=0, help='save model')
parser.add_argument('--epoches', type=int, default=56, help='epoches')
# args = parser.parse_args(args=[])
args = parser.parse_args()
#%%
print('loading data...')
# load data
dataset=''
if args.dataset == 3:
    dataset = 'mimic-iii'
elif args.dataset == 4:
    dataset = 'mimic-iv'
    args.epoches=args.epoches//args.dataset
data,voc,ddi_A,ehr_A=get_data('data/{}/'.format(dataset))
size_diag_voc, size_pro_voc, size_med_voc = len(voc['diag_voc']['idx2word']), len(voc['pro_voc']['idx2word']), len(voc['med_voc']['idx2word'])
voc_size=(size_diag_voc, size_pro_voc, size_med_voc)

split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point:split_point + eval_len]
data_eval = data[split_point+eval_len:]

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

med_freq=get_med_freq(data,size_med_voc).to(device)
ehr_A_gpu=torch.tensor(ehr_A).to(device)
ddi_A_gpu=torch.tensor(ddi_A).to(device)

model = MyNet(emb_dim=args.dim, k=args.visit, voc_size=voc_size,ehr_adj=ehr_A_gpu,ddi_adj=ddi_A_gpu).to(device)
optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-05, weight_decay=1e-06)

# (diag_batch,proc_batch,med_batch,label_batch,ids_batch,batch_seq_len)
infer_batch_size=256
train_batches=create_batches_dict(data_train, args.batch,voc_size,device)
eval_batches=create_batches_dict(data_eval, infer_batch_size,voc_size,device)

#%%
import torch
import torch.nn.functional as F
#%%

#%%
# shape: [batch, size_med_voc]
def loss_func(output, y_gt):
    # focal loss?
    return F.binary_cross_entropy_with_logits(output, y_gt)
    # bce_loss=F.binary_cross_entropy_with_logits(output, y_gt, reduction='none')
    # weights=1.0-med_freq
    # return (weights.view(1,-1).repeat(output.shape[0],1)*bce_loss).sum() * 10

def infer(model, batches, threshold):
    model.eval()
    with torch.no_grad():
        y_gt_full=[]
        y_pred_full=[]
        y_pred_prob_full=[]
        loss_full=0
        n_samples=0
        for batch_id in range(batches['n_batches']):
            output=model(batches['diag_batches'][batch_id], batches['proc_batches'][batch_id], batches['med_batches'][batch_id])
            # y_pred_prob: [batch, size_med_voc], the probability of each med label
            y_pred_prob=F.sigmoid(output)
            # y_pred: [batch, size_med_voc], the predicted med labels using threshold
            y_pred=(y_pred_prob >= threshold)
            y_gt=batches['label_batches'][batch_id]
            loss=loss_func(output,y_gt)
            y_gt_full.append(y_gt)
            y_pred_full.append(y_pred)
            y_pred_prob_full.append(y_pred_prob)
            loss_full+=loss.item() * y_gt.shape[0]
            n_samples+=y_gt.shape[0]
        return torch.cat(y_gt_full).cpu(), torch.cat(y_pred_full), torch.cat(y_pred_prob_full), loss_full/n_samples
#%%
# result=infer(model, eval_batches,threshold=0.3)
#%%
def infer_on_validation_data(model,threshold):
    return infer(model, eval_batches,threshold)

def get_metrics(model, batches, threshold=0.3):
    # restore the original order of samples to aggregate patient-level labels and predictions
    batches=eval_batches
    indexes=np.concatenate([np.array(a) for a in batches['ids_batches']])
    sorted_indexes=sorted([(i,indexes[i]) for i in range(len(indexes))], key=lambda x: x[1])
    o_indexes=[t[0] for t in sorted_indexes]
    p_len=[]
    counter=0
    for i in range(1,len(sorted_indexes)):
        if sorted_indexes[i][1] != sorted_indexes[i-1][1]:
            p_len.append(i-counter)
            counter=i
    p_len.append(len(sorted_indexes)-counter)

    infer_result=infer(model, batches, threshold)
    y_gt, y_pred,y_pred_prob,loss=infer_result
    y_gt=torch.stack([y_gt[i] for i in o_indexes],dim=0)
    y_pred=torch.stack([y_pred[i] for i in o_indexes],dim=0).cpu().numpy().astype(np.int32)
    y_pred_prob=torch.stack([y_pred_prob[i] for i in o_indexes],dim=0).cpu().numpy().astype(np.float64)

    # calculate scores in patient-level
    idx=0
    patient_scores=[]
    for len_visits in p_len:
        pred_meds=np.array(y_pred[idx:idx+len_visits,:])
        pred_meds_list=[np.where(pred_meds[i] == 1)[0].tolist() for i in range(pred_meds.shape[0])]
        result:list=list(multi_label_metric(
            np.array(y_gt[idx:idx+len_visits,:]),
            pred_meds,
            np.array(y_pred_prob[idx:idx+len_visits,:])
        ))
        result.append(ddi_rate_score(pred_meds_list, ddi_A))
        patient_scores.append(result)
        idx += len_visits
    patient_scores=np.array(patient_scores)
    ja,prauc,avg_precision,avg_recall,avg_f1,ddi=patient_scores.mean(axis=0)
    # average med num
    mean_med=y_pred.sum(axis=1).mean()
    return ja,prauc,avg_precision,avg_recall,avg_f1,ddi,mean_med,loss
#%%
# result=get_metrics(model, eval_batches, threshold=0.3)
#%%

def get_metrics_on_validation_data(model):
    return get_metrics(model, eval_batches)

def get_metrics_on_test_data(model):
    results=[]
    for i in range(10):
        random.shuffle(data_test)
        test_batches=create_batches_dict(data_test[:int(len(data_test)*0.8)], infer_batch_size,voc_size,device)
        result = list(get_metrics(model, test_batches))
        print('iteration %d / 10' % (i+1))
        results.append(result)
    results=np.array(results)
    return results.mean(axis=0), results.std(axis=0)
#%%
# result=get_metrics_on_test_data(model)
#%%

def train(model, optimizer):
    log_file=create_log_file('train_mynet')
    log(log_file, 'config: ' + str(args) + '\n')
    log(log_file, "traing...")
    log(log_file, time.strftime("Date: %Y%m%d-%H%M%S"))
    log(log_file, "File: {}".format(__file__) + '\n')
    log(log_file, 'params: %d' % count_parameters(model))
    log(log_file, 'epoch\tja\tprauc\tavg_p\tavg_r\tavg_f1\tddi\tavg_med\tt_loss\tv_loss')

    start_time=time.time()
    continuous_decline=0
    prev_prauc = 0.0
    best=[0]*100
    best_model_params=model.state_dict()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for epoch in range(args.epoches):
        train_loss_sum=0
        n_samples=0
        model.train()
        for i in range(train_batches['n_batches']):
            blen=train_batches['diag_batches'][i].shape[0]
            n_samples+=blen
            output=model(train_batches['diag_batches'][i], train_batches['proc_batches'][i], train_batches['med_batches'][i])
            loss=loss_func(output, train_batches['label_batches'][i])
            train_loss_sum += loss.detach().item() * blen
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('train epoch progress: {}/{}\r'.format(i+1, train_batches['n_batches']), end='', flush=True)
        print('validating...' + ' ' * 100 + '\r',end='', flush=True)
        # check performance
        ja,prauc,avg_precision,avg_recall,avg_f1,ddi,mean_med,v_loss=get_metrics_on_validation_data(model)
        t_loss=train_loss_sum/n_samples
        metric=(epoch+1,ja,prauc,avg_precision,avg_recall,avg_f1,ddi,mean_med,t_loss,v_loss)
        print(' ' * 100 + '\r',end='', flush=True)
        log(log_file, '%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f\t%.4f\t%.4f' % metric)
        # save the best model
        if prauc > best[2]:
            best = metric
            best_model_params = copy.deepcopy(model.state_dict())
        # early stop
        if prauc < prev_prauc:
            continuous_decline += 1
            if continuous_decline >= 2:
                log(log_file, ' [ early stop ] ')
                break
        else:
            continuous_decline = 0
        prev_prauc = prauc
    log(log_file, 'best:')
    log(log_file, '%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f\t%.4f\t%.4f' % best)
    log(log_file, "training stopped.")
    log(log_file, "Time used: %.2f" % (time.time()-start_time))
    log_file.close()
    # end training
    return best_model_params


# %%
best_model_params=train(model, optimizer)
if args.save_model > 0:
    torch.save(best_model_params, 'weights/net_base_v_1.pth')
if args.test > 0:
    model.load_state_dict(best_model_params)
    result=get_metrics_on_test_data(model)
    print(result)
# %%
# model.load_state_dict(torch.load('weights/net_base_v_20240228-202840.pth'))
# y=model(eval_batches['diag_batches'][0], eval_batches['proc_batches'][0], eval_batches['med_batches'][0])
# %%
