#%%
import torch;
from torch import nn;

# see docs/mynet_base.md for details

# class RMSNorm(nn.Module):
#     def __init__(self,
#                  d_model: int,
#                  eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d_model))
#     # x: [batch, seq_len, emb_dim]
#     def forward(self, x):
#         output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
#         # output: same shape as x
#         return output

class SimpleNetV(nn.Module):
    def __init__(self, emb_dim, visit_num, voc_size):
        super(SimpleNetV,self).__init__()
        self.visit_num=visit_num
        self.diag_encoder = nn.Linear(voc_size[0], emb_dim)
        self.proc_encoder = nn.Linear(voc_size[1], emb_dim)
        self.med_encoder = nn.Linear(voc_size[2], emb_dim)
        self.output_layer = nn.Linear(emb_dim * visit_num, voc_size[2])

        # self.visit_encoder1=nn.Linear(emb_dim,emb_dim)
        # self.norm1=RMSNorm(emb_dim)
        # self.visit_encoder2=nn.Linear(emb_dim,emb_dim)
        # self.norm2=RMSNorm(emb_dim)
        # self.visit_encoder3=nn.Linear(emb_dim,emb_dim)
        # self.norm3=RMSNorm(emb_dim)
    # diags: [batch, seq_len, size_diag_voc]
    # procs: [batch, seq_len, size_pro_voc]
    # meds: [batch, seq_len, size_med_voc]
    def forward(self,diags,procs,meds):
        # d/p/m_emb: [batch, seq_len, emb_dim]
        seq_len=diags.size(1)

        with torch.no_grad():
            # crop or pad dim1 to visit_num
            if seq_len < self.visit_num:
                pad_len=self.visit_num - seq_len
                diags=torch.cat((diags,torch.zeros(diags.shape[0],pad_len,diags.shape[2],device=diags.device)),dim=1)
                procs=torch.cat((procs,torch.zeros(procs.shape[0],pad_len,procs.shape[2],device=procs.device)),dim=1)
                meds=torch.cat((meds,torch.zeros(meds.shape[0],pad_len,meds.shape[2],device=meds.device)),dim=1)
            else:
                diags=diags[:,:self.visit_num,:]
                procs=procs[:,:self.visit_num,:]
                meds=meds[:,:self.visit_num,:]
        
        # meds[:,:,:]=0

        # d/p/m_emb: [batch, visit_num, emb_dim]
        diags_emb = self.diag_encoder(diags)
        procs_emb = self.proc_encoder(procs)
        meds_emb = self.med_encoder(meds)
        # visits_emb: [batch, visit_num, emb_dim]
        visits_emb = diags_emb + procs_emb + meds_emb

        # visits_emb=self.visit_encoder1(self.norm1(visits_emb))+visits_emb
        # visits_emb=self.visit_encoder2(self.norm2(visits_emb))+visits_emb
        # visits_emb=self.visit_encoder3(self.norm3(visits_emb))+visits_emb
        # visits_emb=self.visit_encoder1(visits_emb)
        # visits_emb=self.visit_encoder2(visits_emb)
        # visits_emb=self.visit_encoder3(visits_emb)

        # patient_emb: [batch, visit_num * emb_dim]
        patient_emb = visits_emb.view(visits_emb.shape[0], -1)
        # output: [batch, size_med_voc]
        output=self.output_layer(patient_emb)
        return output
#%% A simple test.
# Our model can now handle variable length of sequence properly. The first visit_num of each sequence will be used, and the rest will be ignored. 
# - if the sequence is shorter than visit_num, it will be padded with zeros.)
# - The rest sequence will be utilized in the future.
if __name__ == "__main__":
    model=SimpleNetV(
        emb_dim=128,
        visit_num=3,
        voc_size=(10,20,30)
    ).to('cuda:0')

    seq_len=2
    batch_size=32
    x1 = torch.randn(batch_size, seq_len, 10, device='cuda:0')
    x2 = torch.randn(batch_size, seq_len, 20, device='cuda:0')
    x3 = torch.randn(batch_size, seq_len, 30, device='cuda:0')
    y = model(x1,x2,x3)
    print(y.shape) # (64, 30)
# %%
