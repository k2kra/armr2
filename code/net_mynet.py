#%%
import torch
from torch import nn
from net_base_v import SimpleNetV
import math
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union

#%%
@dataclass
class MambaModelArgs:
    d_model: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaBlock(nn.Module):
    def __init__(self, args: MambaModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

#%%
# test
if __name__ == '__main__':
    b,l,d = 2,3,4
    x = torch.randn(b,l,d)
    m= MambaBlock(MambaModelArgs(d_model=d))
    y = m(x)
#%%

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        device = ehr_adj.device

        self.ehr_adj = self.normalize(ehr_adj + torch.eye(ehr_adj.shape[0]).to(device))
        self.ddi_adj = self.normalize(ddi_adj + torch.eye(ddi_adj.shape[0]).to(device))
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ehr_node_embedding = F.relu(ehr_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)

        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    # def normalize(self, mx):
    #     """Row-normalize sparse matrix"""
    #     rowsum = np.array(mx.sum(1))
    #     r_inv = np.power(rowsum, -1).flatten()
    #     r_inv[np.isinf(r_inv)] = 0.
    #     r_mat_inv = np.diagflat(r_inv)
    #     mx = r_mat_inv.dot(mx)
    #     return mx

    # gpu version
    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        mx = mx.to_dense()
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx
    
#%%
# test
if __name__ == '__main__':
    ehr_adj=(torch.rand((131,131))>0.7).float()
    ddi_adj=(torch.rand((131,131))>0.7).float()
    gcn=GCN(131, 64, ehr_adj, ddi_adj)
    ehr_node_embedding, ddi_node_embedding=gcn()

# %%
class PiecewiseTSL(nn.Module):
    def __init__(self, emb_dim, k):
        super().__init__()
        self.k=k
        self.mamba=MambaBlock(MambaModelArgs(d_model=emb_dim))
        self.lin=torch.nn.Linear(k*emb_dim,k*emb_dim)
        self.norm=torch.nn.LayerNorm(k*emb_dim)

        # self.gru=nn.GRU(emb_dim,emb_dim,batch_first=True)
    # seq: [batch, seq_len, emb_dim]
    # output: [batch, emb_dim*2]
    def forward(self,seq):
        batch=seq.shape[0]
        seq_len=seq.shape[1]
        emb_dim=seq.shape[2]
        if seq_len < self.k:
            pad_len=self.k-seq_len
            seq=torch.cat((seq,torch.zeros(batch,pad_len,emb_dim,device=seq.device)),dim=1)
        # else:
        #     seq=seq[:,:self.k,:]
        # [batch, k, emb_dim]
        near_seq=seq[:,:self.k,:]
        near_h=self.lin(self.norm(near_seq.view(batch,-1))).view(batch,self.k,emb_dim)+near_seq
        far_h=torch.zeros(batch,self.k,emb_dim,device=seq.device)
        if seq_len > self.k:
            # [batch, seq_len-k, emb_dim]
            far_seq=self.mamba(torch.flip(seq[:,self.k:,:],[1]))
            q=near_h
            kv=far_seq
            # scale=torch.sqrt(torch.tensor(kv.size(-1), dtype=torch.float32,device=seq.device))
            scores=attn_scores = torch.bmm(q, kv.transpose(-2, -1))
            scores=nn.functional.softmax(scores, dim=-1)
            far_h = torch.bmm(attn_scores, kv)
        # [batch, emb_dim*2k]
        output = torch.concat((near_h, far_h), dim=1)
        return output

        # result=self.gru(seq)[0]
        # return torch.concat((result,result.clone().detach()),dim=1)
#%%
# q=torch.randn((32,3,128))
# kv=torch.randn((32,5,128))
# attn_scores = torch.bmm(q, kv.transpose(-2, -1))
# far_h=torch.bmm(attn_scores, kv)

# test
if __name__ == '__main__':
    model=PiecewiseTSL(128,3).to('cuda:0')
    seq=torch.randn(32, 2, 128, device='cuda:0')
    y=model(seq)
    seq=torch.randn(32, 5, 128, device='cuda:0')
    y=model(seq)

#%%
class PatientRepLearn(nn.Module):
    def __init__(self, emb_dim, k, d_voc_size, p_voc_size, m_voc_size):
        super().__init__()
        self.emb_dim=emb_dim
        self.tsl_d=PiecewiseTSL(emb_dim,k)
        self.tsl_p=PiecewiseTSL(emb_dim,k)
        self.d_lin=nn.Linear(d_voc_size,emb_dim)
        self.p_lin=nn.Linear(p_voc_size,emb_dim)
        self.m_lin=nn.Linear(m_voc_size,emb_dim)
    # diags: [batch, seq_len, d_voc_size]
    # procs: [batch, seq_len, p_voc_size]

    def forward(self, diags, procs, meds):
        # [batch, seq_len, emb_dim]
        e_d=self.d_lin(diags)
        e_p=self.p_lin(procs)
        e_m=self.m_lin(meds)
        e_h=e_d+e_p+e_m
        # [batch, 2k, emb_dim]
        h_d=self.tsl_d(e_d)
        h_p=self.tsl_p(e_p)

        if diags.size(1) < h_d.size(1):
            pad_len=h_d.size(1)-diags.size(1)
            e_h=torch.cat((e_h,torch.zeros(e_h.shape[0],pad_len,e_h.shape[2],device=e_h.device)),dim=1)
        else:
            e_h=e_h[:,:h_d.size(1),:]
        h_patient=torch.concat((e_h,h_d+h_p),dim=1)
        return h_patient

#%%
# test
if __name__=='__main__':
    diags=torch.randn(32, 5, 10, device='cuda:0')
    procs=torch.randn(32, 5, 20, device='cuda:0')
    model=PatientRepLearn(128,3,10,20).to('cuda:0')
    y=model(diags,procs)
    print(y.shape) # (32, 6, 128)
#%%
class MedRepLearn(nn.Module):
    def __init__(self, emb_dim, k, m_voc_size):
        super().__init__()
        self.emb_dim=emb_dim
        self.m_voc_size=m_voc_size
        self.tsl_old=PiecewiseTSL(emb_dim,k)
        self.tsl_new=PiecewiseTSL(emb_dim,k)
        self.m_embs=nn.Embedding(m_voc_size,emb_dim)
        self.lin_expand=nn.Linear(emb_dim,2*k*emb_dim)
    # meds: [batch, seq_len, m_voc_size]
    def forward(self,meds):
        batch=meds.shape[0]
        # dynamically calculate historical and new drugs
        # [batch, seq_len, m_voc_size]
        history=torch.concat((torch.zeros(batch,1,self.m_voc_size,device=meds.device),(torch.cumsum(meds,dim=1)>0).float()[:,:-1,:]),dim=1)
        old=meds*history
        new=meds-old
        # multi-hot to embeddings
        # [batch, seq_len, emb_dim]
        e_old=torch.matmul(old,self.m_embs.weight)
        e_new=torch.matmul(new,self.m_embs.weight)
        # masks means all the historical drugs, by doing such, we get the new embedding table
        # [batch, m_voc_size,emb_dim]
        masks=(meds.sum(dim=1)>0).float().unsqueeze(2).repeat(1,1,self.emb_dim)
        old_m_embs=self.m_embs.weight.unsqueeze(0).repeat(batch,1,1)*masks
        new_m_embs=self.m_embs.weight.unsqueeze(0).repeat(batch,1,1)*(1-masks)
        # get the drug embeddings by PiecewiseTSL
        # [batch, 2k, emb_dim]
        h_old=self.tsl_old(e_old).view(batch,-1)
        h_new=self.tsl_new(e_new).view(batch,-1)
        # calculate similarity
        # [batch, m_voc_size]
        scores_old=torch.cosine_similarity(h_old.unsqueeze(1),self.lin_expand(old_m_embs),dim=2)
        scores_new=torch.cosine_similarity(h_new.unsqueeze(1),self.lin_expand(new_m_embs),dim=2)
        total_scores=scores_old+scores_new
        # scores + embedding = final medication embeddings
        # [batch, m_voc_size, emb_dim]
        final_m_embs=self.m_embs.weight.unsqueeze(0).repeat(batch,1,1)*total_scores.unsqueeze(2)
        return final_m_embs
#%%
# test
if __name__ == '__main__':
    emb_dim=32
    k=3
    m_voc_size=24
    batch=16
    seq_len=5
    model=MedRepLearn(emb_dim,k,m_voc_size).to('cuda:0')
    meds=(torch.randn((batch,seq_len,m_voc_size),device='cuda:0')>0.9).float()
    y=model(meds)
    print(y.shape) # (16, 24, 32)
#%%

class MyNet(nn.Module):
    def __init__(self, emb_dim, voc_size,k,ehr_adj,ddi_adj):
        super().__init__()
        self.d_voc_size=voc_size[0]
        self.p_voc_size=voc_size[1]
        self.m_voc_size=voc_size[2]
        self.emb_dim=emb_dim
        self.k=k
        self.medrep=MedRepLearn(emb_dim,k,self.m_voc_size)
        self.patrep=PatientRepLearn(emb_dim,k,self.d_voc_size,self.p_voc_size,self.m_voc_size)
        self.lin1=nn.Linear(2*k*emb_dim,self.m_voc_size)
        self.lin2=nn.Linear(2*k*emb_dim,2*k*emb_dim)
        self.norm=torch.nn.LayerNorm(2*k*emb_dim)
        self.gcn=GCN(self.m_voc_size, emb_dim, ehr_adj, ddi_adj)
        self.sm=SimpleNetV(emb_dim,k,voc_size)
        self.lin_med_expand=nn.Linear(emb_dim,2*k*emb_dim)
        self.w=0.7

    def forward(self,diags,procs,meds):
        # meds=torch.zeros_like(meds).to(meds.device)
        # diags=torch.zeros_like(diags).to(diags.device)
        procs=torch.zeros_like(procs).to(procs.device)

        # [batch, 2k*emb_dim]
        batch=diags.shape[0]
        pat=self.patrep(diags,procs,meds)
        h_patient=pat[:,2*self.k:,:].view(batch,-1)
        q=h_patient+self.lin2(self.norm(h_patient))
        # [batch, m_voc_size]
        # [batch, m_voc_size, emb_dim]
        ehr_meds,ddi_meds=self.gcn()
        h_meds=self.medrep(meds)
        h_meds += ehr_meds.unsqueeze(0).repeat(diags.shape[0],1,1) * 1
        h_meds += ddi_meds.unsqueeze(0).repeat(diags.shape[0],1,1) * 0.5
        h_meds=self.lin_med_expand(h_meds)
        # [batch, m_voc_size]
        o_1=self.lin1(pat[:,:2*self.k,:].view(batch,-1))
        o_2=torch.cosine_similarity(q.unsqueeze(1).repeat(1,self.m_voc_size,1),h_meds,dim=2)
        return o_1 * self.w + o_2 * (1 - self.w)

# %%
# test
if __name__ == "__main__":
    emb_dim=32
    k=3
    voc_size=(10,20,24)
    batch=16
    seq_len=5
    ehr_adj=(torch.rand((voc_size[2],voc_size[2]))>0.7).float().to('cuda:0')
    ddi_adj=(torch.rand((voc_size[2],voc_size[2]))>0.7).float().to('cuda:0')
    model=MyNet(emb_dim,voc_size,k,ehr_adj,ddi_adj).to('cuda:0')
    diags=(torch.randn((batch,seq_len,voc_size[0]),device='cuda:0')>0.9).float()
    procs=(torch.randn((batch,seq_len,voc_size[1]),device='cuda:0')>0.9).float()
    meds=(torch.randn((batch,seq_len,voc_size[2]),device='cuda:0')>0.9).float()
    y=model(diags,procs,meds)
    print(y.shape) # (16, 24)
#%%
