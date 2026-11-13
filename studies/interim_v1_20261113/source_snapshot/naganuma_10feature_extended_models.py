"""Compact residual-regression adapters of five time-series architectures.

These are local implementations of the defining operations, NOT reproductions
of the authors' long-horizon benchmarks. See architecture_manifest.json.
All encoders see only the current hour and its six preceding hourly slots.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

KINDS = ['LSTM', 'ModernTCN', 'PatchTST', 'TimesNet', 'iTransformer']
SPECIFICATIONS = {
 'LSTM': dict(hidden_size=32, layers=2, dropout=.05, readout='last hidden, 32 to 1'),
 'ModernTCN': dict(patch_length=3, patch_stride=1, dimensions_per_variable=4,
                   blocks=2, large_kernel=5, small_kernel=3, ffn_ratio=2,
                   operations='depthwise large/small kernel + per-variable FFN + cross-variable FFN'),
 'PatchTST': dict(patch_length=3, patch_stride=2, patches=3, d_model=16,
                  heads=2, layers=2, d_ff=32, channel_independent_encoder=True,
                  readout='flatten patches within variable, concatenate variables to scalar'),
 'TimesNet': dict(d_model=16, d_ff=32, blocks=2, top_k=2, inception_kernels=[1,3,5],
                  period_selection='per sample (intentional batch-independent adaptation)',
                  periods_available=[7,3,2]),
 'iTransformer': dict(d_model=32, heads=4, layers=2, d_ff=64,
                      tokens='35 feature histories plus availability history',
                      readout='shared token projection, concatenate tokens to scalar'),
}


def clean_input(x):
    # Ignore the numerical contents of unavailable temporal slots in every model.
    return torch.cat([x[:, :, :-1] * x[:, :, -1:], x[:, :, -1:]], dim=-1)


class ResidualLSTM(nn.Module):
    def __init__(self, n_features=35, **unused):
        super().__init__()
        self.lstm = nn.LSTM(n_features+1,32,2,dropout=.05,batch_first=True)
        self.head = nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):
        state,_=self.lstm(clean_input(x))
        return self.head(state[:,-1]).squeeze(-1)


class ModernBlock(nn.Module):
    def __init__(self,variables,dim=4):
        super().__init__()
        channels=variables*dim
        self.variables,self.dim=variables,dim
        self.large=nn.Sequential(nn.Conv1d(channels,channels,5,padding=2,groups=channels,bias=False),nn.BatchNorm1d(channels))
        self.small=nn.Sequential(nn.Conv1d(channels,channels,3,padding=1,groups=channels,bias=False),nn.BatchNorm1d(channels))
        self.norm=nn.BatchNorm1d(dim)
        self.variable_ff=nn.Sequential(nn.Conv1d(channels,channels*2,1,groups=variables),nn.GELU(),nn.Dropout(.05),
                                      nn.Conv1d(channels*2,channels,1,groups=variables),nn.Dropout(.05))
        self.cross_ff=nn.Sequential(nn.Conv1d(channels,channels*2,1,groups=dim),nn.GELU(),nn.Dropout(.05),
                                   nn.Conv1d(channels*2,channels,1,groups=dim),nn.Dropout(.05))
    def forward(self,x):
        batch,variables,dim,length=x.shape
        y=x.reshape(batch,variables*dim,length)
        y=self.large(y)+self.small(y)
        y=self.norm(y.reshape(batch*variables,dim,length)).reshape(batch,variables*dim,length)
        y=self.variable_ff(y).reshape(batch,variables,dim,length)
        y=y.permute(0,2,1,3).reshape(batch,dim*variables,length)
        y=self.cross_ff(y).reshape(batch,dim,variables,length).permute(0,2,1,3)
        return x+y


class ResidualModernTCN(nn.Module):
    def __init__(self,n_features=35,sequence_length=7,**unused):
        super().__init__()
        variables=n_features+1
        self.stem=nn.Sequential(nn.Conv1d(1,4,3,stride=1),nn.BatchNorm1d(4))
        self.blocks=nn.Sequential(ModernBlock(variables),ModernBlock(variables))
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.05),nn.Linear(variables*4*(sequence_length-2),1))
    def forward(self,x):
        x=clean_input(x).transpose(1,2)
        b,m,t=x.shape
        z=self.stem(x.reshape(b*m,1,t)).reshape(b,m,4,t-2)
        return self.head(self.blocks(z)).squeeze(-1)


class AttentionBlock(nn.Module):
    def __init__(self,dim,heads,ff):
        super().__init__()
        self.attention=nn.MultiheadAttention(dim,heads,dropout=.05,batch_first=True)
        self.norm1,self.norm2=nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,ff),nn.GELU(),nn.Dropout(.05),nn.Linear(ff,dim),nn.Dropout(.05))
        self.capture_attention=False
        self.last_attention=None
    def forward(self,x):
        y,a=self.attention(x,x,x,need_weights=self.capture_attention,average_attn_weights=False)
        x=self.norm1(x+y)
        if self.capture_attention:
            self.last_attention=a.detach()
        return self.norm2(x+self.ff(x))


class ResidualPatchTST(nn.Module):
    def __init__(self,n_features=35,sequence_length=7,**unused):
        super().__init__()
        self.variables=n_features+1
        self.embedding=nn.Linear(3,16)
        self.position=nn.Parameter(torch.randn(1,3,16)*.02)
        self.blocks=nn.Sequential(AttentionBlock(16,2,32),AttentionBlock(16,2,32))
        self.per_variable=nn.Linear(3*16,1)
        self.head=nn.Sequential(nn.Linear(self.variables,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):
        patches=clean_input(x).transpose(1,2).unfold(-1,3,2)
        batch,variables,count,width=patches.shape
        z=self.embedding(patches.reshape(batch*variables,count,width))+self.position
        z=self.blocks(z).reshape(batch,variables,-1)
        return self.head(self.per_variable(z).squeeze(-1)).squeeze(-1)


class Inception(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.kernels=nn.ModuleList([nn.Conv2d(inputs,outputs,k,padding=k//2) for k in (1,3,5)])
        for conv in self.kernels:
            nn.init.kaiming_normal_(conv.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x):
        return torch.stack([conv(x) for conv in self.kernels],dim=-1).mean(-1)


class TimesBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(Inception(16,32),nn.GELU(),Inception(32,16))
        self.last_periods=None
        self.last_weights=None
        self.capture_periods=False
    def forward(self,x):
        batch,length,channels=x.shape
        amplitude=torch.fft.rfft(x,dim=1).abs().mean(-1)
        # Original batch-mean top-k makes a sample depend on its batch mates.
        # Per-sample top-k preserves independent deployment at each hourly step.
        frequencies=torch.topk(amplitude[:,1:],k=2,dim=-1).indices+1
        periods=torch.div(length,frequencies,rounding_mode='floor')
        weights=torch.softmax(torch.gather(amplitude,1,frequencies),dim=-1)
        outputs=[]
        for rank in range(2):
            accumulated=torch.zeros_like(x)
            for period_tensor in torch.unique(periods[:,rank]):
                period=int(period_tensor.item())
                indices=torch.where(periods[:,rank]==period)[0]
                part=x.index_select(0,indices)
                full=math.ceil(length/period)*period
                part=F.pad(part,(0,0,0,full-length))
                part=part.reshape(len(indices),full//period,period,channels).permute(0,3,1,2)
                part=self.conv(part).permute(0,2,3,1).reshape(len(indices),full,channels)[:,:length]
                accumulated=accumulated.index_copy(0,indices,part)
            outputs.append(accumulated)
        if self.capture_periods:
            self.last_periods=periods.detach()
            self.last_weights=weights.detach()
        return x+(torch.stack(outputs,dim=-1)*weights[:,None,None,:]).sum(-1)


class ResidualTimesNet(nn.Module):
    def __init__(self,n_features=35,sequence_length=7,**unused):
        super().__init__()
        self.embedding=nn.Conv1d(n_features+1,16,3,padding=1,padding_mode='circular',bias=False)
        position=torch.arange(sequence_length)[:,None]
        div=torch.exp(torch.arange(0,16,2)*(-math.log(10000.)/16))
        positional=torch.zeros(sequence_length,16)
        positional[:,0::2]=torch.sin(position*div)
        positional[:,1::2]=torch.cos(position*div)
        self.register_buffer('position',positional)
        self.blocks=nn.ModuleList([TimesBlock(),TimesBlock()])
        self.norms=nn.ModuleList([nn.LayerNorm(16),nn.LayerNorm(16)])
        self.drop=nn.Dropout(.05)
        self.head=nn.Linear(sequence_length*16,1)
    def forward(self,x):
        valid=x[:,:,-1:]
        z=self.drop(self.embedding(clean_input(x).transpose(1,2)).transpose(1,2)+self.position)
        for block,norm in zip(self.blocks,self.norms):
            z=norm(block(z))
        z=self.drop(F.gelu(z))*valid
        return self.head(z.flatten(1)).squeeze(-1)


class ResidualiTransformer(nn.Module):
    def __init__(self,n_features=35,sequence_length=7,**unused):
        super().__init__()
        self.embedding=nn.Linear(sequence_length,32)
        self.drop=nn.Dropout(.05)
        self.blocks=nn.Sequential(AttentionBlock(32,4,64),AttentionBlock(32,4,64))
        self.norm=nn.LayerNorm(32)
        self.project=nn.Linear(32,1)
        self.head=nn.Sequential(nn.Linear(n_features+1,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):
        z=self.drop(self.embedding(clean_input(x).transpose(1,2)))
        z=self.norm(self.blocks(z))
        return self.head(self.project(z).squeeze(-1)).squeeze(-1)


def create_model(kind,n_features=35,n_physical=10,sequence_length=7):
    cls={'LSTM':ResidualLSTM,'ModernTCN':ResidualModernTCN,'PatchTST':ResidualPatchTST,
         'TimesNet':ResidualTimesNet,'iTransformer':ResidualiTransformer}[kind]
    return cls(n_features=n_features,sequence_length=sequence_length)
