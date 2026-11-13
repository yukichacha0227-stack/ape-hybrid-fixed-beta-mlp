"""Residual MLP, causal TCN and feature/temporal self-attention models."""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualMLP(nn.Module):
    def __init__(self, n_features, n_physical=10, **unused):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_features, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.05),
                                     nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.05),
                                     nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, values):
        return self.network(values[:, -1, :-1]).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=2, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=2, dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(.05)

    def forward(self, values):
        x = self.conv1(nn.functional.pad(values, (self.dilation, 0)))
        x = self.dropout(nn.functional.gelu(self.norm1(x.transpose(1, 2)))).transpose(1, 2)
        x = self.conv2(nn.functional.pad(x, (self.dilation, 0)))
        x = self.dropout(nn.functional.gelu(self.norm2(x.transpose(1, 2)))).transpose(1, 2)
        return nn.functional.gelu(x + self.skip(values))


class ResidualTCN(nn.Module):
    def __init__(self, n_features, n_physical=10, **unused):
        super().__init__()
        self.blocks = nn.Sequential(CausalBlock(n_features+1, 16, 1), CausalBlock(16, 16, 2), CausalBlock(16, 16, 4))
        self.head = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, values):
        return self.head(self.blocks(values.transpose(1, 2))[:, :, -1]).squeeze(-1)


class ResidualSelfAttention(nn.Module):
    """Current-hour feature tokens and masked causal temporal attention.

    Past physical covariates are projected linearly; the target hour uses the
    feature-token attention encoder. This keeps CPU training tractable without
    reducing the prescribed ten inputs or seven exact hourly context slots.
    """
    def __init__(self, n_features, n_physical=10, sequence_length=7, **unused):
        super().__init__()
        self.n_physical = n_physical
        self.feature_scale = nn.Parameter(torch.empty(n_physical, 16))
        self.feature_bias = nn.Parameter(torch.empty(n_physical, 16))
        nn.init.normal_(self.feature_scale, std=.1)
        nn.init.normal_(self.feature_bias, std=.1)
        self.feature_attention = nn.MultiheadAttention(16, 2, dropout=.05, batch_first=True)
        self.feature_norm = nn.LayerNorm(16)
        self.feature_ff = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        self.feature_ff_norm = nn.LayerNorm(16)
        self.past_physical = nn.Linear(n_physical, 16)
        self.surface_context = nn.Linear(1, 16, bias=False)
        self.context = nn.Linear(n_features-n_physical, 16)
        self.position = nn.Parameter(torch.zeros(sequence_length, 32))
        nn.init.normal_(self.position, std=.02)
        self.temporal_attention = nn.MultiheadAttention(32, 4, dropout=.05, batch_first=True)
        self.temporal_norm = nn.LayerNorm(32)
        self.temporal_ff = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Dropout(.05), nn.Linear(64, 32))
        self.output_norm = nn.LayerNorm(32)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
        self.capture_attention = False
        self.last_feature_attention = None
        self.last_temporal_attention = None

    def forward(self, values):
        batch, length, _ = values.shape
        valid = values[:, :, -1] > .5
        physical = values[:, :, :self.n_physical]
        tokens = physical[:, -1].unsqueeze(-1) * self.feature_scale + self.feature_bias
        # In static_features, the first auxiliary column is the standardized
        # HSR/TSR indicator. Conditioning before attention allows feature maps
        # to differ by surface, despite identical paired atmospheric inputs.
        surface = values[:, -1, self.n_physical:self.n_physical+1]
        tokens = tokens + self.surface_context(surface).unsqueeze(1)
        attended, weights = self.feature_attention(tokens, tokens, tokens, need_weights=self.capture_attention,
                                                   average_attn_weights=False)
        tokens = self.feature_norm(tokens + attended)
        tokens = self.feature_ff_norm(tokens + self.feature_ff(tokens))
        current = tokens.mean(dim=1).unsqueeze(1)
        past = nn.functional.gelu(self.past_physical(physical[:, :-1]))
        feature_state = torch.cat([past, current], dim=1)
        context = nn.functional.gelu(self.context(values[:, :, self.n_physical:-1]))
        state = torch.cat([feature_state, context], dim=-1) + self.position[:length]
        state = state * valid.unsqueeze(-1)
        allowed = torch.ones(length, length, dtype=torch.bool, device=values.device).tril()[None]
        allowed = allowed & valid[:, None, :]
        # Padding queries attend only to themselves: avoids an all-masked softmax.
        diagonal = torch.eye(length, dtype=torch.bool, device=values.device)[None]
        allowed = torch.where(valid[:, :, None], allowed, diagonal)
        mask = (~allowed).repeat_interleave(4, dim=0)
        attended, temporal = self.temporal_attention(state, state, state, attn_mask=mask,
                                                     need_weights=self.capture_attention, average_attn_weights=False)
        state = self.temporal_norm(state + attended)
        state = self.output_norm(state + self.temporal_ff(state))
        if self.capture_attention:
            self.last_feature_attention = weights.detach()
            self.last_temporal_attention = temporal[:, :, -1, :].detach()
        return self.head(state[:, -1]).squeeze(-1)


def create_model(kind, n_features, n_physical=10, sequence_length=7):
    cls = {'MLP': ResidualMLP, 'TCN': ResidualTCN, 'SelfAttention': ResidualSelfAttention}[kind]
    return cls(n_features=n_features, n_physical=n_physical, sequence_length=sequence_length)


def predict(model, values, batch_size=512):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            outputs.append(model(torch.as_tensor(values[start:start+batch_size], dtype=torch.float32)).numpy())
    return np.concatenate(outputs).astype(float)


def train(model, x_train, y_train, x_early, y_early, settings, name):
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train.astype('float32')), torch.from_numpy(y_train.astype('float32'))),
                        batch_size=settings.batch_size, shuffle=True, generator=torch.Generator().manual_seed(settings.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings.scheduler_gamma)
    loss_fn = nn.HuberLoss(delta=1.)
    best_loss, best_epoch, best_state, stale = float('inf'), 0, None, 0
    rows = []
    start_time = time.perf_counter()
    for epoch in range(1, settings.max_epochs+1):
        model.train()
        total, count, max_grad = 0., 0, 0.
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {name}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            max_grad = max(max_grad, float(norm))
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        val_prediction = predict(model, x_early)
        delta = np.abs(val_prediction-y_early)
        val_loss = float(np.where(delta<=1., .5*delta**2, delta-.5).mean())
        rows.append(dict(epoch=epoch, train_huber_standardized=total/count,
                         validation_early_huber_standardized=val_loss,
                         learning_rate=optimizer.param_groups[0]['lr'], max_gradient_norm_before_clip=max_grad,
                         elapsed_seconds=time.perf_counter()-start_time))
        if val_loss < best_loss-settings.min_delta:
            best_loss, best_epoch, best_state, stale = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        scheduler.step()
        for group in optimizer.param_groups:
            group['lr'] = max(group['lr'], settings.min_learning_rate)
        if epoch == 1 or epoch % 25 == 0:
            print(f'[{name}] epoch={epoch} train={total/count:.6f} val={val_loss:.6f} best={best_epoch} elapsed={time.perf_counter()-start_time:.0f}s', flush=True)
        if stale >= settings.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return pd.DataFrame(rows), dict(best_epoch=best_epoch, best_validation_loss=best_loss,
                                    epochs_executed=len(rows), train_seconds=time.perf_counter()-start_time,
                                    parameters=sum(p.numel() for p in model.parameters()),
                                    torch_cpu_threads=torch.get_num_threads())
