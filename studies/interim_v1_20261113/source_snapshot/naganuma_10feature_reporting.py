"""Scientific diagnostics, attribution maps and Japanese experiment report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch

import naganuma_10feature_data as D
import naganuma_10feature_models as M

PHYSICAL = D.PHYSICAL
RAW_COLUMNS = {'AM':'am', 'omega':'omega', 'AOD550':'aod550', 'TOTANGSTR':'totangstr', 'TO3':'to3',
               **{c:c for c in D.EXTRAS}}
LABELS = ['AM', 'TQV', 'AOD', 'Ångström', 'TO3', 'GHI', 'Kt', 'Diffuse fraction', 'Temperature', 'RH']


def configure_plotting():
    for path in ['C:/Windows/Fonts/meiryo.ttc', 'C:/Windows/Fonts/YuGothM.ttc', 'C:/Windows/Fonts/msgothic.ttc']:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({'axes.unicode_minus':False, 'font.size':10, 'figure.dpi':110, 'savefig.dpi':180})


def save_figure(fig, path):
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(Path(path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def md_table(df, digits=4):
    return D.N.md_table(df, digits=digits)


def input_plots(output,datasets,roles,beta_model,beta_scaler):
    import naganuma_10feature_experiment as E
    train=roles['train_oof']
    a=datasets['PUBLIC_POWER']
    train=train[a.iloc[train].surface.eq('HSR').to_numpy()]
    fig,axes=plt.subplots(2,3,figsize=(12,7.5))
    for ax,feature in zip(axes.flat,D.EXTRAS):
        observed=datasets['NEDO_reference'].iloc[train][feature].to_numpy()
        public=a.iloc[train][feature].to_numpy()
        ax.hexbin(observed,public,gridsize=40,mincnt=1,cmap='viridis')
        lo,hi=min(observed.min(),public.min()),max(observed.max(),public.max())
        ax.plot([lo,hi],[lo,hi],'r--',lw=.8)
        unit={'GHI':'W/m²','Kt':'dimensionless','diffuse_fraction':'dimensionless','temperature_C':'°C','RH_pct':'%'}[feature]
        ax.set(xlabel=f'NEDO hourly reference ({unit})',ylabel=f'Public hourly input ({unit})',title=f'{feature}: Train r={np.corrcoef(observed,public)[0,1]:.3f}')
    axes.flat[-1].axis('off')
    fig.suptitle('Input-source correspondence / Train only / same timestamps')
    fig.tight_layout()
    save_figure(fig,output/'input_source_comparison_train.png')
    for source,data in datasets.items():
        beta=beta_model.predict(beta_scaler.transform(D.N.F.beta_design(data).to_numpy(float)))
        physical=E.static_features(data,beta,10)[PHYSICAL].iloc[train]
        correlation=physical.corr(method='spearman')
        correlation.to_csv(output/(source+'_physical_feature_correlation_train.csv'),encoding='utf-8-sig')
        fig,ax=plt.subplots(figsize=(8,7))
        image=ax.imshow(correlation.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(10),LABELS,rotation=55,ha='right',fontsize=8)
        ax.set_yticks(range(10),LABELS,fontsize=8)
        ax.set_title(source+' / Train Spearman correlation')
        fig.colorbar(image,ax=ax)
        save_figure(fig,output/(source+'_physical_feature_correlation_train.png'))


def basic_plots(output, data, roles, predictions, runs, metrics):
    idx = roles['test']
    y = data['ape'].to_numpy()
    for name, run in runs.items():
        folder = run['folder']
        history = pd.read_csv(folder/'training_history.csv')
        fig, ax = plt.subplots(figsize=(8,4))
        for column, label in [('train_huber_standardized','Train (OOF residual targets)'),
                              ('validation_early_huber_standardized','Early-stopping validation')]:
            ax.plot(history.epoch, history[column], label=label)
        ax.axvline(run['report']['training']['best_epoch'], color='grey', linestyle='--', label='Selected epoch')
        ax.set(xlabel='Epoch', ylabel='Standardized Huber loss', title=name)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        save_figure(fig, folder/'learning_history.png')
        fig, axes = plt.subplots(1,2,figsize=(10,4.5))
        for ax, surface in zip(axes,['HSR','TSR']):
            si = idx[data.iloc[idx].surface.eq(surface).to_numpy()]
            p = predictions[name+'_cal'][si]
            m = metrics.loc[metrics.split.eq('test') & metrics.surface.eq(surface) & metrics.model.eq(name+'_cal')].iloc[0]
            ax.hexbin(y[si], p, gridsize=40, mincnt=1, cmap='viridis')
            lim = [min(y[si].min(),p.min())-.005,max(y[si].max(),p.max())+.005]
            ax.plot(lim,lim,'r--',lw=1)
            ax.set(xlabel='Measured hourly APE (eV)',ylabel='Predicted APE (eV)',title=f'{surface}: R²={m.r2:.3f}, RMSE={m.rmse_meV:.2f} meV',xlim=lim,ylim=lim)
            ax.set_aspect('equal', adjustable='box')
        fig.suptitle(name+' / chronological Test')
        save_figure(fig,folder/'measured_vs_predicted.png')
    chosen = ['fixed_beta_5','CONTROL_5_MLP_cal'] + [s+'_'+k+'_cal' for s in ['PUBLIC_POWER','NEDO_reference'] for k in ['MLP','TCN','SelfAttention','HGB']]
    fig,axes = plt.subplots(1,2,figsize=(13,6))
    for ax,surface in zip(axes,['HSR','TSR']):
        part = metrics.loc[metrics.split.eq('test') & metrics.surface.eq(surface)].set_index('model').reindex(chosen)
        ax.barh(np.arange(len(part)),part.rmse_meV,color=['#777777','#aaaaaa']+['#2878a4']*4+['#dd8452']*4)
        ax.set_yticks(np.arange(len(part)),[v.replace('_cal','') for v in part.index],fontsize=8)
        ax.invert_yaxis()
        ax.set(xlabel='Test RMSE (meV), lower is better',title=surface)
        ax.grid(axis='x',alpha=.2)
    fig.tight_layout()
    save_figure(fig,output/'test_model_comparison.png')
    # Raw/calibrated changes are important for this dataset's calibration drift.
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    names=[s+'_'+k for s in ['PUBLIC_POWER','NEDO_reference'] for k in ['MLP','TCN','SelfAttention']]
    for ax,surface in zip(axes,['HSR','TSR']):
        p=metrics.loc[metrics.split.eq('test') & metrics.surface.eq(surface)].set_index('model')
        x=np.arange(len(names))
        ax.bar(x-.18,[p.loc[n+'_raw','rmse_meV'] for n in names],.36,label='Before calibration')
        ax.bar(x+.18,[p.loc[n+'_cal','rmse_meV'] for n in names],.36,label='After calibration')
        ax.set_xticks(x,[n.replace('PUBLIC_POWER','Public').replace('NEDO_reference','NEDO') for n in names],rotation=35,ha='right',fontsize=8)
        ax.set(ylabel='Test RMSE (meV)',title=surface)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig,output/'calibration_effect.png')


def selected_rows(data, test, per_surface):
    parts=[]
    for surface in ['HSR','TSR']:
        indices=test[data.iloc[test].surface.eq(surface).to_numpy()]
        positions=np.unique(np.linspace(0,len(indices)-1,min(len(indices),per_surface)).astype(int))
        parts.extend(indices[positions])
    return np.asarray(parts,dtype=int)


def apply_calibration(raw, beta, data, idx, calibration):
    result=raw.copy()
    for surface in ['HSR','TSR']:
        mask=data.iloc[idx].surface.eq(surface).to_numpy()
        c=calibration[surface]
        result[mask]=np.clip(c['slope']*raw[mask]+c['intercept'],-c['clip_abs'],c['clip_abs'])
    return beta+result


def permutation_importance(run, roles, lookup, beta_model, beta_scaler):
    import naganuma_10feature_experiment as E
    folder,data=run['folder'],run['data']
    destination=folder/'feature_permutation_importance.csv'
    if destination.exists():
        return pd.read_csv(destination)
    test=roles['test']
    idx=selected_rows(data,test,512)
    nphys=run['config']['n_physical']
    groups={p:[p] for p in PHYSICAL[:nphys]}
    if nphys==10:
        groups.update({'GROUP_atmosphere':PHYSICAL[:5], 'GROUP_radiation':PHYSICAL[5:8], 'GROUP_local_weather':PHYSICAL[8:]})
    truth=data['ape'].to_numpy()[idx]
    original_beta=beta_model.predict(beta_scaler.transform(D.N.F.beta_design(data).to_numpy(float)))
    original_residual=M.predict(run['model'],run['inputs'][idx])*run['std']+run['mean']
    original_cal=apply_calibration(original_residual,original_beta[idx],data,idx,run['calibration'])
    original_raw=original_beta[idx]+original_residual
    rng=np.random.default_rng(20261028)
    times=pd.Index(data.iloc[test].datetime.unique())
    first=data.loc[data.surface.eq('HSR')].set_index('datetime')
    rows=[]
    static_width=run['static'].shape[1]
    test_times=data.iloc[test].datetime
    for group, members in groups.items():
        for repeat in range(3):
            donors=pd.Series(rng.permutation(times),index=times)
            changed=data.copy()
            for name in members:
                column=RAW_COLUMNS[name]
                donor_values=first[column].reindex(donors.to_numpy()).to_numpy()
                mapper=pd.Series(donor_values,index=times)
                changed.loc[test,column]=test_times.map(mapper).to_numpy()
            b=beta_model.predict(beta_scaler.transform(D.N.F.beta_design(changed).to_numpy(float)))
            static=E.static_features(changed,b,nphys)
            scaled=(static.to_numpy()-run['scaler'].mean_[:static_width])/run['scaler'].scale_[:static_width]
            x=run['inputs'][idx].copy()
            prev=lookup[idx]
            present=prev>=0
            x[:,:,:static_width][present]=scaled[prev[present]]
            residual=M.predict(run['model'],x)*run['std']+run['mean']
            cal=apply_calibration(residual,b[idx],data,idx,run['calibration'])
            raw=b[idx]+residual
            for surface in ['HSR','TSR']:
                mask=data.iloc[idx].surface.eq(surface).to_numpy()
                for stage,base,new in [('raw',original_raw,raw),('calibrated',original_cal,cal)]:
                    increase=1000*(np.abs(new[mask]-truth[mask]).mean()-np.abs(base[mask]-truth[mask]).mean())
                    rows.append(dict(feature=group,surface=surface,stage=stage,repeat=repeat,mae_increase_meV=increase,n=int(mask.sum())))
        print(f'[IMPORTANCE {run["name"]}] {group}',flush=True)
    detailed=pd.DataFrame(rows)
    detailed.to_csv(folder/'permutation_repetitions.csv',index=False,encoding='utf-8-sig')
    result=detailed.groupby(['feature','surface','stage'],as_index=False).agg(mae_increase_meV=('mae_increase_meV','mean'),repeat_sd_meV=('mae_increase_meV','std'),n=('n','first'))
    result.to_csv(destination,index=False,encoding='utf-8-sig')
    return result


def importance_plot(run, importance):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    physical=PHYSICAL[:run['config']['n_physical']]
    for ax,surface in zip(axes,['HSR','TSR']):
        for offset,stage,color in [(-.18,'raw','#2878a4'),(.18,'calibrated','#dd8452')]:
            part=importance.loc[importance.surface.eq(surface)&importance.stage.eq(stage)].set_index('feature').reindex(physical)
            ax.barh(np.arange(len(physical))+offset,part.mae_increase_meV,.36,xerr=part.repeat_sd_meV,color=color,label=stage)
        ax.axvline(0,color='grey',lw=.7)
        ax.set_yticks(np.arange(len(physical)),LABELS[:len(physical)])
        ax.invert_yaxis()
        ax.set(xlabel='Increase in MAE after permutation (meV)',title=surface)
        ax.legend(fontsize=8)
    fig.suptitle(run['name']+' / frozen-history pipeline sensitivity')
    fig.tight_layout()
    save_figure(fig,run['folder']/'feature_permutation_importance.png')


def attribution_maps(run, roles, beta_model, beta_scaler):
    if (run['folder']/'attribution_verification.json').exists() and (run['folder']/'feature_effect_attribution_map.png').exists():
        return
    model,data=run['model'],run['data']
    if hasattr(model,'capture_attention'):
        model.capture_attention=False
    idx=selected_rows(data,roles['test'],128)
    factual=torch.tensor(run['inputs'][idx],dtype=torch.float32)
    nphys=run['config']['n_physical']
    names=run['config']['feature_names']
    mu=torch.tensor(run['scaler'].mean_,dtype=torch.float32)
    scale=torch.tensor(run['scaler'].scale_,dtype=torch.float32)
    raw_factual=factual[:,:,:-1]*scale+mu
    actual=raw_factual[:,:,:nphys].detach()
    baseline=mu[:nphys].expand_as(actual).clone()
    valid=factual[:,:,-1]
    coefficients={name:float(coef/std) for name,coef,std in zip(D.N.F.beta_design(data).columns,beta_model.coef_,beta_scaler.scale_)}
    intercept=float(beta_model.intercept_-np.sum(beta_model.coef_*beta_scaler.mean_/beta_scaler.scale_))
    slopes=torch.tensor([run['calibration'][s]['slope'] for s in data.iloc[idx].surface],dtype=torch.float32)
    offsets=torch.tensor([run['calibration'][s]['intercept'] for s in data.iloc[idx].surface],dtype=torch.float32)
    clips=torch.tensor([run['calibration'][s]['clip_abs'] for s in data.iloc[idx].surface],dtype=torch.float32)

    def conditional_pipeline(physical):
        columns={name:raw_factual[:,:,j] for j,name in enumerate(names)}
        for j,name in enumerate(PHYSICAL[:nphys]):
            columns[name]=physical[:,:,j]
        for name in PHYSICAL[:5]:
            alias={'AM':'AM','omega':'omega','AOD550':'AOD550','TOTANGSTR':'TOTANGSTR','TO3':'TO3'}[name]
            columns['TSR_x_'+alias]=columns['TSR_label']*columns[name]
        beta=intercept+sum(coef*columns[name] for name,coef in coefficients.items())
        columns['beta_prediction']=beta
        columns['AM_x_AOD550']=columns['AM']*columns['AOD550']
        columns['AM_x_TOTANGSTR']=columns['AM']*columns['TOTANGSTR']
        columns['AOD550_x_TOTANGSTR']=columns['AOD550']*columns['TOTANGSTR']
        columns['AM_x_omega']=columns['AM']*columns['omega']
        columns['transmittance_proxy']=torch.exp(-torch.clamp(columns['AM_x_AOD550'],0,50))
        raw=torch.stack([columns[name] for name in names],dim=-1)
        values=torch.cat([((raw-mu)/scale)*valid.unsqueeze(-1),valid.unsqueeze(-1)],dim=-1)
        residual=model(values)*run['std']+run['mean']
        correction=torch.clamp(slopes*residual+offsets,min=-clips,max=clips)
        return beta[:,-1]+correction

    with torch.no_grad():
        actual_prediction=conditional_pipeline(actual)
        baseline_prediction=conditional_pipeline(baseline)
        base_design=D.N.F.beta_design(data).to_numpy(float)
        current_beta=beta_model.predict(beta_scaler.transform(base_design))[idx]
        r=M.predict(model,run['inputs'][idx])*run['std']+run['mean']
        direct_prediction=apply_calibration(r,current_beta,data,idx,run['calibration'])
        reconstruction_error=float(np.max(np.abs(actual_prediction.numpy()-direct_prediction)))
        assert reconstruction_error<1e-5, f'Attribution pipeline mismatch: {reconstruction_error}'
    gradient=torch.zeros_like(actual)
    for alpha in np.linspace(1/32,1,32):
        point=(baseline+float(alpha)*(actual-baseline)).detach().requires_grad_(True)
        prediction=conditional_pipeline(point)
        gradient+=torch.autograd.grad(prediction.sum(),point)[0].detach()/32
    attribution=(actual-baseline)*gradient
    ig=attribution.detach().numpy()*1000
    completeness=(attribution.sum(dim=(1,2))-(actual_prediction-baseline_prediction)).abs().numpy()*1000
    D.save_json(run['folder']/'attribution_verification.json',dict(
        pipeline_reconstruction_max_abs_eV=reconstruction_error,steps=32,
        numerical_completeness_median_meV=float(np.median(completeness)),
        numerical_completeness_max_meV=float(completeness.max()),
        interpretation='conditional final calibrated APE; fixed beta and derived paths included; predicted-residual history frozen'))
    np.savez_compressed(run['folder']/'feature_attribution_values.npz',indices=idx,integrated_gradients_meV=ig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    rows=[]
    for ax,surface in zip(axes,['HSR','TSR']):
        mask=data.iloc[idx].surface.eq(surface).to_numpy()
        values=np.abs(ig[mask]).mean(axis=0)
        if run['kind']=='MLP':
            values=values[-1:]
        image=ax.imshow(values,aspect='auto',cmap='magma',vmin=0)
        ax.set_xticks(np.arange(nphys),LABELS[:nphys],rotation=55,ha='right',fontsize=8)
        lag_labels=['0'] if run['kind']=='MLP' else [str(v) for v in range(-6,1)]
        ax.set_yticks(np.arange(len(values)),lag_labels)
        ax.set(ylabel='Elapsed hours relative to target',title=surface)
        fig.colorbar(image,ax=ax,label='Mean |conditional final-APE IG| (meV)')
        for t,row in enumerate(values):
            for j,value in enumerate(row):
                rows.append(dict(surface=surface,lag_hours=int(lag_labels[t]),feature=PHYSICAL[j],absolute_IG_meV=float(value)))
    fig.suptitle(run['name']+' / attribution map (not attention weights)')
    fig.tight_layout()
    save_figure(fig,run['folder']/'feature_effect_attribution_map.png')
    pd.DataFrame(rows).to_csv(run['folder']/'feature_effect_attribution_map.csv',index=False,encoding='utf-8-sig')


def attention_maps(run, roles):
    if (run['folder']/'attention_weights_test.npz').exists() and (run['folder']/'temporal_attention_map.png').exists():
        return
    model=run['model']
    model.eval()
    model.capture_attention=True
    indices=roles['test']
    fs,ts=[],[]
    with torch.no_grad():
        for start in range(0,len(indices),256):
            idx=indices[start:start+256]
            model(torch.from_numpy(run['inputs'][idx]))
            fs.append(model.last_feature_attention.numpy())
            ts.append(model.last_temporal_attention.numpy())
    feature,temporal=np.concatenate(fs),np.concatenate(ts)
    assert np.isfinite(feature).all() and np.isfinite(temporal).all()
    assert np.allclose(feature.sum(-1),1,atol=1e-5)
    assert np.allclose(temporal.sum(-1),1,atol=1e-5)
    present=run['inputs'][indices,:,-1]>0
    assert np.max(np.abs(temporal*~present[:,None,:]))<1e-6
    np.savez_compressed(run['folder']/'attention_weights_test.npz',indices=indices,feature_attention=feature,
                        temporal_current_query_attention=temporal,past_covariate_available=present)
    feature_rows=[]
    temporal_rows=[]
    for surface in ['HSR','TSR']:
        mask=run['data'].iloc[indices].surface.eq(surface).to_numpy()
        mean=feature[mask].mean(axis=(0,1))
        fig,ax=plt.subplots(figsize=(8,6.8))
        image=ax.imshow(mean,cmap='viridis',vmin=0)
        ax.set_xticks(range(10),LABELS,rotation=50,ha='right',fontsize=9)
        ax.set_yticks(range(10),LABELS,fontsize=9)
        ax.set(xlabel='Key feature (receives attention)',ylabel='Query feature',title=run['name']+' / '+surface+' / Test mean')
        fig.colorbar(image,ax=ax,label='Attention weight (mean of heads and Test rows)')
        save_figure(fig,run['folder']/f'feature_attention_map_{surface}.png')
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for h,ax in enumerate(axes):
            image=ax.imshow(feature[mask,h].mean(axis=0),cmap='viridis',vmin=0)
            ax.set_xticks(range(10),LABELS,rotation=65,ha='right',fontsize=7)
            ax.set_yticks(range(10),LABELS,fontsize=7)
            ax.set_title(f'{surface} feature attention head {h+1}')
            fig.colorbar(image,ax=ax)
        fig.tight_layout()
        save_figure(fig,run['folder']/f'feature_attention_heads_{surface}.png')
        for q in range(10):
            for k in range(10):
                feature_rows.append(dict(surface=surface,query_feature=PHYSICAL[q],key_feature=PHYSICAL[k],mean_attention=float(mean[q,k])))
        for head in range(4):
            for j in range(7):
                temporal_rows.append(dict(surface=surface,head=head+1,lag_hours=j-6,mean_attention=float(temporal[mask,head,j].mean()),availability=float(present[mask,j].mean())))
    temporal_frame=pd.DataFrame(temporal_rows)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    for ax,surface in zip(axes,['HSR','TSR']):
        frame=temporal_frame.loc[temporal_frame.surface.eq(surface)].pivot(index='head',columns='lag_hours',values='mean_attention')
        image=ax.imshow(frame.to_numpy(),aspect='auto',cmap='viridis',vmin=0)
        ax.set_xticks(range(7),range(-6,1))
        ax.set_yticks(range(4),range(1,5))
        ax.set(xlabel='Elapsed hours relative to target',ylabel='Temporal head',title=surface+' / current-query attention')
        fig.colorbar(image,ax=ax)
    fig.tight_layout()
    save_figure(fig,run['folder']/'temporal_attention_map.png')
    pd.DataFrame(feature_rows).to_csv(run['folder']/'feature_attention_mean.csv',index=False,encoding='utf-8-sig')
    temporal_frame.to_csv(run['folder']/'temporal_attention_mean.csv',index=False,encoding='utf-8-sig')
    model.capture_attention=False


def model_report(run,metrics,importance):
    name=run['name']
    selected=metrics.loc[metrics.split.eq('test')&metrics.surface.ne('ALL')&metrics.model.isin([name+'_raw',name+'_cal'])]
    interpretation=[]
    if len(importance):
        for surface in ['HSR','TSR']:
            part=importance.loc[importance.surface.eq(surface)&importance.stage.eq('calibrated')&~importance.feature.str.startswith('GROUP_')]
            strongest=part.sort_values('mae_increase_meV',ascending=False).head(3)
            interpretation.append(f"- {surface}の校正後Permutation MAE増分上位："+'、'.join(f'{r.feature} {r.mae_increase_meV:.3f} meV' for r in strongest.itertuples()))
    maps=''
    if run['kind']=='SelfAttention':
        maps='\n![HSR特徴量間Attention](feature_attention_map_HSR.png)\n\n![TSR特徴量間Attention](feature_attention_map_TSR.png)\n\n![時間Attention](temporal_attention_map.png)\n'
    text=f'''# {name}：結果・考察

長沼の1時間平均APEを、固定β＋{run['kind']}残差＋表面別校正で予測した。βは大気5変数で共通学習、残差モデルの基礎大気・気象入力は10種類。取得元はモデル名で区別する。

## Test結果

{md_table(selected[['surface','model','n','r2','rmse_meV','mae_meV','mape_pct','mbe_truth_minus_prediction_meV']])}

![正解値と予測値](measured_vs_predicted.png)

![学習履歴](learning_history.png)

早期停止の最良epochは{run['report']['training']['best_epoch']}。Testを学習・設定選択・校正に使っていない。rawとcalの差は校正による変化であり、校正が全ての条件で有効とは限らない。

## 特徴量の効き目

{chr(10).join(interpretation)}

![Permutation importance](feature_permutation_importance.png)

![特徴量×時間の条件付きIG](feature_effect_attribution_map.png)

Permutationは大気の派生項・βも再計算し、予測残差履歴は固定した感度。IGは固定βと残差経路を含む最終APEへの寄与を、過去の予測残差履歴を固定して計算した。いずれも因果効果を示さない。TCNとMLPに固有のAttention重みはないため、IG・Permutation図をAttention図とは呼ばない。
{maps}

比較条件、公開入力とNEDO参照入力との差、校正の注意点、全モデルの比較は [共通の結果・考察](../結果・考察.md) に記載。現在は長沼内の時間外評価であり、未知地点への移植精度の実証ではない。
'''
    (run['folder']/'結果・考察.md').write_text(text,encoding='utf-8')


def paired_block_bootstrap(output,data,roles,predictions):
    comparisons=[]
    for source in ['PUBLIC_POWER','NEDO_reference']:
        for kind in ['MLP','TCN','SelfAttention']:
            comparisons.append((source+'_'+kind+'_cal','CONTROL_5_MLP_cal'))
            comparisons.append((source+'_'+kind+'_cal','fixed_beta_5'))
        comparisons.extend((source+'_'+kind+'_cal',source+'_MLP_cal') for kind in ['TCN','SelfAttention'])
    comparisons.extend(('PUBLIC_POWER_'+kind+'_cal','NEDO_reference_'+kind+'_cal') for kind in ['MLP','TCN','SelfAttention'])
    rows=[]
    rng=np.random.default_rng(20261028)
    for surface in ['HSR','TSR']:
        idx=roles['test'][data.iloc[roles['test']].surface.eq(surface).to_numpy()]
        dates=data.iloc[idx].datetime.dt.tz_convert('Asia/Tokyo').dt.tz_localize(None).dt.normalize()
        blocks=((dates-dates.min()).dt.days//7).to_numpy()
        unique=np.unique(blocks)
        counts=np.array([(blocks==b).sum() for b in unique])
        weights=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=2000)
        denominator=weights@counts
        truth=data.iloc[idx].ape.to_numpy()
        for model,reference in comparisons:
            error_a=(predictions[model][idx]-truth)**2
            error_b=(predictions[reference][idx]-truth)**2
            sums_a=np.array([error_a[blocks==b].sum() for b in unique])
            sums_b=np.array([error_b[blocks==b].sum() for b in unique])
            changes=1000*(np.sqrt((weights@sums_a)/denominator)-np.sqrt((weights@sums_b)/denominator))
            lo,hi=np.quantile(changes,[.025,.975])
            rows.append(dict(surface=surface,model=model,reference=reference,
                             delta_RMSE_meV=1000*(np.sqrt(error_a.mean())-np.sqrt(error_b.mean())),
                             bootstrap_95pct_low_meV=lo,bootstrap_95pct_high_meV=hi,
                             n_rows=len(idx),n_week_blocks=len(unique),bootstrap_repetitions=2000))
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'paired_week_block_bootstrap.csv',index=False,encoding='utf-8-sig')
    return frame


def finish(output,datasets,prep,roles,lookup,beta_model,beta_scaler,predictions,runs,metrics,interpret=True):
    configure_plotting()
    data=datasets['PUBLIC_POWER']
    input_plots(output,datasets,roles,beta_model,beta_scaler)
    basic_plots(output,data,roles,predictions,runs,metrics)
    importances=[]
    if interpret:
        for name,run in runs.items():
            if name.startswith('CONTROL'):
                continue
            print(f'[EXPLANATION START] {name}',flush=True)
            importance=permutation_importance(run,roles,lookup,beta_model,beta_scaler)
            importance['model']=name
            importances.append(importance)
            importance_plot(run,importance)
            attribution_maps(run,roles,beta_model,beta_scaler)
            if run['kind']=='SelfAttention':
                attention_maps(run,roles)
            model_report(run,metrics,importance)
    combined=pd.concat(importances,ignore_index=True) if importances else pd.DataFrame()
    if len(combined):
        combined.to_csv(output/'feature_importance_all_models.csv',index=False,encoding='utf-8-sig')
    detailed=[]
    for name,values in predictions.items():
        if not name.endswith('_cal') and name!='fixed_beta_5':
            continue
        for surface in ['HSR','TSR']:
            idx=roles['test'][data.iloc[roles['test']].surface.eq(surface).to_numpy()]
            train=roles['train_oof'][data.iloc[roles['train_oof']].surface.eq(surface).to_numpy()]
            lo,hi=np.quantile(data.iloc[train].ape,[.1,.9])
            y=data.iloc[idx].ape.to_numpy()
            for band,mask in [('all',np.ones(len(idx),dtype=bool)),('low',y<=lo),('middle',(y>lo)&(y<hi)),('high',y>=hi)]:
                if mask.sum()<2:
                    continue
                met=D.N.F.regression_metrics(y[mask],values[idx][mask])
                met['mape_pct']=100*np.mean(np.abs(y[mask]-values[idx][mask])/y[mask])
                detailed.append(dict(model=name,surface=surface,band=band,low_threshold_train_eV=lo,high_threshold_train_eV=hi,**met))
    pd.DataFrame(detailed).to_csv(output/'test_metrics_by_APE_band.csv',index=False,encoding='utf-8-sig')
    seasonal=[]
    months=data.datetime.dt.tz_convert('Asia/Tokyo').dt.month
    for name,values in predictions.items():
        if not name.endswith('_cal'):
            continue
        for surface in ['HSR','TSR']:
            idx=roles['test'][data.iloc[roles['test']].surface.eq(surface).to_numpy()]
            for season,selected in [('spring',[3,4,5]),('summer',[6,7,8]),('autumn',[9,10,11]),('winter',[12,1,2])]:
                si=idx[months.iloc[idx].isin(selected).to_numpy()]
                if len(si)>1:
                    seasonal.append(dict(model=name,surface=surface,season=season,**D.N.F.regression_metrics(data.iloc[si].ape,values[si])))
    pd.DataFrame(seasonal).to_csv(output/'test_metrics_by_season.csv',index=False,encoding='utf-8-sig')
    weather_comparison=[]
    for split,idx in roles.items():
        hsr=idx[data.iloc[idx].surface.eq('HSR').to_numpy()]
        for feature in D.EXTRAS:
            public=datasets['PUBLIC_POWER'].iloc[hsr][feature].to_numpy()
            nedo=datasets['NEDO_reference'].iloc[hsr][feature].to_numpy()
            weather_comparison.append(dict(split=split,feature=feature,n=len(hsr),correlation=np.corrcoef(public,nedo)[0,1],
                                           mean_public_minus_NEDO=float((public-nedo).mean()),rmse=float(np.sqrt(np.mean((public-nedo)**2)))))
    comparison=pd.DataFrame(weather_comparison)
    comparison.to_csv(output/'public_vs_NEDO_input_comparison.csv',index=False,encoding='utf-8-sig')
    paired_block_bootstrap(output,data,roles,predictions)
    write_report(output,datasets,prep,roles,runs,metrics,combined,comparison)


def write_report(output,datasets,prep,roles,runs,metrics,importance,weather):
    data=datasets['PUBLIC_POWER']
    final=metrics.loc[metrics.split.eq('test')&metrics.surface.ne('ALL')&(metrics.model.str.endswith('_cal')|metrics.model.eq('fixed_beta_5'))].copy()
    columns=['surface','model','n','r2','rmse_meV','mae_meV','mape_pct','mbe_truth_minus_prediction_meV']
    periods=[]
    for role,idx in roles.items():
        periods.append(dict(split=role,hours=data.iloc[idx].datetime.nunique(),start=data.iloc[idx].datetime.min(),end=data.iloc[idx].datetime.max()))
    findings=[]
    indexed=final.set_index(['surface','model'])
    for surface in ['HSR','TSR']:
        base=indexed.loc[(surface,'CONTROL_5_MLP_cal')]
        public=indexed.loc[(surface,'PUBLIC_POWER_MLP_cal')]
        fixed=indexed.loc[(surface,'fixed_beta_5')]
        findings.append(f"- {surface}：同一母集団で5変数MLPから公開10変数MLPへの変更により、R²は{base.r2:.4f}→{public.r2:.4f}、RMSEは{base.rmse_meV:.3f}→{public.rmse_meV:.3f} meV（差{public.rmse_meV-base.rmse_meV:+.3f} meV）。")
        findings.append(f"- {surface}：固定βのみのRMSEは{fixed.rmse_meV:.3f} meV。公開10変数MLPとの差（MLP−固定β）は{public.rmse_meV-fixed.rmse_meV:+.3f} meVであり、5変数MLPとの比較と区別して判断する。")
        for kind in ['MLP','TCN','SelfAttention']:
            p=indexed.loc[(surface,'PUBLIC_POWER_'+kind+'_cal')]
            n=indexed.loc[(surface,'NEDO_reference_'+kind+'_cal')]
            findings.append(f"- {surface} {kind}：公開入力RMSE={p.rmse_meV:.3f}、NEDO参照入力RMSE={n.rmse_meV:.3f} meV。差（公開−NEDO）={p.rmse_meV-n.rmse_meV:+.3f} meV。")
        candidates=final.loc[final.surface.eq(surface)&final.model.str.startswith('PUBLIC_POWER_')]
        best=candidates.sort_values('rmse_meV').iloc[0]
        findings.append(f"- {surface}の公開入力モデル群で、今回のTest RMSEが最小だったのは{best.model}（{best.rmse_meV:.3f} meV）。Testを見て設定を変更する選択には使っていない。")
    calibration=[]
    calibration_changes=[]
    for name,run in runs.items():
        for surface,c in run['calibration'].items():
            calibration.append(dict(model=name,surface=surface,**c))
            a=metrics.loc[metrics.split.eq('test')&metrics.surface.eq(surface)&metrics.model.eq(name+'_raw')].iloc[0]
            b=metrics.loc[metrics.split.eq('test')&metrics.surface.eq(surface)&metrics.model.eq(name+'_cal')].iloc[0]
            calibration_changes.append(dict(model=name,surface=surface,r2_raw=a.r2,r2_cal=b.r2,rmse_raw_meV=a.rmse_meV,rmse_cal_meV=b.rmse_meV,delta_rmse_meV=b.rmse_meV-a.rmse_meV))
    top=[]
    if len(importance):
        for (name,surface),part in importance.loc[importance.stage.eq('calibrated')&~importance.feature.str.startswith('GROUP_')].groupby(['model','surface']):
            part=part.sort_values('mae_increase_meV',ascending=False).head(3)
            top.append(f"- {name} / {surface}："+'、'.join(f"{r.feature}（MAE増分{r.mae_increase_meV:.3f} meV）" for r in part.itertuples()))
    training=pd.DataFrame([dict(model=name,**run['report']['training']) for name,run in runs.items()])
    if 'torch_cpu_threads' in training:
        training['torch_cpu_threads']=training['torch_cpu_threads'].fillna(4).astype(int)
    bootstrap=pd.read_csv(output/'paired_week_block_bootstrap.csv')
    saved_predictions=pd.read_csv(output/'predictions_all_models.csv')
    shift_rows=[]
    for phase in ['validation_calibration','test']:
        for surface in ['HSR','TSR']:
            part=saved_predictions.loc[saved_predictions.split.eq(phase)&saved_predictions.surface.eq(surface)]
            base=part.fixed_beta_5.to_numpy()
            shift_rows.append(dict(split=phase,surface=surface,n=len(part),
                              true_residual_mean_meV=float(1000*np.mean(part.ape_true.to_numpy()-base)),
                              public_MLP_raw_correction_mean_meV=float(1000*np.mean(part.PUBLIC_POWER_MLP_raw.to_numpy()-base)),
                              public_MLP_cal_correction_mean_meV=float(1000*np.mean(part.PUBLIC_POWER_MLP_cal.to_numpy()-base))))
    shift=pd.DataFrame(shift_rows)
    shift.to_csv(output/'calibration_vs_test_residual_mean_shift.csv',index=False,encoding='utf-8-sig')
    attention_ranges=[]
    for source in ['PUBLIC_POWER','NEDO_reference']:
        weights=np.load(output/(source+'_SelfAttention')/'attention_weights_test.npz')['feature_attention']
        mean=weights.mean(axis=(0,1))
        attention_ranges.append(f"{source}で{mean.min():.4f}～{mean.max():.4f}")
    time_audit=''
    if (output/'time_alignment_audit.json').exists():
        audit=json.loads((output/'time_alignment_audit.json').read_text(encoding='utf-8'))
        time_audit='\nTrain内の完全6観測の時間について、保存太陽高度から近似計算した大気外水平面日射量とPOWERのTOAを比較した。採用した時刻対応と±1/2時間のずれを診断し、APEやTest精度を時刻選択に用いていない。下表は補助診断であり、近似軌道補正・格子位置・10分標本中心の差も残る。\n\n'+md_table(pd.DataFrame(audit['results']))+'\n'
    report=f"""# 長沼：10特徴量・1時間平均APEの予測結果と考察

作成日：{pd.Timestamp.now(tz='Asia/Tokyo').date()}。本解析は長沼内の時間順検証であり、五地点から未知地点への移植精度を実証した解析ではない。今後変更され得る特徴量設計の初回比較である。

## 1. 目的・データと入力の保全

基礎変数はAM・TQV・AOD・Ångström指数・TO3に、GHI・晴天度指数Kt・水平面散乱比・気温・相対湿度を加えた10種類。共通{prep['common_hours']:,}時間、HSR/TSR合計{len(data):,}行で比較した。

- Naganuma_DS.csvは読み取り専用。SHA256：`{prep['source_sha256_before']}`。最終照合はinput_integrity.json。
- 正解は有効な10分APEを各表面・JST [hh:00, hh+1:00)で算術平均した値。各時間3点以上。平均スペクトルから再計算したAPEではない。
- 基礎大気データは既存MERRA-2値、AMは既存の幾何計算値。毎時中心と完全一致結合し、10分への補間はしていない。
- 公開主解析：NASA POWERで2011～2015年のGHI・DHI・TOA日射量・気温・RHを取得。UTCを明示し、HHをその時間窓の開始ラベルとしてHH:30中心へ対応させた。
- 長沼の取得座標は北緯43.05°・東経141.75°。[座標の出典](https://kaken.nii.ac.jp/ja/file/KAKENHI-PROJECT-21K05688/21K05688seika.pdf)。元のMERRA-2格子43.0°・141.875°と混同しない。
- NEDO参照解析：GHI・気温・RH・水平面散乱比の同時有効10分値を用い、品質備考1/2、各時間3点以上。DHI=GHI×保存散乱比を10分で求めて時間平均した。
- 両入力条件ともKt=時間GHI/公開TOA日射量、散乱比=時間DHI/時間GHIとし、算出定義を統一。CSV保存Ktをそのまま平均した値とは異なる。GHIが小さい時間等の除外条件はdata_preparation_report.json。
- POWER時間別日射の応答単位Wh/m²は1時間で割った平均W/m²へ対応し、数値は同じ。変数メタデータと生応答JSONをpublic_data/に保存。
- 観測値条件と公開入力条件は時刻・正解を完全一致させた。NEDO参照実験の特徴量も公開取得可能な物理量だが、実際に使用したデータ源は明確に区別する。

{time_audit}

## 2. 学習・評価方法

{md_table(pd.DataFrame(periods))}

基準Fと同じ70/15/15の時間順分割を使用。Validation前半で早期停止、後半だけで校正。分割境界は6個の採用時刻を除外する既存仕様。新たな欠損条件により過去の実験の分割日付とは一致しないため、5変数MLPも今回の共通母集団で再学習した。

- β：大気5変数＋表面ラベル＋各大気変数×表面の11列を標準化し、Ridge α=1。Trainだけで求め、全比較モデルで共有・固定。
- Trainの残差教師：5分割の時間順OOF β予測から作成。Trainの初期OOF不能区間は残差学習から除外。Testの正解値は標準化・モデル学習・早期停止・校正に使わない。
- 残差MLP：従来の128–128–64、LayerNorm/GELU、Dropout 0.05。10変数条件では派生・表面・時刻周期・β予測・残差履歴を含め35入力。5変数対照は30入力。
- TCN：因果Conv1d、kernel=2、dilation=1/2/4、各ブロック2畳み込み、16チャネル。現在＋過去6実時間の7枠を入力し現在の残差を出力。欠測・夜間・split外を単に直前の採用行で詰めない。
- Self-Attention：現在時刻の大気・気象10変数を別トークンにし、表面ラベルの埋め込みを加えて16次元・2ヘッドの特徴量間Attentionを実施。同一気象時刻でも表面に応じて重みを変えられる。過去6時間の大気・気象は線形投影し、補助入力を結合して32次元・4ヘッドの因果時間Attentionで7枠を統合。CPUでの計算量を抑えた構成をSelf-AttentionのTest比較前に固定。現在時刻の気象は利用可能とする同時刻APE推定であり、将来気象は入力しない。
- TCN/Attentionには過去気象も加わるため、MLPとの差をネットワーク形式だけの効果とは解釈しない。総パラメータ数も同一ではない。
- HGB：10変数と静的補助入力で非線形対照。300反復、seed42、内部のランダム早期停止は無効、独立Calibrationで校正。
- NN共通：seed42、AdamW、lr=0.0002、Huber、batch512、最大300 epoch、patience100、勾配ノルム5でclip。実行時間・最良epoch・勾配履歴・重みを保存。
- 早期停止時は従来どおり残差ラグ0/利用可否0。最終推論は過去1/2/3/6時間の**予測**残差を利用し、split・表面ごとに履歴をリセット。学習時のOOF実測残差ラグと推論時の予測残差ラグには違いが残る。
- 校正は表面別のアフィン変換、傾き[0,1.5]、補正上限はCalibration残差のMAD由来robust sigma×3。Testの真の残差で校正しない。
- HSR=0/TSR=32という角度列は従来の表面符号であり、長沼実設置角の確認値ではない。Train指標は残差モデルの学習内診断を含み汎化精度とは呼ばない。

{md_table(training)}

## 3. Test結果

{md_table(final[columns])}

RMSE/MAE/MBEはmeV、MAPEは%。MBEは正解−予測。raw/calの両方をmetrics_all_models.csvに保存した。

![モデル比較](test_model_comparison.png)

![校正の影響](calibration_effect.png)

{md_table(pd.DataFrame(calibration_changes))}

## 4. 結果の考察

{chr(10).join(findings)}

以上は単一seed・単一の時間順分割での結果であり、性能差の統計的有意性や全季節・全地点での優位性は主張しない。公開−NEDOの差は実際の入力情報の違いを含む比較である。公開入力の空間代表性、雲・散乱光の推定誤差、局地的な湿度などが候補要因だが、今回の差だけから個別の原因は特定できない。

Testを開始日から7暦日ごとにまとめた対応ブロックbootstrap（2000反復）によるRMSE差も保存した。負の差が比較モデルの改善を示す。区間が0を含む場合、今回の期間のブロック再標本化では改善方向が安定しているとまでは言えない。この区間は学習seed・地点・データ源の誤差まで含むものではなく、複数比較を補正した検定でもない。

{md_table(bootstrap.loc[bootstrap.reference.eq('CONTROL_5_MLP_cal')])}

校正により改善した面・モデルと悪化した面・モデルは分けて読む。長沼では従来もCalibrationとTestの季節・残差分布差が問題になっており、今回も校正後だけで残差モデルを判断しない。

今回の公開入力3種のNNは、HSR・TSRの全条件で校正後RMSEが増加した。例として公開MLPは、HSRのR²が0.4047→0.2492、TSRが0.6430→0.5660となった。校正前の公開NNは両面とも固定βのR²を上回っているが、校正後HSRでは固定βを下回る。この違いを隠して校正後の数値だけから特徴量やNN全体の有効性を判断しない。Test結果を使った校正係数の再調整は行っていない。

### 校正期間とTestの平均残差の移動

{md_table(shift)}

同じ固定βに対して、必要な平均補正はHSRで校正期間の+22.665 meVからTestの+2.411 meVへ、TSRで+10.190 meVから−2.384 meVへ変化した。公開MLPの校正後補正はTest平均でHSR+12.251 meV、TSR+5.049 meVとなり、必要量を超える平均補正が残った。これは校正後の過大予測と整合する。季節構成等の違いは候補要因だが、積雪・測器・雲・採用データ品質のどれがこの移動を起こしたかは今回だけでは特定できない。

Testで見えた悪化を根拠に校正を都合よく切り替えて最終性能を主張することはせず、固定校正の移植性を次の未使用期間・地点で検証する。

### 公開入力と現地観測入力の対応（Train）

{md_table(weather.loc[weather.split.eq('train_oof')])}

![入力の対応](input_source_comparison_train.png)

GHI・Kt・散乱比は互いに関連し、TQVとRHも異なる物理量ながら関連し得る。相関や重要度を独立した因果効果とは解釈しない。公開データの平均化・格子化では現地の短時間変動が失われ得るが、その影響をβが必ず補えるわけではない。

## 5. 特徴量の効き目とAttention図

### 10変数の残差NNに共通の重要度

Permutation importanceはTestから時間順に均等抽出した各表面最大512行、3反復で算出。モデル・校正は固定し、同時刻のHSR/TSRに同じ置換を適用。大気変数の置換ではβ予測と関連静的交互作用も再計算する。過去の予測残差履歴は元の推論値に固定した**条件付き感度**であり、置換後の全再帰経路を再シミュレーションした重要度ではない。

棒の値は置換後のMAE増分、エラーバーは置換反復間の標準偏差（信頼区間ではない）。負の値は、その置換で誤差が減ったことを示す。相関する特徴量の個別置換は現実には起こりにくい組合せを作り得るため、大気5変数・日射3変数・地上気象2変数の群単位置換も保存した。最良モデル選択には使用していない。

{chr(10).join(top)}

### モデル別の図の意味

- MLP/TCN：通常はAttention重みを持たない。特徴量×時間のIntegrated Gradients図（feature_effect_attribution_map.png）とPermutation importance図を保存した。これらをAttention重みと偽って扱わない。
- IGは表面ごと最大128行・32ステップ。基礎入力10列を学習平均から実入力へ移動し、β予測・大気交互作用・透過率proxyを微分可能な形で再計算する。時間・表面条件・予測残差履歴・利用可否・校正係数は固定。図は固定βと残差補正を含む校正後APEへの条件付き絶対寄与meVであり、因果寄与ではない。学習平均へ同時に移す基準状態は物理的な晴天等を意味しない。
- IG計算用の予測が実際の推論を再現することを数値検証し、32ステップ積分の近似誤差指標もattribution_verification.jsonへ保存する。予測残差の再帰履歴を入力置換後に全て更新する寄与ではない。
- Self-Attention：上記2種に加え、10×10の特徴量間Attention、各ヘッド、現在クエリから過去6時間への時間AttentionをHSR/TSR別に保存。Attentionの行はquery、列はkey。全Testの平均と元の数値重みを保存し、欠測時間への現在クエリの重みが0であることを確認した。
- Attention重みが大きいことを、その特徴量の因果効果や予測改善量と同一視しない。重みとモデル感度は異なる指標である。[Jain & Wallace (2019)](https://aclanthology.org/N19-1357/)はNLPでこの問題を検証しており、本研究では独立のPermutation/IGも併記する。

全Test・両面・ヘッドを平均した特徴量間重みの範囲は、{'、'.join(attention_ranges)}だった。均等配分の0.1付近であり、平均Attention図だけから強い特徴量選択が得られたとは判断しない。表面・ヘッド別の図と数値重み、Permutation/IGを併せて確認する。

![公開MLPの重要度](PUBLIC_POWER_MLP/feature_permutation_importance.png)

![公開TCNの特徴量・時間寄与](PUBLIC_POWER_TCN/feature_effect_attribution_map.png)

![公開Attention HSR](PUBLIC_POWER_SelfAttention/feature_attention_map_HSR.png)

![公開Attention TSR](PUBLIC_POWER_SelfAttention/feature_attention_map_TSR.png)

## 6. 季節・APE帯域と保存結果

test_metrics_by_APE_band.csvに低・中・高・全体のMAPE等を保存。低・高の閾値は各表面のTrain正解の10/90パーセンタイルで固定。test_metrics_by_season.csvに季節別結果を保存。Testに含まれる時期だけの評価であり、冬季全体の精度ではない。

### 校正係数

{md_table(pd.DataFrame(calibration))}

### 保存ファイル

- 各モデルフォルダ：model.pt、preprocessors.joblib、config.json、model_report.json、training_history.csv、learning_history.png/pdf、measured_vs_predicted.png/pdf。
- 10変数の各NN：feature_permutation_importance.csv/png/pdf、feature_effect_attribution_map.csv/png/pdf、feature_attribution_values.npz。
- SelfAttentionの各条件：feature_attention_map_HSR/TSR.png/pdf、feature_attention_heads_HSR/TSR.png/pdf、temporal_attention_map.png/pdf、attention_weights_test.npz、平均重みCSV。
- 共通：predictions_all_models.csv、metrics_all_models.csv、fixed_beta_coefficients.json、split_report.json、入力条件別hourly_dataset CSV、生の公開データJSON、data_preparation_report.json、input_integrity.json。
- 実行：`.venv/Scripts/python.exe -u naganuma_10feature_experiment.py`。同じ設定・データの保存済み学習重みは再利用する。別設定は別出力先を指定する。

## 7. 適用範囲・次の検証

現在は長沼の地点内時間外検証であり、全国推定や五地点からの未知地点β補間の実証ではない。次段階では五地点の公開入力を統一し、4地点のみでβ・残差モデル・校正・移植設定を決め、残り1地点を未知地点として評価する。最終的な特徴量・モデル・分解能は今後の結果で変更し得る。

同時刻の再解析・衛星推定値を使うAPE推定であり、将来予報ではない。元CSV自体がHSR/TSRの採用10分時刻へ限定されているため、夜間・低太陽高度・全気象状態に無条件で外挿しない。1時間平均APEの精度を10分変動の再現精度と読み替えない。

### 取得・実装の一次資料

- [NASA POWER Hourly API](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/)
- [NASA POWERの放射源・空間分解能](https://power.larc.nasa.gov/docs/methodology/data/sources_fluxes/)
- [TCN：Bai et al. (2018)](https://vladlen.info/publications/tcn/)
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

"""
    (output/'結果・考察.md').write_text(report,encoding='utf-8')
    (output/'評価.md').write_text('# 評価結果\n\n詳細な結果・考察、図、手法は [結果・考察.md](結果・考察.md) にまとめた。\n\n'+md_table(final[columns])+'\n',encoding='utf-8')
