"""Saved-model diagnostics and fair two-site comparison; no model selection on Test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import naganuma_10feature_data as D
import naganuma_10feature_experiment as E
import naganuma_10feature_models as M
import naganuma_10feature_reporting as R
import tsukuba_naganuma_10feature_data as C

CONTEXTS=[(design,site) for design in ['native','common_hours'] for site in ['Tsukuba','Naganuma']]
SITE_JA={'Tsukuba':'つくば','Naganuma':'長沼'}


def copy_reference(output):
    folder=C.context_folder(output,'native','Naganuma')
    for source in C.SOURCES:
        shutil.copy2(D.OUTPUT/f'hourly_dataset_{source}.csv',folder/f'hourly_dataset_{source}.csv')
    original=pd.read_csv(D.OUTPUT/'predictions_all_models.csv')
    cache=joblib.load(folder/'prepared.joblib')
    for name in C.MODELS:
        source=D.OUTPUT/name;destination=folder/name;destination.mkdir(exist_ok=True)
        paths=[p for p in source.iterdir() if p.is_file() and (p.name in
            ['model.pt','preprocessors.joblib','config.json','model_report.json','architecture.json','training_history.csv','recursive_predictions.npz',
             'permutation_repetitions.csv','attribution_verification.json','attention_weights_test.npz'] or
             p.name.startswith(('feature_','temporal_attention_')))]
        for path in paths:
            if not (destination/path.name).exists():shutil.copy2(path,destination/path.name)
        frame=original[['datetime','surface','ape_true','ape_samples_in_hour',name+'_raw',name+'_cal']].rename(columns={'ape_true':'ape'})
        frame.to_csv(destination/'predictions.csv',index=False,encoding='utf-8-sig')
        predictions={name+'_'+stage:original[name+'_'+stage].to_numpy() for stage in ['raw','cal']}
        E.metrics_all(cache['datasets']['PUBLIC_POWER'],cache['roles'],predictions).to_csv(destination/'metrics.csv',index=False,encoding='utf-8-sig')
        D.save_json(destination/'job_status.json',dict(status='complete',version=C.VERSION,reused=True,original_folder=str(source),
            copied_files={p.name:D.sha(p) for p in paths if p.name in ['model.pt','preprocessors.joblib','recursive_predictions.npz']}))
    print('[REFERENCE] Original Naganuma weights/predictions copied without fitting; originals unchanged',flush=True)


def load_run(output,design,site,name):
    folder=C.context_folder(output,design,site)
    cache=joblib.load(folder/'prepared.joblib')
    destination=folder/name
    prep=joblib.load(destination/'preprocessors.joblib');config=prep['config']
    model=M.create_model(config['kind'],config['n_features'],config['n_physical'],config['sequence_length'])
    saved=torch.load(destination/'model.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict']);model.eval()
    source='NEDO_reference' if name.startswith('NEDO') else 'PUBLIC_POWER'
    data=cache['datasets'][source];archive=np.load(destination/'recursive_predictions.npz')
    run=dict(model=model,data=data,static=E.static_features(data,cache['beta'],config['n_physical']),
        scaler=prep['feature_scaler'],mean=config['residual_mean'],std=config['residual_std'],inputs=archive['inputs'],
        config=config,calibration=prep['calibration'],folder=destination,name=name,kind=config['kind'],
        report=json.loads((destination/'model_report.json').read_text(encoding='utf-8')),site=site,design=design)
    return run,cache


def basic_plots(run,cache):
    folder=run['folder'];name=run['name'];data=run['data'];test=cache['roles']['test']
    predictions=pd.read_csv(folder/'predictions.csv');metrics=pd.read_csv(folder/'metrics.csv')
    title=f'{run["site"]} / {run["design"]} / {name}'
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for row,surface in enumerate(['HSR','TSR']):
        idx=test[data.iloc[test].surface.eq(surface).to_numpy()]
        truth=data.ape.to_numpy()[idx]
        values=[predictions[name+'_'+stage].to_numpy()[idx] for stage in ['raw','cal']]
        lo=min(truth.min(),*(v.min() for v in values))-.002
        hi=max(truth.max(),*(v.max() for v in values))+.002
        for col,(stage,prediction) in enumerate(zip(['raw','cal'],values)):
            ax=axes[row,col];ax.hexbin(truth,prediction,gridsize=40,mincnt=1,cmap='viridis')
            ax.plot([lo,hi],[lo,hi],'r--',lw=1);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal',adjustable='box')
            m=metrics.loc[(metrics.split=='test')&(metrics.surface==surface)&(metrics.model==name+'_'+stage)].iloc[0]
            ax.set(xlabel='True hourly APE (eV)',ylabel='Predicted APE (eV)',title=f'{surface} / {stage}: R²={m.r2:.4f}, MAPE={m.mape_pct:.4f}%')
    fig.suptitle(title+f' / held-out Test, n={len(test)//2}/surface');fig.tight_layout()
    R.save_figure(fig,folder/'true_vs_predicted.png')
    history=pd.read_csv(folder/'training_history.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(history.epoch,history.train_huber_standardized,label='Train: OOF teacher residual lags')
    axes[0].plot(history.epoch,history.validation_early_huber_standardized,label='Early validation: empty lags')
    best=run['report']['training']['best_epoch'];axes[0].axvline(best,color='gray',ls='--',label=f'Selected {best}')
    axes[0].set(xlabel='Epoch',ylabel='Standardized residual Huber loss');axes[0].legend(fontsize=8)
    axes[1].plot(history.epoch,history.learning_rate);axes[1].set(xlabel='Epoch',ylabel='Learning rate')
    fig.suptitle(title);fig.tight_layout();R.save_figure(fig,folder/'training_history.png')


def explain(output,design,site,name):
    run,cache=load_run(output,design,site,name)
    basic_plots(run,cache)
    display=dict(run,name=f'{site} / {design} / {name}')
    importance=R.permutation_importance(display,cache['roles'],cache['lookup'],cache['beta_model'],cache['beta_scaler'])
    R.importance_plot(display,importance)
    R.attribution_maps(display,cache['roles'],cache['beta_model'],cache['beta_scaler'])
    if run['kind']=='SelfAttention':R.attention_maps(display,cache['roles'])
    D.save_json(run['folder']/'explanation_status.json',dict(status='complete',completed_utc=pd.Timestamp.now(tz='UTC')))
    print('[EXPLANATION COMPLETE] '+design+'/'+site+'/'+name,flush=True)


def collect_context(output,design,site):
    folder=C.context_folder(output,design,site);cache=joblib.load(folder/'prepared.joblib')
    data=cache['datasets']['PUBLIC_POWER'];roles=cache['roles']
    beta=cache['beta'].copy();available=np.isfinite(cache['beta_oof']);beta[available]=cache['beta_oof'][available]
    predictions={'fixed_beta_5':beta}
    for name in C.MODELS:
        assert (folder/name/'job_status.json').exists()
        frame=pd.read_csv(folder/name/'predictions.csv')
        assert np.array_equal(pd.to_datetime(frame.datetime,utc=True),data.datetime)
        assert np.array_equal(frame.surface,data.surface)
        assert np.allclose(frame.ape,data.ape,rtol=0,atol=1e-12)
        for stage in ['raw','cal']:predictions[name+'_'+stage]=frame[name+'_'+stage].to_numpy()
    if cache['reused']:
        original=pd.read_csv(D.OUTPUT/'predictions_all_models.csv')
        for name,values in predictions.items():assert np.allclose(original[name],values,rtol=0,atol=1e-12,equal_nan=True)
    metrics=E.metrics_all(data,roles,predictions)
    metrics.to_csv(folder/'metrics_all_models.csv',index=False,encoding='utf-8-sig')
    labels=np.full(len(data),'purged',dtype=object)
    for role,idx in roles.items():labels[idx]=role
    result=data[['datetime','SiteNum','surface','ape','ape_samples_in_hour']].rename(columns={'ape':'ape_true'}).copy()
    result['split']=labels
    for name,values in predictions.items():result[name]=values
    result.to_csv(folder/'predictions_all_models.csv',index=False,encoding='utf-8-sig')
    metrics['design']=design;metrics['site']=site
    return dict(folder=folder,cache=cache,data=data,roles=roles,predictions=predictions,metrics=metrics,result=result,design=design,site=site)


def metric_values(y,p):
    y=np.asarray(y);p=np.asarray(p);valid=np.isfinite(y)&np.isfinite(p);y=y[valid];p=p[valid]
    if not len(y):return dict(n=0,r2=np.nan,rmse_meV=np.nan,mae_meV=np.nan,mape_pct=np.nan,mbe_meV=np.nan)
    error=y-p;variance=np.var(y)
    return dict(n=len(y),r2=float(1-np.mean(error**2)/variance) if len(y)>1 and variance>0 else np.nan,
        rmse_meV=float(1000*np.sqrt(np.mean(error**2))),mae_meV=float(1000*np.mean(np.abs(error))),
        mape_pct=float(100*np.mean(np.abs(error)/y)),mbe_meV=float(1000*np.mean(error)))


def summary_tables(output,contexts):
    metrics=pd.concat([c['metrics'] for c in contexts.values()],ignore_index=True)
    metrics.to_csv(output/'metrics_all_contexts.csv',index=False,encoding='utf-8-sig')
    metrics.loc[metrics.split.eq('test')].to_csv(output/'test_metrics_all_contexts.csv',index=False,encoding='utf-8-sig')
    calibration=[];distribution=[];coefficients=[];shift=[];features=[];architecture=[];correlations=[]
    for (design,site),context in contexts.items():
        cache,data,roles=context['cache'],context['data'],context['roles'];y=data.ape.to_numpy()
        test=metrics.loc[(metrics.design==design)&(metrics.site==site)&(metrics.split=='test')].set_index(['surface','model'])
        for name in C.MODELS:
            source='NEDO_reference' if name.startswith('NEDO') else 'PUBLIC_POWER' if name!='CONTROL_5_MLP' else 'CONTROL'
            kind='MLP' if name=='CONTROL_5_MLP' else name[len(source)+1:]
            config=json.loads((context['folder']/name/'config.json').read_text(encoding='utf-8'))
            report=json.loads((context['folder']/name/'model_report.json').read_text(encoding='utf-8'))
            architecture.append(dict(design=design,site=site,model=name,kind=kind,n_physical=config['n_physical'],
                n_features=config['n_features'],sequence_length=config['sequence_length'],
                n_train_oof_rows=report['n_train_oof_rows'],**report['training']))
            for surface in ['HSR','TSR','ALL']:
                a=test.loc[(surface,name+'_raw')];b=test.loc[(surface,name+'_cal')]
                calibration.append(dict(design=design,site=site,source=source,kind=kind,model=name,surface=surface,n=int(a.n),
                    r2_raw=a.r2,r2_cal=b.r2,delta_r2=b.r2-a.r2,mape_raw_pct=a.mape_pct,mape_cal_pct=b.mape_pct,
                    delta_mape_percentage_points=b.mape_pct-a.mape_pct,rmse_raw_meV=a.rmse_meV,rmse_cal_meV=b.rmse_meV,
                    mbe_raw_meV=a.mbe_truth_minus_prediction_meV,mbe_cal_meV=b.mbe_truth_minus_prediction_meV))
            for surface in ['HSR','TSR']:
                for role in ['validation_calibration','test']:
                    idx=roles[role];idx=idx[data.iloc[idx].surface.eq(surface).to_numpy()]
                    p=context['predictions'][name+'_raw'][idx]-cache['beta'][idx]
                    shift.append(dict(design=design,site=site,model=name,surface=surface,split=role,
                        true_beta_residual_mean_meV=1000*np.mean(y[idx]-cache['beta'][idx]),
                        predicted_raw_residual_mean_meV=1000*np.mean(p),**report['calibration'][surface]))
        for role,indices in roles.items():
            for surface in ['HSR','TSR']:
                idx=indices[data.iloc[indices].surface.eq(surface).to_numpy()];present=cache['lookup'][idx]>=0
                distribution.append(dict(design=design,site=site,split=role,surface=surface,n=len(idx),
                    target_mean_eV=y[idx].mean(),target_sd_meV=1000*y[idx].std(),
                    mean_ape_samples_per_hour=float(data.iloc[idx].ape_samples_in_hour.mean()),
                    mean_within_hour_ape_sd_meV=float(1000*data.iloc[idx].ape_std_within_hour.mean()),
                    beta_residual_mean_meV=1000*np.mean(y[idx]-cache['beta'][idx]),
                    mean_available_hour_slots=present.sum(1).mean(),full_seven_hour_pct=100*present.all(1).mean(),
                    start_utc=data.iloc[idx].datetime.min(),end_utc=data.iloc[idx].datetime.max()))
        original=json.loads((context['folder']/'fixed_beta_coefficients.json').read_text(encoding='utf-8'))
        terms=original['original_unit_coefficients'];intercept=original['original_unit_intercept']
        for surface,s in [('HSR',0),('TSR',1)]:
            coefficients.append(dict(design=design,site=site,surface=surface,term='intercept',coefficient=intercept+s*terms['TSR_label']))
            for term in D.PHYSICAL[:5]:
                coefficients.append(dict(design=design,site=site,surface=surface,term=term,coefficient=terms[term]+s*terms['TSR_x_'+term]))
        for source,frame in cache['datasets'].items():
            physical=E.static_features(frame,cache['beta'],10)
            idx=roles['train_oof'];idx=idx[frame.iloc[idx].surface.eq('HSR').to_numpy()]
            for name in D.PHYSICAL:
                v=physical.iloc[idx][name].to_numpy()
                features.append(dict(design=design,site=site,source=source,feature=name,n=len(v),mean=v.mean(),std=v.std(),
                    p10=np.quantile(v,.1),median=np.median(v),p90=np.quantile(v,.9)))
            for surface in ['HSR','TSR']:
                selected=roles['train_oof'];selected=selected[frame.iloc[selected].surface.eq(surface).to_numpy()]
                target=y[selected]
                for name in D.PHYSICAL:
                    v=physical.iloc[selected][name].to_numpy()
                    correlations.append(dict(design=design,site=site,source=source,surface=surface,feature=name,n=len(v),
                        pearson_r=float(np.corrcoef(v,target)[0,1]),
                        spearman_r=float(np.corrcoef(pd.Series(v).rank(),pd.Series(target).rank())[0,1])))
    paired=pd.DataFrame(calibration);stats=pd.DataFrame(distribution)
    paired.to_csv(output/'calibration_comparison.csv',index=False,encoding='utf-8-sig')
    stats.to_csv(output/'target_variance_context_and_splits.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(shift).to_csv(output/'calibration_distribution_shift.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coefficients).to_csv(output/'beta_effective_coefficients.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(features).to_csv(output/'physical_feature_distributions_train.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(correlations).to_csv(output/'feature_APE_correlations_train.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(architecture).to_csv(output/'training_and_architecture_summary.csv',index=False,encoding='utf-8-sig')
    comparisons=[]
    for design in ['native','common_hours']:
        a=metrics.loc[(metrics.design==design)&(metrics.site=='Tsukuba')&(metrics.split=='test')].set_index(['model','surface'])
        b=metrics.loc[(metrics.design==design)&(metrics.site=='Naganuma')&(metrics.split=='test')].set_index(['model','surface'])
        for key in a.index:
            first,second=a.loc[key],b.loc[key]
            row=dict(design=design,model=key[0],surface=key[1],n_tsukuba=int(first.n),n_naganuma=int(second.n))
            for field in ['r2','rmse_meV','mae_meV','mape_pct','mbe_truth_minus_prediction_meV']:
                row[field+'_tsukuba']=first[field];row[field+'_naganuma']=second[field]
                row['delta_'+field]=first[field]-second[field]
            comparisons.append(row)
    comparison=pd.DataFrame(comparisons)
    comparison.to_csv(output/'site_comparison_all_models.csv',index=False,encoding='utf-8-sig')
    return metrics,paired,stats,comparison


def detailed_tables(output,contexts):
    bands=[];seasons=[];thresholds={}
    pooled={}
    for surface in ['HSR','TSR']:
        values=[]
        for site in ['Tsukuba','Naganuma']:
            c=contexts[('common_hours',site)];idx=c['roles']['train_oof']
            idx=idx[c['data'].iloc[idx].surface.eq(surface).to_numpy()]
            values.extend(c['data'].ape.to_numpy()[idx])
        pooled[surface]=np.quantile(values,[.1,.9])
    for (design,site),c in contexts.items():
        y=c['data'].ape.to_numpy()
        for surface in ['HSR','TSR']:
            train=c['roles']['train_oof'];train=train[c['data'].iloc[train].surface.eq(surface).to_numpy()]
            test=c['roles']['test'];test=test[c['data'].iloc[test].surface.eq(surface).to_numpy()]
            schemes={'local_train_quantile':np.quantile(y[train],[.1,.9])}
            if design=='common_hours':schemes['pooled_train_common_cutoffs']=pooled[surface]
            for scheme,(low,high) in schemes.items():
                thresholds[f'{design}/{site}/{surface}/{scheme}']=dict(low_eV=low,high_eV=high)
                masks={'low':y[test]<low,'middle':(y[test]>=low)&(y[test]<=high),'high':y[test]>high,'all':np.ones(len(test),bool)}
                for band,mask in masks.items():
                    idx=test[mask]
                    for name,p in c['predictions'].items():
                        bands.append(dict(design=design,site=site,surface=surface,scheme=scheme,band=band,model=name,**metric_values(y[idx],p[idx])))
            months=c['data'].iloc[test].datetime.dt.tz_convert('Asia/Tokyo').dt.month.to_numpy()
            season=np.select([np.isin(months,[3,4,5]),np.isin(months,[6,7,8]),np.isin(months,[9,10,11])],['spring','summer','autumn'],default='winter')
            for label in np.unique(season):
                idx=test[season==label]
                for name,p in c['predictions'].items():
                    seasons.append(dict(design=design,site=site,surface=surface,season=label,model=name,**metric_values(y[idx],p[idx])))
    pd.DataFrame(bands).to_csv(output/'test_metrics_by_APE_band.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(seasons).to_csv(output/'test_metrics_by_season.csv',index=False,encoding='utf-8-sig')
    D.save_json(output/'APE_band_thresholds.json',thresholds)


def bootstrap_tables(output,contexts):
    rng=np.random.default_rng(20261029);rows=[];selections={}
    def evaluate(kind,design,site,name,base,y1,p1,y2,p2,samples,surface,base_site=None):
        delta={};draws={k:[] for k in ['r2','rmse_meV','mae_meV','mape_pct']}
        a,b=metric_values(y1,p1),metric_values(y2,p2)
        for key in draws:delta[key]=a[key]-b[key]
        for sample in samples:
            a,b=metric_values(y1[sample],p1[sample]),metric_values(y2[sample],p2[sample])
            for key in draws:draws[key].append(a[key]-b[key])
        row=dict(comparison_type=kind,design=design,site=site,base_site=base_site or site,model=name,baseline=base,
            surface=surface,n=len(y1),bootstrap_repetitions=len(samples))
        for key,values in draws.items():
            lower,upper=np.quantile(values,[.025,.975]);row['delta_'+key]=delta[key]
            row[key+'_ci95_low']=lower;row[key+'_ci95_high']=upper
        rows.append(row)
    for key,c in contexts.items():
        design,site=key;data=c['data'];y=data.ape.to_numpy()
        for surface in ['HSR','TSR']:
            idx=c['roles']['test'];idx=idx[data.iloc[idx].surface.eq(surface).to_numpy()]
            local=data.iloc[idx].datetime.dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
            block=((local-local.min()).dt.total_seconds()//(7*86400)).to_numpy(int)
            groups=[np.flatnonzero(block==value) for value in np.unique(block)]
            samples=[np.concatenate([groups[g] for g in rng.integers(0,len(groups),len(groups))]) for _ in range(500)]
            selections[(design,site,surface)]=samples
            for name in C.MODELS:
                evaluate('cal_minus_raw',design,site,name+'_cal',name+'_raw',y[idx],c['predictions'][name+'_cal'][idx],
                    y[idx],c['predictions'][name+'_raw'][idx],samples,surface)
            for source in C.SOURCES:
                for stage in ['raw','cal']:
                    base=source+'_MLP_'+stage
                    for kind in ['TCN','SelfAttention']:
                        name=source+'_'+kind+'_'+stage
                        evaluate('backbone_minus_MLP',design,site,name,base,y[idx],c['predictions'][name][idx],
                            y[idx],c['predictions'][base][idx],samples,surface)
                    name=source+'_MLP_'+stage;base='CONTROL_5_MLP_'+stage
                    evaluate('ten_minus_five_features',design,site,name,base,y[idx],c['predictions'][name][idx],
                        y[idx],c['predictions'][base][idx],samples,surface)
    a,b=contexts[('common_hours','Tsukuba')],contexts[('common_hours','Naganuma')]
    for surface in ['HSR','TSR']:
        idx=a['roles']['test'];idx=idx[a['data'].iloc[idx].surface.eq(surface).to_numpy()]
        assert a['data'].iloc[idx].datetime.equals(b['data'].iloc[idx].datetime)
        for name in a['predictions']:
            evaluate('Tsukuba_minus_Naganuma','common_hours','Tsukuba',name,name,a['data'].ape.to_numpy()[idx],a['predictions'][name][idx],
                b['data'].ape.to_numpy()[idx],b['predictions'][name][idx],selections[('common_hours','Tsukuba',surface)],surface,'Naganuma')
    result=pd.DataFrame(rows)
    assert len(result)==182
    result.to_csv(output/'paired_week_block_bootstrap.csv',index=False,encoding='utf-8-sig')
    return result


def comparison_plots(output,metrics,paired,contexts):
    labels=[k+'/'+stage for k in C.KINDS for stage in ['raw','cal']]
    for design in ['common_hours','native']:
        for field,ylabel in [('r2','Test R²'),('mape_pct','Test MAPE (%)')]:
            fig,axes=plt.subplots(2,2,figsize=(13,9))
            for row,source in enumerate(C.SOURCES):
                names=[source+'_'+k+'_'+s for k in C.KINDS for s in ['raw','cal']]
                for col,surface in enumerate(['HSR','TSR']):
                    ax=axes[row,col]
                    for offset,site,color in [(-.18,'Tsukuba','#2878a4'),(.18,'Naganuma','#dd8452')]:
                        part=metrics.loc[(metrics.design==design)&(metrics.site==site)&(metrics.split=='test')&(metrics.surface==surface)].set_index('model')
                        ax.bar(np.arange(len(names))+offset,part.loc[names,field],.36,label=site,color=color)
                        ax.axhline(part.loc['fixed_beta_5',field],color=color,ls='--',lw=.9,label=site+' fixed beta')
                    ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right',fontsize=8)
                    ax.set(ylabel=ylabel,title=source+' / '+surface)
            fig.suptitle(design+' / '+('identical UTC samples and chronological splits' if design=='common_hours' else 'native splits: different dates and sample counts'))
            fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,fontsize=8)
            fig.tight_layout(rect=(0,.055,1,.95));R.save_figure(fig,output/f'{design}_site_comparison_{field}.png')
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,surface in zip(axes,['HSR','TSR']):
            p=paired.loc[(paired.design==design)&(paired.surface==surface)&paired.source.isin(C.SOURCES)].copy()
            p=p.sort_values(['site','source','kind'])
            colors=np.where(p.site.eq('Tsukuba'),'#2878a4','#dd8452')
            labels2=p.site+' '+p.source.str.replace('PUBLIC_POWER','Public',regex=False).str.replace('NEDO_reference','NEDO',regex=False)+' '+p.kind
            ax.barh(np.arange(len(p)),p.delta_r2,color=colors);ax.axvline(0,color='gray',lw=.8)
            ax.set_yticks(range(len(p)),labels2,fontsize=7);ax.set(xlabel='R²(cal) - R²(raw)',title=surface)
        fig.suptitle(design+' / calibration effect');fig.tight_layout();R.save_figure(fig,output/f'{design}_calibration_effect.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    a,b=contexts[('common_hours','Tsukuba')],contexts[('common_hours','Naganuma')]
    for ax,surface in zip(axes,['HSR','TSR']):
        values=[]
        for c in [a,b]:
            idx=c['roles']['test'];idx=idx[c['data'].iloc[idx].surface.eq(surface).to_numpy()];values.append(c['data'].ape.to_numpy()[idx])
        bins=np.linspace(min(v.min() for v in values),max(v.max() for v in values),40)
        for v,site,color in zip(values,['Tsukuba','Naganuma'],['#2878a4','#dd8452']):
            ax.hist(v,bins=bins,density=True,alpha=.5,label=f'{site}: SD={1000*v.std():.2f} meV',color=color)
        ax.set(xlabel='True hourly APE (eV)',ylabel='Density',title=surface);ax.legend(fontsize=8)
    fig.suptitle('Common Test target distributions / identical UTC timestamps');fig.tight_layout();R.save_figure(fig,output/'common_target_distribution.png')


def input_and_alignment_audit(output,contexts):
    rows=[]
    fig,axes=plt.subplots(2,5,figsize=(16,6))
    for row,site in enumerate(['Tsukuba','Naganuma']):
        c=contexts[('common_hours',site)];idx=c['roles']['train_oof'];idx=idx[c['data'].iloc[idx].surface.eq('HSR').to_numpy()]
        for ax,feature in zip(axes[row],D.EXTRAS):
            reference=c['cache']['datasets']['NEDO_reference'].iloc[idx][feature].to_numpy()
            public=c['cache']['datasets']['PUBLIC_POWER'].iloc[idx][feature].to_numpy()
            correlation=float(np.corrcoef(reference,public)[0,1])
            rows.append(dict(site=site,feature=feature,n=len(idx),pearson_r=correlation,bias_public_minus_reference=float(np.mean(public-reference)),
                rmse_public_minus_reference=float(np.sqrt(np.mean((public-reference)**2)))))
            ax.hexbin(reference,public,gridsize=30,mincnt=1,cmap='viridis')
            lo,hi=min(reference.min(),public.min()),max(reference.max(),public.max());ax.plot([lo,hi],[lo,hi],'r--',lw=.7)
            ax.set(xlabel='NEDO reference',ylabel='Public POWER',title=f'{site} / {feature}\nr={correlation:.3f}')
    fig.suptitle('Input-source agreement / Common Train only');fig.tight_layout();R.save_figure(fig,output/'input_source_agreement_train.png')
    pd.DataFrame(rows).to_csv(output/'input_source_agreement_train.csv',index=False,encoding='utf-8-sig')
    # Independent geometry check for the pre-specified HH -> HH:30 POWER relabel.
    by_site={}
    for site in ['Tsukuba','Naganuma']:
        raw=C.load_tsukuba_source() if site=='Tsukuba' else D.N.read_source(D.SOURCE)
        alt=pd.to_numeric(raw.solar_altitude_deg,errors='coerce')
        day=raw.datetime.dt.tz_convert('Asia/Tokyo').dt.dayofyear.to_numpy()
        proxy=1361*(1+.033*np.cos(2*np.pi*day/365))*np.maximum(0,np.sin(np.deg2rad(alt)))
        geometry=pd.DataFrame({'datetime':E.F.hourly_center_utc(raw.datetime),'proxy':proxy}).groupby('datetime').proxy.mean()
        power_root=output if site=='Tsukuba' else D.OUTPUT
        power=pd.read_csv(power_root/'public_data/NASA_POWER_hourly_2011_2015.csv')
        power.datetime=pd.to_datetime(power.datetime,utc=True);power=power.set_index('datetime').TOA_SW_DWN
        c=contexts[('native',site)];times=c['data'].iloc[c['roles']['train_oof']].datetime.drop_duplicates()
        audit={}
        for offset in [-1,0,1]:
            a=geometry.reindex(times).to_numpy();b=power.reindex(pd.DatetimeIndex(times)+pd.Timedelta(hours=offset)).to_numpy()
            ok=np.isfinite(a)&np.isfinite(b);audit[str(offset)]=dict(n=int(ok.sum()),pearson_r=float(np.corrcoef(a[ok],b[ok])[0,1]))
        assert audit['0']['pearson_r']>.98 and audit['0']['pearson_r']>max(audit['-1']['pearson_r'],audit['1']['pearson_r'])
        by_site[site]=audit
    D.save_json(output/'POWER_time_alignment_audit.json',dict(train_only=True,by_site=by_site,
        convention='predefined UTC HH -> centre HH:30, same as Naganuma; audit did not choose or refit a time offset'))


def feature_comparison_plots(output,contexts):
    tables=[]
    for (design,site),context in contexts.items():
        for name in C.MODELS:
            table=pd.read_csv(context['folder']/name/'feature_permutation_importance.csv')
            table['design']=design;table['site']=site;table['model']=name
            tables.append(table)
    all_features=pd.concat(tables,ignore_index=True)
    all_features.to_csv(output/'feature_permutation_importance_all_contexts.csv',index=False,encoding='utf-8-sig')
    features=D.PHYSICAL+['GROUP_atmosphere','GROUP_radiation','GROUP_local_weather']
    labels=R.LABELS+['Atmosphere group','Radiation group','Weather group']
    for source in C.SOURCES:
        selected=all_features.loc[(all_features.design=='common_hours')&all_features.model.str.startswith(source+'_')]
        rows=[(kind,surface,site) for kind in ['MLP','TCN','SelfAttention'] for surface in ['HSR','TSR'] for site in ['Tsukuba','Naganuma']]
        arrays=[]
        for stage in ['raw','calibrated']:
            part=selected.loc[selected.stage.eq(stage)].set_index(['model','surface','site','feature'])
            arrays.append(np.array([[part.loc[(source+'_'+kind,surface,site,feature),'mae_increase_meV']
                for feature in features] for kind,surface,site in rows]))
        limit=max(float(np.max(np.abs(a))) for a in arrays)
        fig,axes=plt.subplots(1,2,figsize=(17,7.2),sharey=True,layout='constrained')
        for ax,array,stage in zip(axes,arrays,['raw: beta + residual','cal: beta + calibrated residual']):
            im=ax.imshow(array,cmap='coolwarm',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_xticks(range(len(features)),labels,rotation=55,ha='right',fontsize=8)
            ax.set_yticks(range(len(rows)),[f'{k} / {s} / {site}' for k,s,site in rows],fontsize=8)
            ax.set_title(stage)
            for y in [3.5,7.5]:ax.axhline(y,color='black',lw=.8)
        fig.colorbar(im,ax=axes,label='Test MAE increase after permutation (meV)',shrink=.85)
        fig.suptitle(source+' / same Test times / physical-input sensitivity with predicted history fixed')
        R.save_figure(fig,output/f'common_hours_{source}_feature_sensitivity.png')


def write_reports(output,contexts,metrics,paired,stats,comparison,bootstrap):
    primary=paired.loc[(paired.design=='common_hours')&paired.source.isin(C.SOURCES)&paired.surface.ne('ALL')]
    table=primary[['site','source','kind','surface','r2_raw','r2_cal','delta_r2','mape_raw_pct','mape_cal_pct','rmse_raw_meV','rmse_cal_meV']]
    brief=[];winners=[]
    for site in ['Tsukuba','Naganuma']:
        for source in C.SOURCES:
            for surface in ['HSR','TSR']:
                p=metrics.loc[(metrics.design=='common_hours')&(metrics.site==site)&(metrics.split=='test')&(metrics.surface==surface)&metrics.model.str.startswith(source+'_')]
                best=p.sort_values('r2',ascending=False).iloc[0];mape=p.sort_values('mape_pct').iloc[0]
                winners.append(dict(site=site,source=source,surface=surface,r2_best_model=best.model,r2=best.r2,
                    mape_best_model=mape.model,mape_pct=mape.mape_pct))
                brief.append(f'- {SITE_JA[site]}・{source}・{surface}：R²最大は{best.model} **{best.r2:.4f}**、MAPE最小は{mape.model} **{mape.mape_pct:.4f}%**。')
    pd.DataFrame(winners).to_csv(output/'common_hours_descriptive_best_models.csv',index=False,encoding='utf-8-sig')
    effects=[]
    for site in ['Tsukuba','Naganuma']:
        for source in C.SOURCES:
            for surface in ['HSR','TSR']:
                p=primary.loc[(primary.site==site)&(primary.source==source)&(primary.surface==surface)]
                effects.append(f'- {SITE_JA[site]}・{source}・{surface}：校正でR²改善 {int((p.delta_r2>0).sum())}/3、MAPE改善 {int((p.delta_mape_percentage_points<0).sum())}/3。')
    common_test=stats.loc[(stats.design=='common_hours')&(stats.split=='test')]
    variance_table=common_test[['site','surface','n','target_mean_eV','target_sd_meV','mean_available_hour_slots','full_seven_hour_pct','mean_ape_samples_per_hour']]
    variance_notes=[]
    for surface in ['HSR','TSR']:
        p=common_test.loc[common_test.surface.eq(surface)].set_index('site')
        a,b=p.loc['Tsukuba'],p.loc['Naganuma']
        variance_notes.append(f'- 共通Testの{surface}標準偏差：つくば {a.target_sd_meV:.3f} meV、長沼 {b.target_sd_meV:.3f} meV。比（つくば／長沼）は{a.target_sd_meV/b.target_sd_meV:.3f}。')
    baseline=metrics.loc[(metrics.design=='common_hours')&(metrics.split=='test')&metrics.model.eq('fixed_beta_5')&metrics.surface.ne('ALL'),
        ['site','surface','r2','rmse_meV','mape_pct']]
    shift=stats.loc[(stats.design=='common_hours')&stats.split.isin(['validation_calibration','test'])]
    shift_table=shift[['site','split','surface','n','beta_residual_mean_meV','target_sd_meV']]
    site_table=comparison.loc[(comparison.design=='common_hours')&comparison.surface.ne('ALL')&comparison.model.str.startswith(tuple(C.SOURCES))]
    site_cols=['model','surface','r2_tsukuba','r2_naganuma','delta_r2','rmse_meV_tsukuba','rmse_meV_naganuma','mape_pct_tsukuba','mape_pct_naganuma']
    dates=stats.loc[stats.surface.eq('HSR'),['design','site','split','n','start_utc','end_utc']]
    coefficients=pd.read_csv(output/'beta_effective_coefficients.csv')
    coefficient_table=coefficients.loc[coefficients.design.eq('common_hours')].pivot(index=['surface','term'],columns='site',values='coefficient').reset_index()
    coefficient_table['Tsukuba_minus_Naganuma']=coefficient_table.Tsukuba-coefficient_table.Naganuma
    coefficient_table.to_csv(output/'common_beta_coefficient_comparison.csv',index=False,encoding='utf-8-sig')
    agreement=pd.read_csv(output/'input_source_agreement_train.csv')
    model_intervals=bootstrap.loc[(bootstrap.comparison_type=='backbone_minus_MLP')&
        (bootstrap.design=='common_hours')&bootstrap.model.str.endswith('_cal')].copy()
    model_intervals['R2_difference_CI95']=[f'[{lo:.4f}, {hi:.4f}]' for lo,hi in
        zip(model_intervals.r2_ci95_low,model_intervals.r2_ci95_high)]
    model_intervals['MAPE_difference_CI95']=[f'[{lo:.4f}, {hi:.4f}]' for lo,hi in
        zip(model_intervals.mape_pct_ci95_low,model_intervals.mape_pct_ci95_high)]
    interval_cols=['site','surface','model','delta_r2','R2_difference_CI95','delta_mape_pct','MAPE_difference_CI95']
    agreement_notes=[]
    for feature in ['GHI','diffuse_fraction']:
        p=agreement.loc[agreement.feature.eq(feature)].set_index('site')
        a,b=p.loc['Tsukuba'],p.loc['Naganuma']
        agreement_notes.append(f'- {feature}の公開値と参照値の対応：つくば r={a.pearson_r:.4f}・差のRMSE={a.rmse_public_minus_reference:.4f}、長沼 r={b.pearson_r:.4f}・差のRMSE={b.rmse_public_minus_reference:.4f}（GHIはW/m²、散乱比は無次元）。')
    environment=json.loads((output/'execution_plan.json').read_text(encoding='utf-8'))['environment']
    test_n=int(common_test.iloc[0].n)
    root=f'''# つくば・長沼：10特徴量、MLP／TCN／Self-Attentionの比較

## 主比較の結論

両地点で共通する8,876時間に限定し、**同じTrain 6,213時間、早期停止Validation 657時間、校正Validation 662時間、Test {test_n}時間／面**で両地点を学習し直した。共通Testは2015年4月28日〜12月30日（UTC中心時刻）。各地点のHSR/TSRを合同学習し、β・前処理・ネットワーク・校正は地点別に学習する。

{chr(10).join(brief)}

上記はこのTestでの記述的な最大・最小。Testを使って構造、特徴量、校正の採否を再調整していない。単一seed 42で、差が小さいモデルの普遍的な優劣を確定したものではない。

## 校正前後の全結果：共通時刻比較

**raw＝固定β＋未校正残差、cal＝固定β＋面別校正残差。rawはβ単体ではない。** β単体はfixed_beta_5。MAPEは100×平均(|予測−実測|/実測)で単位は%。RMSE・MAEはmeV。MBEは実測−予測で、負は過大予測。ALLは両面の実測・予測ペアの縦結合で、APEの加算でもR²の平均でもない。主表は面別。

{R.md_table(table)}

![共通時刻R²](common_hours_site_comparison_r2.png)

![共通時刻MAPE](common_hours_site_comparison_mape_pct.png)

## 同じモデルでの地点差

ΔR²はつくば−長沼。R²は各地点の真値分散で正規化されるため、正だから絶対誤差が必ず小さいとは限らない。RMSE・MAPEと下の標準偏差を一緒に読む。

{R.md_table(site_table[site_cols])}

![APE分布](common_target_distribution.png)

{R.md_table(variance_table)}

{chr(10).join(variance_notes)}

以下は補正前の固定β単体。同じ特徴量・同じ学習時間数で、基礎線形関係の再現度に差があるかを確認する。

{R.md_table(baseline)}

7時間窓でも7枠すべてに観測があるとは限らない。共通時刻比較では両地点の可用性配置も同じになる。ただし1時間内の有効10分観測の数・時刻まで一致させた比較ではない。

## キャリブレーションの違い

{chr(10).join(effects)}

![校正効果](common_hours_calibration_effect.png)

校正は独立したCAL期間のみで面別にr_true≈a×r_raw+bを回帰し、aを0〜1.5へ制限する。補正量はCALの実測β残差の3×1.4826×MADでclip。βそのものは変更しない。raw/calは同じraw予測残差履歴を使い、校正後の値を履歴へ戻す再推論はしていない。

校正の転用を考える際は、CALとTestのβ残差平均がどれだけ変わったかが手掛かりになる。固定切片による冬季の補正が他季節で過補正になる場合がある。ただし季節、入力源、観測・選別、モデルの応答を因果的に分離した実験ではない。モデル別の傾き・切片・raw残差平均はcalibration_distribution_shift.csv。

{R.md_table(shift_table)}

## 地点差を解釈するためのデータ側の確認

同じ時刻・手法にそろえても、AM、水蒸気、エアロゾル、日射・散乱比、温湿度の値とAPEの関係は地点ごとに異なり得る。physical_feature_distributions_train.csvにTrainのみの平均・標準偏差・10/50/90パーセンタイル、feature_APE_correlations_train.csvにTrainの単変量Pearson/Spearman相関を保存。相関は季節や他変数の共変動を調整した因果効果ではない。真値の標準偏差が近い場合にRMSE差も大きければ、R²の分母だけでは精度差を説明できない。またNEDO参照でも地点差が残るなら、公開入力の代表性だけで説明することもできない。共通時刻での公開入力とNEDO参照の対応は次の通り。

{R.md_table(agreement)}

{chr(10).join(agreement_notes)}

![入力源の対応](input_source_agreement_train.png)

公開値と地上観測の対応が地点で異なる場合、空間平均化、雲の再現、測定と再解析の代表性の違いなどが候補になる。入力源を変えた同モデルの誤差差は参考になるが、対応の相関だけで予測差の原因を断定しない。地域の気象・季節に加え、雪・地表反射、TSRの実設置条件、分光装置・品質選別、1時間内の観測数も介在し得る。これらは今回追加した説明変数や対照実験で個別に実証した原因ではない。

## 固定βの比較

大気5変数＋面ラベル＋交互作用5列を標準化したRidge（α=1）。両面の誤差を同時に最小化し、切片は罰しない。表は元の単位へ戻した各面の実効係数。TSRは基準β＋面差γ。入力の単位・変数対応は両地点でそろえた。係数には変数の相関や標準化後の正則化が影響し、因果的な感度そのものではない。

{R.md_table(coefficient_table,digits=8)}

βは各地点のTrainだけで求めて固定する。つくばβを長沼へ流用していない。**共通時刻版では長沼βも共通Trainで再推定**しており、以前の長沼の数値との差にはコホートと学習対象の変更が含まれる。長沼の改善実験と解釈しない。

## 再現した解析条件

- 物理量10種類：AM、TQV（omega）、AOD550、Ångström指数、TO3、GHI、晴天度指数Kt、水平面散乱比、気温、相対湿度。
- PUBLIC_POWER：追加気象・日射量をNASA POWERから取得。NEDO_reference：DSの観測参照値を利用。両条件でKtの分母G0は同じPOWERのTOA。水平面DHIは10分GHI×散乱比を作ってから時間平均し、散乱比は時間平均DHI/GHIで再構成する。気象リマーク1/2、同時に有効なGHI・散乱比・気温・RHの観測が3点以上の時間枠を採用。長沼と同じ品質条件。
- APE正解は有効10分APEの算術時間平均、面ごとに3点以上。平均スペクトルから再計算したAPEではない。DSのAPEを使用し、元のCSVは書き換えない。つくばDSのAPEは原pklの同時刻値との一致を確認した。つくば原pklの波長帯は350〜1050 nmの記録がある一方、長沼DS自体には波長帯・装置の全情報が保存されないため、長沼の元スペクトルから波長帯を再計算した確認は今回行っていない。
- HSR/TSRは別の正解値と別の行。面情報を入力した合同モデルで学習し、過去残差・校正は面別。0/32の傾斜列は既存の表面符号を維持したもので、両地点の実設置角が等しいと確認した数値ではない。標準化後は二値面ラベルと重複する。
- 残差モデルは**以前の長沼と同じ実装・全層構成**。MLP 128→128→64、TCNは16ch・kernel2・dilation1/2/4の因果畳み込み、Self-Attentionは現在の物理量token注意＋過去物理量の射影＋補助情報と因果時間注意。抽出部・出力層とも最初から学習し、βだけを固定。新たにエンコーダーを凍結した比較ではない。
- 物理10列＋面・季節・時刻・β予測等の補助17列＋過去残差1/2/3/6時間の値・可用性8列＝35列。これに時間枠の可用性1列を加え、現在枠＋過去6時間の7枠を与える。MLPは現在枠の35列（過去残差列を含む）を使い、TCN/Self-Attentionは7枠を処理する。そのため、MLPと時間モデル間の比較には過去共変量の使い方の差も含む。一方、同じモデルでの地点間比較は入力の構成と履歴長を一致させている。5変数MLP対照は同じ補助情報を残して物理量5種類に限定。
- Trainのβ予測は時間順5分割OOFを作って残差教師にする。NNの特徴量標準化と残差の平均・標準偏差はOOF教師があるTrainだけで推定。OOFの初期warm-upはNN学習対象外。入力中のbeta_predictionはTrain全体で推定した固定βの値。Train指標は最終NNを使う診断値で、完全なOOF汎化評価ではない。
- 推論の過去残差にはモデル自身のraw予測だけを使う。Testの実測APEを履歴に入れない。面別・区間別に初期化し、欠けた実経過時間を詰めて「直近7観測」にはしない。早期停止Validationは過去残差を空にする既存方式で、Trainの観測OOF履歴・推論の自己予測履歴との分布差が残る。時間モデル間差の候補であり、今回その学習方式は変更しない。
- 同じseed42、AdamW、Huber損失、batch512、lr 0.0002、最大300epoch、patience100、勾配clip5、指数lr減衰0.99。学習条件はTestで再探索しない。パラメータ数・選択epochはtraining_and_architecture_summary.csv。lossは地点ごとの残差標準化後の値なので、異なる地点のloss値を絶対誤差として直接比較しない。
- 校正前後、β単体、5変数MLP対照、APE低/中/高/全体、季節別、散布図、学習履歴、Permutation・IG寄与図、Self-Attention実重みを保存。通常のMLP/TCNにAttention重みを作ったと表示しない。Attentionは予測寄与度や因果効果と同一視しない。Permutation/IGは生成済みの過去予測履歴を固定した条件付き感度で、履歴全体を再実行した介入ではない。
- 1時間枠中心HH:30にラベルし、MERRA-2と完全一致で結合。NASA POWERのUTC HHは同じ時間枠のHH:30へ付け替え、事前に決めた定義をTrainの太陽高度・大気上端日射で独立確認する。POWER_time_alignment_audit.jsonを参照。対象枠全体の平均値を使う同時刻推定であり、時間枠完了前や1時間先のリアルタイム予報の実証ではない。

## 領域別誤差・参考区間

test_metrics_by_APE_band.csvに、各地点Trainの10/90パーセンタイルを境界にした従来の領域区分を保存。共通時刻比較では**両地点のTrainを合わせた共通の絶対APE閾値**による結果も追加した。同じ「低域」という名前でも地点別閾値では範囲が異なるため、地点間の領域比較はpooled_train_common_cutoffsを使う。狭い領域では真値分散が小さく、R²が大きく負になる場合があるためMAPE・MAE・件数を優先して読む。

paired_week_block_bootstrap.csvは各Test開始からの連続7日区間を500回再抽出した参考95%区間。共通時刻では同じ時計を再抽出して地点差を計算する。校正有無、TCN/Attention−MLP、10−5特徴量も含む。差は名前の前者−後者で、R²は正が改善、MAPE/RMSE/MAEは負が改善。区間が0を含む場合は符号が安定しない。このTestの時間依存を部分的に考慮するもので、学習seed・別年・別地点の不確実性や多数比較の補正を含まない。

次表は校正後のTCN／Self-Attentionと、同じ入力源・同じ面のMLPとの差。MAPE差の単位はパーセントポイント。MLPとの比較には過去共変量の使い方の差も含む。

{R.md_table(model_intervals[interval_cols])}

## 同じモデルでの特徴量感度の比較

下図は各物理量を入れ替えたときの最終予測のMAE増加量（meV）。赤いほど、その条件の予測がその入力に依存する。βと残差補正を合わせた感度で、過去の自己予測残差は固定した。青は入れ替え後にMAEが下がった場合。3回の入れ替え・各面最大512点の推定で、共変量の相関や非現実的な入力組合せの影響も含む。単変量の大小だけで原因や物理的重要性を断定しない。末尾3列は変数群をまとめて入れ替えた結果で、個別列の合計ではない。

![公開入力の特徴量感度](common_hours_PUBLIC_POWER_feature_sensitivity.png)

![参照入力の特徴量感度](common_hours_NEDO_reference_feature_sensitivity.png)

数値と繰返し標準偏差はfeature_permutation_importance_all_contexts.csv。各モデル配下にはIG寄与図を、Self-Attentionには特徴量・時間Attentionの実重みの図も保存した。Attentionと上図の予測感度は別の量。

## 従来条件での参考比較

つくばの全有効11,689時間に従来と同じ70/15/15ルールを適用したnative版も学習した。長沼native版は既存の保存重み・予測を再利用し、改変しない。**native同士はTestの日付・件数が違う**ため、厳密な同時刻比較と混同しない。

![native R²](native_site_comparison_r2.png)

![native MAPE](native_site_comparison_mape_pct.png)

{R.md_table(dates)}

## 適用範囲と未検証事項

主比較は同じ時間枠、同じ区間、同じ学習時間数をそろえた地点内の時系列評価。両地点で正解を持つTrainからβとNNを学んでおり、長沼モデルをつくばへ移す転移実験でも、未知地点でのβ生成の実証でもない。全国展開には地点を外す検証が別途必要。

データは両面のAPEと両入力源が有効な時間に限定している。共通時刻版はさらに両地点の有効性で選別するため、全日中時間の代表ではない。履歴の欠測配置にも観測・品質選別が反映され、未知地点で公開データだけから全時間枠を作る条件とは異なる。地点差を気象だけの原因と断定しない。

## 実行環境と保存先

{R.md_table(pd.DataFrame([dict(package=k,version=v) for k,v in environment['versions'].items()]))}

Windowsの実行制御で既存SciPy 1.18のDLLを読み込めなかったため、`.venv-site-comparison`に公式配布のSciPy 1.17.1を分離し、同じPyTorch/NumPy/scikit-learn等を利用した。既存環境とセキュリティ設定は変更していない。長沼nativeのβは保存モデルとの予測一致を1e-12 eV以下で確認。共通時刻版の両地点は同じ実行環境で学習した。

各design/site/modelフォルダに予測・評価・学習履歴・散布図・寄与図・モデル重み・前処理・校正係数・結果と考察を保存。native/Naganumaは元の保存物のコピーで、元の長沼の結果も元DSも変更していない。実行・入力・コードの記録はexecution_plan.json、input_integrity.json、verification.json。

実行：`.venv-site-comparison/Scripts/python.exe tsukuba_naganuma_10feature_data.py`、学習：同Pythonで`tsukuba_naganuma_10feature_experiment.py --run-all`、集計：同Pythonで`tsukuba_naganuma_10feature_reporting.py --finalize`。

## データ・位置の資料

- [NASA POWER Hourly API](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/)
- [NEDO日射スペクトルデータベースの5地点](https://www.nedo.go.jp/news/press/AA5_100960.html)
- [NEDOデータ利用時の観測地点座標の記載がある研究成果報告書](https://kaken.nii.ac.jp/ja/file/KAKENHI-PROJECT-21K05688/21K05688seika.pdf)：つくば36°03′、140°08′、長沼43°03′、141°45′の公称位置を使用。
'''
    (output/'結果・考察.md').write_text(root,encoding='utf-8')
    evaluation=f'''# 評価：つくば・長沼10特徴量の同モデル比較

主評価はcommon_hours。同じ8,876時間、同じ分割、Testは両地点・各面{test_n}時間。物理10特徴量、β＋MLP/TCN/Self-Attention、校正前後を比較。

{R.md_table(table)}

raw=β＋未校正残差、cal=β＋校正残差。MAPEは平均絶対予測誤差率（%）、RMSEはmeV。β単体と5変数MLP対照、全4区間、ALL/HSR/TSRの指標はmetrics_all_contexts.csv。地点差はsite_comparison_all_models.csv、校正差はcalibration_comparison.csv。低/中/高域の比較では閾値方式にも注意する。

[結果・考察](結果・考察.md)にnative参考結果、データ分布、β、校正の分布差、制約を記載。散布図と学習履歴は各モデル配下。数値の独立再計算と入力保全はverification.json。
'''
    (output/'評価.md').write_text(evaluation,encoding='utf-8')
    for key,c in contexts.items():
        design,site=key
        selected=paired.loc[(paired.design==design)&(paired.site==site)&paired.source.isin(C.SOURCES)&paired.surface.ne('ALL')]
        cols=['source','kind','surface','r2_raw','r2_cal','delta_r2','mape_raw_pct','mape_cal_pct','rmse_raw_meV','rmse_cal_meV']
        text=f'# {SITE_JA[site]}・{design}：結果・考察\n\n'+R.md_table(selected[cols])+f'\n\n[地点間比較と条件の詳細](../../結果・考察.md)\n'
        (c['folder']/'結果・考察.md').write_text(text,encoding='utf-8')
        (c['folder']/'評価.md').write_text('# 評価\n\n'+R.md_table(selected[cols])+'\n',encoding='utf-8')
        for name in C.MODELS:
            folder=c['folder']/name
            m=c['metrics'].loc[(c['metrics'].split=='test')&c['metrics'].surface.ne('ALL')&c['metrics'].model.isin([name+'_raw',name+'_cal'])]
            cols2=['surface','model','n','r2','mape_pct','rmse_meV','mae_meV','mbe_truth_minus_prediction_meV']
            images='\n\n'.join(f'![{label}]({file})' for label,file in [('正解対予測','true_vs_predicted.png'),('学習履歴','training_history.png'),
                ('Permutation感度','feature_permutation_importance.png'),('条件付きIG','feature_effect_attribution_map.png')])
            if name.endswith('SelfAttention'):
                images+='\n\n![HSR特徴量Attention](feature_attention_map_HSR.png)\n\n![TSR特徴量Attention](feature_attention_map_TSR.png)\n\n![時間Attention](temporal_attention_map.png)'
            text=f'''# {SITE_JA[site]}・{design}・{name}

{R.md_table(m[cols2])}

rawはβ＋未校正残差、calはβ＋面別校正残差。補正モデルの抽出部・出力部をともに学習し、βは固定。HSR/TSRの履歴は分離。{('既存の長沼重み・予測・寄与解析を再利用した参考条件。' if c['cache']['reused'] else '今回学習した条件。')}

{images}

寄与図は履歴を固定したモデル感度で因果効果ではない。Attentionの重みも寄与度そのものではない。IGの再構成・完全性誤差はattribution_verification.json。

[全体の比較・条件・考察](../../../結果・考察.md)
'''
            (folder/'結果・考察.md').write_text(text,encoding='utf-8')
            (folder/'評価.md').write_text('# 評価\n\n'+R.md_table(m[cols2])+'\n\n[図・考察](結果・考察.md)\n',encoding='utf-8')


def finalize(output):
    contexts={key:collect_context(output,*key) for key in CONTEXTS}
    metrics,paired,stats,comparison=summary_tables(output,contexts)
    detailed_tables(output,contexts)
    bootstrap=bootstrap_tables(output,contexts)
    comparison_plots(output,metrics,paired,contexts)
    feature_comparison_plots(output,contexts)
    input_and_alignment_audit(output,contexts)
    write_reports(output,contexts,metrics,paired,stats,comparison,bootstrap)
    print('[REPORTS SAVED] '+str(output),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=C.OUTPUT)
    p.add_argument('--copy-reference',action='store_true')
    p.add_argument('--audit-data',action='store_true')
    p.add_argument('--explain',nargs=3,metavar=('DESIGN','SITE','MODEL'))
    p.add_argument('--finalize',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);R.configure_plotting()
    if args.copy_reference:copy_reference(args.output)
    if args.audit_data:
        contexts={}
        for key in CONTEXTS:
            cache=joblib.load(C.context_folder(args.output,*key)/'prepared.joblib')
            contexts[key]=dict(cache=cache,data=cache['datasets']['PUBLIC_POWER'],roles=cache['roles'])
        input_and_alignment_audit(args.output,contexts)
    if args.explain:explain(args.output,*args.explain)
    if args.finalize:finalize(args.output)


if __name__=='__main__':main()
