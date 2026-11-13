"""Independent audit of two-site matching, predictions, calibration and results."""
from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
import naganuma_10feature_data as D
import naganuma_10feature_experiment as E
import naganuma_10feature_models as M
import tsukuba_naganuma_10feature_data as C


def calculate(y,p):
    y=np.asarray(y);p=np.asarray(p);e=y-p
    return dict(r2=1-np.mean(e**2)/np.var(y),rmse_meV=1000*np.sqrt(np.mean(e**2)),
        mae_meV=1000*np.mean(np.abs(e)),mape_pct=100*np.mean(np.abs(e)/y),
        mbe_truth_minus_prediction_meV=1000*e.mean())


def verify(output=C.OUTPUT):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    plan=json.loads((output/'execution_plan.json').read_text(encoding='utf-8'))
    for path,digest in plan['input_sha256'].items():assert D.sha(path)==digest
    current_model_sha=D.sha(D.ROOT/'naganuma_10feature_models.py')
    original_status=json.loads((D.OUTPUT/'execution_status.json').read_text(encoding='utf-8'))
    assert current_model_sha==original_status['code_sha256']['naganuma_10feature_models.py']
    contexts={};max_metric_error=0.;max_lag_error=0.;max_replay_error=0.;checked=0;histories=0
    for design in ['native','common_hours']:
        for site in ['Tsukuba','Naganuma']:
            folder=C.context_folder(output,design,site);cache=joblib.load(folder/'prepared.joblib')
            data=cache['datasets']['PUBLIC_POWER'];roles=cache['roles']
            frame=pd.read_csv(folder/'predictions_all_models.csv')
            assert np.array_equal(pd.to_datetime(frame.datetime,utc=True),data.datetime)
            assert np.array_equal(frame.surface,data.surface) and np.allclose(frame.ape_true,data.ape,atol=1e-12,rtol=0)
            assert frame.groupby('datetime').split.nunique().eq(1).all()
            role_times={r:set(data.iloc[idx].datetime) for r,idx in roles.items()}
            for a in role_times:
                for b in role_times:
                    if a!=b:assert role_times[a].isdisjoint(role_times[b])
            other=cache['datasets']['NEDO_reference']
            assert data[['datetime','surface','ape']].equals(other[['datetime','surface','ape']])
            assert data.ape_samples_in_hour.ge(3).all()
            metrics=pd.read_csv(folder/'metrics_all_models.csv');assert len(metrics)==180
            for row in metrics.itertuples():
                part=frame.loc[frame.split==row.split]
                if row.surface!='ALL':part=part.loc[part.surface==row.surface]
                y,p=part.ape_true.to_numpy(),part[row.model].to_numpy()
                assert np.isfinite(p).all() and row.n==len(y)
                for field,value in calculate(y,p).items():
                    error=abs(value-getattr(row,field));assert error<1e-8;max_metric_error=max(max_metric_error,error)
                checked+=1
            design_matrix=E.F.beta_design(data)
            beta=cache['beta_model'].predict(cache['beta_scaler'].transform(design_matrix.to_numpy(float)))
            assert np.allclose(beta,cache['beta'],atol=1e-12,rtol=0)
            report=json.loads((folder/'fixed_beta_coefficients.json').read_text(encoding='utf-8'))
            original=report['original_unit_intercept']+sum(design_matrix[name].to_numpy()*c for name,c in report['original_unit_coefficients'].items())
            assert np.max(np.abs(original-beta))<1e-10
            beta_eval=beta.copy();available=np.isfinite(cache['beta_oof']);beta_eval[available]=cache['beta_oof'][available]
            assert np.max(np.abs(beta_eval-frame.fixed_beta_5.to_numpy()))<1e-12
            times=data.datetime.dt.as_unit('ns').astype('int64').to_numpy();surfaces=data.surface.to_numpy();hour=3600*10**9
            active=np.concatenate(list(roles.values()))
            for name in C.MODELS:
                model_folder=folder/name
                status=json.loads((model_folder/'job_status.json').read_text(encoding='utf-8'));assert status['status']=='complete'
                assert json.loads((model_folder/'explanation_status.json').read_text(encoding='utf-8'))['status']=='complete'
                prep=joblib.load(model_folder/'preprocessors.joblib');config=prep['config']
                source='NEDO_reference' if name.startswith('NEDO') else 'PUBLIC_POWER'
                assert config['data_sha256']==D.sha(folder/f'hourly_dataset_{source}.csv')
                history=pd.read_csv(model_folder/'training_history.csv')
                assert np.isfinite(history.to_numpy()).all() and history.epoch.to_list()==list(range(1,len(history)+1))
                training=json.loads((model_folder/'model_report.json').read_text(encoding='utf-8'))['training']
                assert training['epochs_executed']==len(history)
                assert abs(history.set_index('epoch').loc[training['best_epoch'],'validation_early_huber_standardized']-training['best_validation_loss'])<1e-12
                histories+=1
                archive=np.load(model_folder/'recursive_predictions.npz');raw,inputs=archive['raw'],archive['inputs']
                assert np.isfinite(raw[active]).all()
                reconstructed=inputs[:,-1,:-1]*prep['feature_scaler'].scale_+prep['feature_scaler'].mean_
                for role,idx in roles.items():
                    previous={(times[i],surfaces[i]):i for i in idx}
                    for lag in [1,2,3,6]:
                        column=config['feature_names'].index(f'residual_lag_{lag}h')
                        donor=np.array([previous.get((times[i]-lag*hour,surfaces[i]),-1) for i in idx])
                        expected=np.where(donor>=0,raw[np.maximum(donor,0)],0.)
                        error=np.max(np.abs(reconstructed[idx,column]-expected));max_lag_error=max(max_lag_error,float(error));assert error<2e-6
                        assert np.max(np.abs(reconstructed[idx,column+1]-(donor>=0)))<2e-6
                assert np.nanmax(np.abs(frame[name+'_raw'].to_numpy()-(beta_eval+raw)))<1e-12
                correction=raw.copy()
                for surface,p in prep['calibration'].items():
                    mask=surfaces==surface
                    correction[mask]=np.clip(raw[mask]*p['slope']+p['intercept'],-p['clip_abs'],p['clip_abs'])
                assert np.nanmax(np.abs(frame[name+'_cal'].to_numpy()-(beta_eval+correction)))<1e-12
                model=M.create_model(config['kind'],config['n_features'],config['n_physical'],config['sequence_length'])
                state=torch.load(model_folder/'model.pt',map_location='cpu',weights_only=False)
                assert state['fingerprint']==D.sha(model_folder/'config.json')
                model.load_state_dict(state['state_dict']);model.eval()
                idx=roles['test'][np.linspace(0,len(roles['test'])-1,64,dtype=int)]
                replay=M.predict(model,inputs[idx])*config['residual_std']+config['residual_mean']
                error=float(np.max(np.abs(replay-raw[idx])));max_replay_error=max(max_replay_error,error);assert error<2e-6
                for stem in ['true_vs_predicted','training_history','feature_permutation_importance','feature_effect_attribution_map']:
                    for ext in ['.png','.pdf']:assert (model_folder/(stem+ext)).stat().st_size>500
                assert json.loads((model_folder/'attribution_verification.json').read_text(encoding='utf-8'))['pipeline_reconstruction_max_abs_eV']<1e-5
                if config['kind']=='SelfAttention':
                    attention=np.load(model_folder/'attention_weights_test.npz')
                    assert np.allclose(attention['feature_attention'].sum(-1),1,atol=1e-5)
                    assert np.allclose(attention['temporal_current_query_attention'].sum(-1),1,atol=1e-5)
                    assert np.max(np.abs(attention['temporal_current_query_attention']*~attention['past_covariate_available'][:,None,:]))<1e-6
                if cache['reused']:
                    for filename in ['model.pt','preprocessors.joblib','recursive_predictions.npz']:
                        assert D.sha(model_folder/filename)==D.sha(D.OUTPUT/name/filename)
            if cache['reused']:
                old=pd.read_csv(D.OUTPUT/'predictions_all_models.csv')
                for name in ['fixed_beta_5',*[n+'_'+s for n in C.MODELS for s in ['raw','cal']]]:
                    assert np.allclose(old[name],frame[name],atol=1e-12,rtol=0,equal_nan=True)
            contexts[(design,site)]=(cache,frame,metrics)
    a,b=contexts[('common_hours','Tsukuba')],contexts[('common_hours','Naganuma')]
    assert a[1][['datetime','surface','split']].equals(b[1][['datetime','surface','split']])
    for role in a[0]['roles']:
        assert np.array_equal(a[0]['roles'][role],b[0]['roles'][role])
    all_metrics=pd.read_csv(output/'metrics_all_contexts.csv');assert len(all_metrics)==720
    test=all_metrics.loc[all_metrics.split.eq('test')].set_index(['design','site','surface','model'])
    paired=pd.read_csv(output/'calibration_comparison.csv');assert len(paired)==84
    for row in paired.itertuples():
        raw=test.loc[(row.design,row.site,row.surface,row.model+'_raw')];cal=test.loc[(row.design,row.site,row.surface,row.model+'_cal')]
        assert abs(row.delta_r2-(cal.r2-raw.r2))<1e-10
        assert abs(row.delta_mape_percentage_points-(cal.mape_pct-raw.mape_pct))<1e-10
    comparison=pd.read_csv(output/'site_comparison_all_models.csv');assert len(comparison)==90
    for row in comparison.itertuples():
        first=test.loc[(row.design,'Tsukuba',row.surface,row.model)];second=test.loc[(row.design,'Naganuma',row.surface,row.model)]
        for field in ['r2','mae_meV','rmse_meV','mape_pct']:
            assert abs(getattr(row,'delta_'+field)-(first[field]-second[field]))<1e-8
    intervals=pd.read_csv(output/'paired_week_block_bootstrap.csv');assert len(intervals)==182
    for row in intervals.itertuples():
        first=test.loc[(row.design,row.site,row.surface,row.model)];second=test.loc[(row.design,row.base_site,row.surface,row.baseline)]
        for field in ['r2','rmse_meV','mae_meV','mape_pct']:
            assert abs(getattr(row,'delta_'+field)-(first[field]-second[field]))<1e-8
            assert getattr(row,field+'_ci95_low')<=getattr(row,field+'_ci95_high')
    architectures=pd.read_csv(output/'training_and_architecture_summary.csv');assert len(architectures)==28
    for model,group in architectures.groupby('model'):
        for field in ['n_physical','n_features','sequence_length','parameters']:assert group[field].nunique()==1
    importance=pd.read_csv(output/'feature_permutation_importance_all_contexts.csv')
    for (design,site),(_,_,_) in contexts.items():
        for name in C.MODELS:
            saved=pd.read_csv(C.context_folder(output,design,site)/name/'feature_permutation_importance.csv')
            combined=importance.loc[(importance.design==design)&(importance.site==site)&(importance.model==name),saved.columns]
            pd.testing.assert_frame_equal(saved.reset_index(drop=True),combined.reset_index(drop=True),check_exact=False,atol=1e-12,rtol=0)
    images=0;links=0
    for path in output.rglob('*.png'):
        with Image.open(path) as image:image.verify()
        images+=1
    for path in output.rglob('*.md'):
        for relative in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            assert (path.parent/relative).is_file(),f'Missing image {path}: {relative}'
            links+=1
    assert (output/'評価.md').exists() and (output/'結果・考察.md').exists()
    result=dict(status='pass',metric_rows_recomputed=checked,max_metric_absolute_error=max_metric_error,
        calibration_pairs_recomputed=len(paired),site_comparisons_recomputed=len(comparison),bootstrap_point_differences_recomputed=len(intervals),
        histories_verified=histories,recursive_lag_max_abs_eV=max_lag_error,saved_model_replay_max_abs_eV=max_replay_error,
        common_timestamps_and_splits_identical=True,common_test_hours_per_surface=int(a[1].loc[(a[1].split=='test')&(a[1].surface=='HSR')].shape[0]),
        model_architecture_identical_to_previous_naganuma=True,original_inputs_unchanged=True,original_naganuma_models_predictions_unchanged=True,
        verified_png_count=images,verified_markdown_image_links=links)
    D.save_json(output/'verification.json',result)
    D.save_json(output/'input_integrity.json',dict(unchanged=True,sha256=plan['input_sha256']))
    snapshot=output/'analysis_code';snapshot.mkdir(exist_ok=True)
    code_names=['tsukuba_naganuma_10feature_data.py','tsukuba_naganuma_10feature_experiment.py',
        'tsukuba_naganuma_10feature_reporting.py','tsukuba_naganuma_10feature_verify.py',
        'naganuma_10feature_models.py','naganuma_10feature_experiment.py','naganuma_10feature_reporting.py',
        'naganuma_10feature_data.py','naganuma_f_to3_hourly.py',str(Path(E.F.__file__).resolve().relative_to(D.ROOT))]
    for name in code_names:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(D.ROOT/name,target)
    D.save_json(output/'execution_status.json',dict(status='complete',completed_utc=pd.Timestamp.now(tz='UTC'),
        new_training_jobs=21,reused_models=7,verification='pass',source_unchanged=True,code_sha256={name:D.sha(snapshot/name) for name in code_names}))
    (output/'進捗.md').write_text('# 完了\n\n21条件を新規学習し、長沼の7条件を再利用しました。全数値と入力保全の検算に合格。\n\n[評価](評価.md)\n\n[結果・考察](結果・考察.md)\n',encoding='utf-8')
    memory=D.ROOT/'研究引継ぎ用マスタープロンプト.md'
    text=memory.read_text(encoding='utf-8');left='<!-- TWO_SITE_THREE_MODELS_START -->';right='<!-- TWO_SITE_THREE_MODELS_END -->'
    block=f'''{left}
### つくば・長沼の10特徴量3モデル比較（2026-10-29完了・今後変更可能）

- 同じMLP/TCN/Self-Attention実装、特徴量・表面・OOF・自己予測残差履歴・校正有無で比較済み。主比較は両地点・両入力源・両面で共通の8,876時間にそろえ、Train6,213時間、Test各面1,326時間で両地点を学習し直した。β・標準化・NN・校正は地点別にTrain/CALで推定。転移・未知地点の評価ではない。
- つくば全有効時間のnative版も学習。長沼native版は旧重みと予測をそのまま再利用。native同士のTest時刻は異なり、主比較と区別する。β単体・5物理変数MLP対照も保存。新規21学習、再利用7条件。
- 出力は `比較実験_つくば長沼_10特徴量_MLP_TCN_Attention_20261029/評価.md` と `結果・考察.md`。全720行の指標、校正差84組、地点差90組、bootstrap差182組を独立検算した。両地点のDSと元の長沼結果は変更していない。
- Windowsが既存SciPy1.18のDLLを拒否したため、`.venv-site-comparison`でSciPy1.17.1を分離して実行。同じPyTorch2.13/NumPy2.5.1/scikit-learn1.9を使用し、長沼nativeβの一致を確認。セキュリティ設定は変更していない。
- データの分散・入力源対応・校正区間とTestの残差平均差を区別する。観測有効時間に依存する履歴欠測、実TSR角度や波長帯・分光装置の独立確認、単一seed、全国展開は引き続き検討事項。方針は暫定で変更可能。
{right}'''
    if left in text and right in text:text=text[:text.index(left)]+block+text[text.index(right)+len(right):]
    else:text=text.replace('**以下の本文には初期の3変数',block+'\n\n**以下の本文には初期の3変数',1)
    memory.write_text(text,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':verify()
