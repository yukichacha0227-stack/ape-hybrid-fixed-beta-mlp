"""Hourly ten-feature Naganuma experiment; preserves all source CSVs.

Run: .venv/Scripts/python.exe -u naganuma_10feature_experiment.py
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import torch

import naganuma_10feature_data as D
import naganuma_10feature_models as M

F = D.N.F
LAGS = (1, 2, 3, 6)
SEQUENCE_HOURS = 7
VERSION = '10feature-causal-v1'


def load_datasets(output):
    if not (output / 'data_preparation_report.json').exists():
        return D.prepare(output)
    report = json.loads((output / 'data_preparation_report.json').read_text(encoding='utf-8'))
    if report['source_sha256_before'] != D.sha(D.SOURCE):
        raise ValueError('Original source has changed since preparation')
    frames = {}
    for source in ['PUBLIC_POWER', 'NEDO_reference']:
        frame = pd.read_csv(output / f'hourly_dataset_{source}.csv', encoding='utf-8-sig')
        frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True).dt.as_unit('ns')
        frames[source] = frame
    return frames, report


def static_features(data, beta, n_physical):
    original = F.residual_static_features(data, beta)
    physical_names = D.PHYSICAL[:5] if n_physical == 5 else D.PHYSICAL
    physical = pd.DataFrame(index=data.index)
    for name in physical_names:
        physical[name] = original[name] if name in original else data[name]
    return pd.concat([physical, original.drop(columns=D.PHYSICAL[:5])], axis=1)


def sequence_lookup(data, roles):
    n = len(data)
    result = np.full((n, SEQUENCE_HOURS), -1, dtype=np.int64)
    times = data['datetime'].dt.as_unit('ns').astype('int64').to_numpy()
    surface = data['surface'].map({'HSR': 0, 'TSR': 1}).to_numpy()
    hour_ns = 3600 * 10**9
    for indices in roles.values():
        mapping = {(times[i], surface[i]): i for i in indices}
        for offset in range(SEQUENCE_HOURS):
            lag = SEQUENCE_HOURS-1-offset
            result[indices, offset] = [mapping.get((times[i]-lag*hour_ns, surface[i]), -1) for i in indices]
    return result


def tensor_sequences(scaled, lookup):
    n, length = lookup.shape
    result = np.zeros((n, length, scaled.shape[1]+1), dtype=np.float32)
    present = lookup >= 0
    result[:, :, :-1][present] = scaled[lookup[present]]
    result[:, :, -1] = present
    return result


def observed_lags(data, static, residual):
    """Numerically equivalent to F.add_observed_lags, without repeated iloc."""
    result=static.copy()
    times=data['datetime'].dt.as_unit('ns').astype('int64').to_numpy()
    surfaces=data['surface'].map({'HSR':0,'TSR':1}).to_numpy()
    history=[{},{}]
    for stamp,surface,value in zip(times,surfaces,residual):
        if np.isfinite(value):
            history[surface][stamp]=float(value)
    for lag in LAGS:
        previous=times-lag*3600*10**9
        result[f'residual_lag_{lag}h']=[history[s].get(t,0.) for t,s in zip(previous,surfaces)]
        result[f'residual_lag_{lag}h_available']=[float(t in history[s]) for t,s in zip(previous,surfaces)]
    return result


def recursive_inference(model, data, roles, static, scaler, residual_mean, residual_std, lookup):
    n, p = len(data), len(scaler.mean_)
    empty = F.add_empty_lags(static, LAGS).to_numpy(float)
    raw = np.full(n, np.nan)
    scaled_rows = np.zeros((n, p), dtype=np.float32)
    evaluation = np.zeros((n, SEQUENCE_HOURS, p+1), dtype=np.float32)
    times = data['datetime'].dt.as_unit('ns').astype('int64').to_numpy()
    surfaces = data['surface'].map({'HSR': 0, 'TSR': 1}).to_numpy()
    hour_ns = 3600*10**9
    for role, indices in roles.items():
        histories = [{}, {}]
        unique = np.unique(times[indices])
        groups = {stamp: [] for stamp in unique}
        for i in indices:
            groups[times[i]].append(i)
        role_start = time.perf_counter()
        for stamp in unique:
            current = np.asarray(groups[stamp], dtype=int)
            values = empty[current].copy()
            for row, i in enumerate(current):
                history = histories[surfaces[i]]
                for j, lag in enumerate(LAGS):
                    previous = stamp-lag*hour_ns
                    if previous in history:
                        values[row, -8+2*j] = history[previous]
                        values[row, -8+2*j+1] = 1.
            scaled_rows[current] = (values-scaler.mean_)/scaler.scale_
            seq = np.zeros((len(current), SEQUENCE_HOURS, p+1), dtype=np.float32)
            previous = lookup[current]
            present = previous >= 0
            seq[:, :, :-1][present] = scaled_rows[previous[present]]
            seq[:, :, -1] = present
            with torch.no_grad():
                prediction = model(torch.from_numpy(seq)).numpy()*residual_std+residual_mean
            evaluation[current] = seq
            raw[current] = prediction
            for i, value in zip(current, prediction):
                histories[surfaces[i]][stamp] = float(value)
        print(f'[INFERENCE {role}] {len(indices):,} rows {time.perf_counter()-role_start:.1f}s', flush=True)
    return raw, evaluation


def fit_nn(output, name, kind, data, roles, lookup, beta_oof, beta_prediction, settings, n_physical=10):
    folder = output / name
    folder.mkdir(exist_ok=True)
    static = static_features(data, beta_prediction, n_physical)
    y = data['ape'].to_numpy(float)
    train_idx = roles['train_oof']
    available = train_idx[np.isfinite(beta_oof[train_idx])]
    residual = np.full(len(data), np.nan)
    residual[available] = y[available]-beta_oof[available]
    teacher = observed_lags(data, static, residual)
    scaler = StandardScaler().fit(teacher.iloc[available])
    mean, std = float(residual[available].mean()), float(residual[available].std())
    assert std > 1e-10
    config = dict(version=VERSION, name=name, kind=kind, n_physical=n_physical,
                  n_features=teacher.shape[1], feature_names=teacher.columns.tolist(),
                  sequence_length=SEQUENCE_HOURS, settings=asdict(settings),
                  data_sha256=D.sha(output / ('hourly_dataset_NEDO_reference.csv' if name.startswith('NEDO') else 'hourly_dataset_PUBLIC_POWER.csv')),
                  residual_mean=mean, residual_std=std)
    if kind=='SelfAttention':
        config['architecture']='surface-conditioned-current-feature-attention-and-causal-temporal-attention-v2'
    D.save_json(folder / 'config.json', config)
    fingerprint = D.sha(folder / 'config.json')
    torch.manual_seed(settings.seed)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    model = M.create_model(kind, teacher.shape[1], n_physical, SEQUENCE_HOURS)
    checkpoint = folder / 'model.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['fingerprint'] != fingerprint:
            raise ValueError(f'Checkpoint configuration mismatch: {name}')
        model.load_state_dict(saved['state_dict'])
        training = saved['training']
        print(f'[RESUME] {name} trained checkpoint', flush=True)
    else:
        all_train = tensor_sequences(scaler.transform(teacher), lookup)
        early = tensor_sequences(scaler.transform(F.add_empty_lags(static, LAGS)), lookup)
        history, training = M.train(model, all_train[available], (residual[available]-mean)/std,
                                    early[roles['validation_early']],
                                    (y[roles['validation_early']]-beta_prediction[roles['validation_early']]-mean)/std,
                                    settings, name)
        history.to_csv(folder / 'training_history.csv', index=False, encoding='utf-8-sig')
        torch.save(dict(state_dict=model.state_dict(), fingerprint=fingerprint, training=training), checkpoint)
    model.eval()
    if (folder / 'recursive_predictions.npz').exists():
        archive = np.load(folder / 'recursive_predictions.npz')
        raw = archive['raw']
        evaluation = archive['inputs']
    else:
        raw, evaluation = recursive_inference(model, data, roles, static, scaler, mean, std, lookup)
        np.savez_compressed(folder / 'recursive_predictions.npz', raw=raw, inputs=evaluation)
    calibrated, calibration = F.fit_surface_affine(data, y-beta_prediction, raw, roles['validation_calibration'],
                                                  slope_bounds=(0., 1.5), correction_clip=True)
    beta_evaluation = beta_prediction.copy()
    beta_evaluation[available] = beta_oof[available]
    predictions = {name+'_raw': beta_evaluation+raw, name+'_cal': beta_evaluation+calibrated}
    report = dict(training=training, calibration=calibration, n_train_oof_rows=len(available),
                  train_diagnostic_only=True, early_stopping='empty residual lags, same reference F protocol',
                  recursive_inference='surface-specific predicted residuals, reset for every split; no observed APE at inference',
                  temporal_context='current plus 6 exact elapsed hours; missing/past split positions padded and explicitly masked',
                  attention='native feature-token 10x10 plus temporal current-query weights' if kind=='SelfAttention' else 'no native attention; use attribution and permutation maps')
    D.save_json(folder / 'model_report.json', report)
    D.save_json(folder / 'architecture.json',dict(module=str(model),parameters=sum(p.numel() for p in model.parameters())))
    joblib.dump(dict(feature_scaler=scaler, config=config, training=training, calibration=calibration), folder / 'preprocessors.joblib')
    return predictions, dict(model=model, data=data, static=static, scaler=scaler, mean=mean, std=std,
                             calibration=calibration, inputs=evaluation, config=config, folder=folder,
                             report=report, name=name, kind=kind)


def metrics_all(data, roles, predictions):
    metrics = F.metrics_table(data, roles, predictions)
    rows = []
    for role, indices in roles.items():
        for surface in ['ALL', 'HSR', 'TSR']:
            idx = indices if surface=='ALL' else indices[data.iloc[indices]['surface'].eq(surface).to_numpy()]
            for name, values in predictions.items():
                valid = np.isfinite(values[idx])
                mape = 100*np.mean(np.abs(values[idx][valid]-data['ape'].to_numpy()[idx][valid])/data['ape'].to_numpy()[idx][valid])
                rows.append(dict(split=role, surface=surface, model=name, mape_pct=mape))
    return metrics.merge(pd.DataFrame(rows), on=['split', 'surface', 'model'], validate='one_to_one')


def smoke_test():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for kind in ['MLP', 'TCN', 'SelfAttention']:
        model = M.create_model(kind, 35, 10, 7)
        x = torch.randn(8, 7, 36)
        x[:, :, -1] = 1
        x[0, :6] = 0
        y = model(x)
        assert y.shape==(8,) and torch.isfinite(y).all()
        y.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        if kind=='SelfAttention':
            model.capture_attention = True
            with torch.no_grad():
                model(x)
            assert torch.isfinite(model.last_feature_attention).all()
            assert torch.allclose(model.last_feature_attention.sum(-1), torch.ones(8, 2, 10), atol=1e-5)
            assert model.last_temporal_attention[0, :, :6].abs().max() < 1e-7
        print(f'[SMOKE PASS] {kind}: finite forward/backward and masks', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=D.OUTPUT)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--skip-interpretation', action='store_true')
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    datasets, prep = load_datasets(output)
    before = D.sha(D.SOURCE)
    data = datasets['PUBLIC_POWER']
    settings = F.Settings(max_epochs=args.epochs)
    roles, split_report = F.unique_time_split(data, settings)
    D.save_json(output / 'split_report.json', split_report)
    lookup = sequence_lookup(data, roles)
    assert np.all(lookup[np.concatenate(list(roles.values())), -1]>=0)
    beta_oof, beta_prediction, beta_report, beta_model, beta_scaler = F.fit_beta_oof(data, roles, settings)
    D.save_json(output / 'fixed_beta_coefficients.json', beta_report)
    joblib.dump(dict(model=beta_model, scaler=beta_scaler), output / 'fixed_beta_model.joblib')
    beta_evaluation = beta_prediction.copy()
    available = np.isfinite(beta_oof)
    beta_evaluation[available] = beta_oof[available]
    predictions = {'fixed_beta_5': beta_evaluation}
    runs = {}
    plan = [('CONTROL_5_MLP', 'MLP', data, 5)]
    for source in ['PUBLIC_POWER', 'NEDO_reference']:
        plan.extend((source+'_'+kind, kind, datasets[source], 10) for kind in ['MLP', 'TCN', 'SelfAttention'])
    D.save_json(output / 'execution_plan.json', dict(status='running', version=VERSION,
                 model_order=[p[0] for p in plan], settings=asdict(settings), source_sha256=before,
                 versions=dict(torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__,
                               sklearn=importlib.metadata.version('scikit-learn'),matplotlib=importlib.metadata.version('matplotlib'),
                               python=sys.version,torch_cpu_threads=torch.get_num_threads()),
                 scope='Naganuma within-site chronological evaluation; not five-site spatial generalisation'))
    for name, kind, frame, n_physical in plan:
        print(f'\n[START] {name}', flush=True)
        values, run = fit_nn(output, name, kind, frame, roles, lookup, beta_oof, beta_prediction, settings, n_physical)
        predictions.update(values)
        runs[name] = run
        partial = metrics_all(data, roles, predictions)
        partial.to_csv(output / 'metrics_all_models.csv', index=False, encoding='utf-8-sig')
        result = partial.loc[partial['split'].eq('test') & partial['model'].eq(name+'_cal') & partial['surface'].ne('ALL'), ['surface', 'r2', 'rmse_meV', 'mape_pct']]
        print(result.to_string(index=False), flush=True)
    for source, frame in datasets.items():
        static = static_features(frame, beta_prediction, 10).drop(columns='beta_prediction')
        hgb = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=31, learning_rate=.05,
                                            l2_regularization=1., early_stopping=False, random_state=42)
        hgb.fit(static.iloc[roles['train_oof']], frame.iloc[roles['train_oof']]['ape'])
        raw = hgb.predict(static)
        cal, calibration = F.fit_surface_affine(frame, frame['ape'].to_numpy(), raw, roles['validation_calibration'],
                                                slope_bounds=(.5, 1.5), correction_clip=False)
        predictions[source+'_HGB_raw'] = raw
        predictions[source+'_HGB_cal'] = cal
        joblib.dump(dict(model=hgb, calibration=calibration, features=static.columns.tolist()), output / (source+'_HGB.joblib'))
    metrics = metrics_all(data, roles, predictions)
    metrics.to_csv(output / 'metrics_all_models.csv', index=False, encoding='utf-8-sig')
    labels = np.full(len(data), 'purged', dtype=object)
    for role, idx in roles.items():
        labels[idx] = role
    result = data[['datetime', 'surface', 'ape', 'ape_samples_in_hour']].rename(columns={'ape': 'ape_true'}).copy()
    result['split'] = labels
    for name, values in predictions.items():
        result[name] = values
    result.to_csv(output / 'predictions_all_models.csv', index=False, encoding='utf-8-sig')
    # Remaining scientific figures, diagnostics and report are generated from saved runs.
    import naganuma_10feature_reporting as R
    R.finish(output, datasets, prep, roles, lookup, beta_model, beta_scaler, predictions, runs, metrics,
             interpret=not args.skip_interpretation)
    after = D.sha(D.SOURCE)
    assert before==after, 'Original CSV integrity failure'
    D.save_json(output / 'input_integrity.json', dict(before_sha256=before, after_sha256=after, unchanged=True))
    D.save_json(output / 'execution_status.json', dict(status='complete', models=list(predictions),
                 source_unchanged=True, completed_utc=pd.Timestamp.now(tz='UTC'), code_sha256={
                     p:D.sha(D.ROOT/p) for p in ['naganuma_10feature_data.py', 'naganuma_10feature_models.py',
                                                'naganuma_10feature_experiment.py', 'naganuma_10feature_reporting.py']}))
    print(f'[COMPLETE] {output}', flush=True)


if __name__=='__main__':
    main()
