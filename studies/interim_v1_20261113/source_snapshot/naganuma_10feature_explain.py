"""Rebuild feature-effect figures from a completed saved model, without fitting."""
import argparse
import json

import joblib
import numpy as np
import torch

import naganuma_10feature_data as D
import naganuma_10feature_experiment as E
import naganuma_10feature_models as M
import naganuma_10feature_reporting as R


def explain(name, output=D.OUTPUT):
    torch.set_num_threads(1)
    frames,_=E.load_datasets(output)
    roles,_=E.F.unique_time_split(frames['PUBLIC_POWER'],E.F.Settings())
    lookup=E.sequence_lookup(frames['PUBLIC_POWER'],roles)
    folder=output/name
    processors=joblib.load(folder/'preprocessors.joblib')
    config=processors['config']
    model=M.create_model(config['kind'],config['n_features'],config['n_physical'],config['sequence_length'])
    state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict'])
    model.eval()
    archive=np.load(folder/'recursive_predictions.npz')
    beta=joblib.load(output/'fixed_beta_model.joblib')
    source='NEDO_reference' if name.startswith('NEDO') else 'PUBLIC_POWER'
    data=frames[source]
    prediction=beta['model'].predict(beta['scaler'].transform(E.F.beta_design(data).to_numpy(float)))
    report=json.loads((folder/'model_report.json').read_text(encoding='utf-8'))
    run=dict(model=model,data=data,static=E.static_features(data,prediction,config['n_physical']),
             scaler=processors['feature_scaler'],mean=config['residual_mean'],std=config['residual_std'],
             calibration=processors['calibration'],inputs=archive['inputs'],config=config,folder=folder,
             report=report,name=name,kind=config['kind'])
    R.configure_plotting()
    importance=R.permutation_importance(run,roles,lookup,beta['model'],beta['scaler'])
    R.importance_plot(run,importance)
    R.attribution_maps(run,roles,beta['model'],beta['scaler'])
    if config['kind']=='SelfAttention':
        R.attention_maps(run,roles)
    D.save_json(folder/'interpretation_complete.json',dict(model=name,completed_utc=D.pd.Timestamp.now(tz='UTC')))
    print(f'[EXPLANATION COMPLETE] {name}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    explain(args.model)
