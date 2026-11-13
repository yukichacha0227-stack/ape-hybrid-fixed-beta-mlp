"""Fit exactly the original Naganuma MLP/TCN/SelfAttention on two-site cohorts."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import naganuma_10feature_data as D
import naganuma_10feature_experiment as E
import tsukuba_naganuma_10feature_data as C


def fit_job(output,design,site,name):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    folder=C.context_folder(output,design,site)
    cache=joblib.load(folder/'prepared.joblib')
    for path,digest in cache['metadata']['source_sha256'].items():assert D.sha(path)==digest
    destination=folder/name
    if (destination/'job_status.json').exists():
        status=json.loads((destination/'job_status.json').read_text(encoding='utf-8'))
        assert status['version']==C.VERSION
        print('[ALREADY COMPLETE] '+str(destination),flush=True);return
    source='NEDO_reference' if name.startswith('NEDO') else 'PUBLIC_POWER'
    kind='MLP' if name=='CONTROL_5_MLP' else name[len(source)+1:]
    nphys=5 if name=='CONTROL_5_MLP' else 10
    data=cache['datasets'][source]
    predictions,run=E.fit_nn(folder,name,kind,data,cache['roles'],cache['lookup'],cache['beta_oof'],
        cache['beta'],cache['settings'],nphys)
    result=data[['datetime','surface','ape','ape_samples_in_hour']].copy()
    for key,value in predictions.items():result[key]=value
    result.to_csv(destination/'predictions.csv',index=False,encoding='utf-8-sig')
    metrics=E.metrics_all(data,cache['roles'],predictions)
    metrics.to_csv(destination/'metrics.csv',index=False,encoding='utf-8-sig')
    D.save_json(destination/'job_status.json',dict(version=C.VERSION,status='complete',design=design,site=site,
        name=name,completed_utc=pd.Timestamp.now(tz='UTC')))
    print(metrics.loc[(metrics.split=='test')&metrics.surface.ne('ALL'),['model','surface','r2','mape_pct']].to_string(index=False),flush=True)


def launch(output):
    plan=json.loads((output/'execution_plan.json').read_text(encoding='utf-8'))['new_training_jobs']
    # Start one short full job for early validation, then schedule expensive backbones first.
    first=next(p for p in plan if p==dict(design='native',site='Tsukuba',name='PUBLIC_POWER_MLP'))
    remaining=[p for p in plan if p!=first]
    remaining.sort(key=lambda p:(0 if p['name'].endswith('SelfAttention') else 1 if p['name'].endswith('TCN') else 2,
                                 0 if p['design']=='native' else 1,p['site'],p['name']))
    jobs=[first,*remaining]
    def call(job):
        tag='/'.join(job.values())
        log=output/job['design']/job['site']/(job['name']+'.log')
        print('[START] '+tag,flush=True)
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--output',str(output),
                '--job',job['design'],job['site'],job['name']],stdout=stream,stderr=subprocess.STDOUT,cwd=D.ROOT)
        if result.returncode:raise RuntimeError(f'{tag} exit {result.returncode}: {log}')
        print('[DONE] '+tag,flush=True)
    failures=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(call,p) for p in jobs]
        for future in as_completed(futures):
            try:future.result()
            except Exception as exc:failures.append(str(exc));print('[FAILED] '+str(exc),flush=True)
    if failures:raise RuntimeError('; '.join(failures))
    print('[ALL NEW MODELS COMPLETE]',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=C.OUTPUT)
    p.add_argument('--run-all',action='store_true')
    p.add_argument('--job',nargs=3,metavar=('DESIGN','SITE','MODEL'))
    args=p.parse_args()
    if args.job:fit_job(args.output,*args.job)
    if args.run_all:launch(args.output)


if __name__=='__main__':main()
