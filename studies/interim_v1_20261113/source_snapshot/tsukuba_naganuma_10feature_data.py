"""Read-only Tsukuba input adapter and fully time-matched two-site cohorts."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import requests

import naganuma_10feature_data as D
import naganuma_10feature_experiment as E

ROOT=D.ROOT
OUTPUT=ROOT/'比較実験_つくば長沼_10特徴量_MLP_TCN_Attention_20261029'
TSUKUBA_SOURCE=ROOT/'入力/Tsukuba_DS.csv'
SOURCES=['PUBLIC_POWER','NEDO_reference']
KINDS=['MLP','TCN','SelfAttention']
MODELS=['CONTROL_5_MLP']+[s+'_'+k for s in SOURCES for k in KINDS]
COORDINATE=(36+3/60,140+8/60)
COORDINATE_SOURCE='https://kaken.nii.ac.jp/ja/file/KAKENHI-PROJECT-21K05688/21K05688seika.pdf'
VERSION='two-site-hourly-three-backbones-v1'


def load_tsukuba_source():
    frame=pd.read_csv(TSUKUBA_SOURCE,encoding='utf-8-sig',skipinitialspace=True)
    frame.columns=frame.columns.str.strip()
    frame=frame.rename(columns={'TO3':'to3'})
    frame['datetime']=pd.to_datetime(frame.datetime,utc=True).dt.as_unit('ns')
    for name in set(D.N.ATMOSPHERE)|{'am','SiteNum','ape_hsr','ape_tsr','remark_hsr','remark_tsr'}:
        frame[name]=pd.to_numeric(frame[name],errors='coerce')
    assert frame.SiteNum.eq(301).all()
    assert not frame.duplicated(['datetime','SiteNum']).any()
    return frame.sort_values('datetime').reset_index(drop=True)


def fetch_power(year,cache):
    path=cache/f'NASA_POWER_tsukuba_{year}_UTC.json'
    params=dict(parameters=','.join(D.POWER_PARAMETERS),community='RE',latitude=COORDINATE[0],longitude=COORDINATE[1],
        start=f'{year}0101',end=f'{year}1231',format='JSON',**{'time-standard':'UTC'})
    if path.exists():obj=json.loads(path.read_text(encoding='utf-8'))
    else:
        for attempt in range(5):
            try:
                response=requests.get(D.POWER_URL,params=params,timeout=(20,120))
                response.raise_for_status();obj=response.json()
                if not set(D.POWER_PARAMETERS).issubset(obj.get('properties',{}).get('parameter',{})):
                    raise ValueError('POWER parameters missing')
                path.write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')
                print(f'[POWER] Tsukuba {year} {path.stat().st_size:,} bytes',flush=True)
                break
            except (requests.RequestException,ValueError):
                if attempt==4:raise
                time.sleep(min(30,3*(attempt+1)))
    frame=pd.DataFrame(obj['properties']['parameter'])
    frame.index=pd.to_datetime(frame.index,format='%Y%m%d%H',utc=True)+pd.Timedelta(minutes=30)
    frame.index.name='datetime'
    frame=frame.replace(obj['header'].get('fill_value',-999),np.nan).astype(float)
    for name in D.POWER_PARAMETERS[:3]:
        assert obj['parameters'][name]['units'] in ['Wh/m^2','Wh/m2','W/m^2','W/m2']
    return frame,dict(year=year,request_parameters=params,header=obj['header'],parameters=obj['parameters'],
        geometry=obj.get('geometry'),file=str(path),sha256=D.sha(path))


def prepare_tsukuba(output):
    cache=output/'public_data';cache.mkdir(exist_ok=True,parents=True)
    source=load_tsukuba_source()
    base,report=D.N.prepare_from_source(source,E.F.Settings())
    report.update(experiment='Tsukuba, identical paired-DS hourly algorithm',
        definition='arithmetic hourly means from read-only Tsukuba_DS',
        source_cohort_warning='Tsukuba_DS is already paired at 10-minute timestamps; unpaired spectra are not reintroduced')
    # Validate supplied DS targets against the independently stored original targets.
    audits={}
    for surface,path in [('HSR',E.F.DEFAULT_HSR),('TSR',E.F.DEFAULT_TSR)]:
        raw,meta=E.F.load_target(path,surface)
        target=source[['datetime',f'ape_{surface.lower()}']].merge(raw[['datetime','APE']],on='datetime',validate='one_to_one')
        error=float(np.max(np.abs(target[f'ape_{surface.lower()}']-target.APE)))
        assert len(target)==len(source) and error<1e-10
        audits[surface]=dict(n=len(target),max_abs_eV=error,original_target_sha256=meta['sha256'])
    columns={'horizontal_irradiance_w_m2':'GHI','diffuse_fraction_horizontal':'diffuse_fraction',
        'air_temperature_c':'temperature_C','relative_humidity_pct':'RH_pct'}
    flags=['horizontal_irradiance_remark','diffuse_fraction_horizontal_remark','air_temperature_remark','relative_humidity_remark']
    for name in [*columns,*flags]:source[name]=pd.to_numeric(source[name],errors='coerce')
    valid=np.isfinite(source[list(columns)]).all(axis=1)&source[flags].isin([1,2]).all(axis=1)
    valid&=source.horizontal_irradiance_w_m2.gt(0)&source.relative_humidity_pct.between(0,100)&source.diffuse_fraction_horizontal.between(0,1)
    weather=source.loc[valid,['datetime',*columns]].rename(columns=columns).copy()
    weather['datetime']=E.F.hourly_center_utc(weather.datetime)
    weather['DHI_reconstructed']=weather.GHI*weather.diffuse_fraction
    weather=weather.groupby('datetime').agg(GHI=('GHI','mean'),DHI=('DHI_reconstructed','mean'),
        temperature_C=('temperature_C','mean'),RH_pct=('RH_pct','mean'),weather_samples=('GHI','size')).reset_index()
    weather=weather.loc[weather.weather_samples.ge(3)].copy()
    with ThreadPoolExecutor(max_workers=3) as pool:
        obtained=list(pool.map(lambda year:fetch_power(year,cache),range(2011,2016)))
    power=pd.concat([x[0] for x in obtained]).sort_index()
    assert power.index.is_unique
    power.to_csv(cache/'NASA_POWER_hourly_2011_2015.csv',encoding='utf-8-sig')
    public=power.rename(columns={'ALLSKY_SFC_SW_DWN':'GHI','ALLSKY_SFC_SW_DIFF':'DHI',
        'TOA_SW_DWN':'G0','T2M':'temperature_C','RH2M':'RH_pct'}).reset_index()
    public['Kt']=public.GHI/public.G0;public['diffuse_fraction']=public.DHI/public.GHI
    weather=weather.merge(public[['datetime','G0']],on='datetime',validate='one_to_one')
    weather['Kt']=weather.GHI/weather.G0;weather['diffuse_fraction']=weather.DHI/weather.GHI
    datasets={}
    for name,met in [('PUBLIC_POWER',public),('NEDO_reference',weather)]:
        ok=np.isfinite(met[D.EXTRAS]).all(axis=1)&met.GHI.gt(1)&met.G0.gt(20)&met.Kt.between(0,2)
        ok&=met.diffuse_fraction.between(0,1.02)&met.RH_pct.between(0,100)
        met=met.loc[ok].copy();met['diffuse_fraction']=met.diffuse_fraction.clip(0,1)
        datasets[name]=base.merge(met[['datetime',*D.EXTRAS]],on='datetime',validate='many_to_one')
    common=set(datasets['PUBLIC_POWER'].datetime)&set(datasets['NEDO_reference'].datetime)
    for name,data in datasets.items():
        data=data.loc[data.datetime.isin(common)].sort_values(['datetime','surface']).reset_index(drop=True)
        assert data.groupby('datetime').surface.nunique().eq(2).all()
        datasets[name]=data
    return datasets,dict(base_hourly_report=report,original_target_audit=audits,source=str(TSUKUBA_SOURCE),
        coordinate=list(COORDINATE),coordinate_reference=COORDINATE_SOURCE,
        public_data_metadata=[x[1] for x in obtained],common_hours=len(common),source_rows=len(source),
        weather_quality='same as Naganuma: simultaneous valid GHI/diffuse/T/RH, remarks 1/2, >=3 records/hour',
        surface_codes='HSR 0 / TSR 32 inherited coding, not verified installation angles')


def context_folder(output,design,site):return output/design/site


def matched_site_frames(pairs):
    common=set.intersection(*(set(frames[source].datetime) for frames in pairs.values() for source in SOURCES))
    result={site:{s:df.loc[df.datetime.isin(common)].sort_values(['datetime','surface']).reset_index(drop=True)
                  for s,df in frames.items()} for site,frames in pairs.items()}
    expected=None
    for frames in result.values():
        for df in frames.values():
            assert not df.duplicated(['datetime','surface']).any()
            assert df.groupby('datetime').surface.nunique().eq(2).all()
            keys=df[['datetime','surface']]
            if expected is None:expected=keys
            else:assert keys.equals(expected)
    return result,sorted(common)


def prepare_context(folder,datasets,metadata,reused=False):
    folder.mkdir(parents=True,exist_ok=True)
    for name,data in datasets.items():
        data.to_csv(folder/f'hourly_dataset_{name}.csv',index=False,encoding='utf-8-sig')
    settings=E.F.Settings();data=datasets['PUBLIC_POWER']
    roles,split_report=E.F.unique_time_split(data,settings)
    oof,beta,report,model,scaler=E.F.fit_beta_oof(data,roles,settings)
    if reused:
        old=joblib.load(D.OUTPUT/'fixed_beta_model.joblib')
        prediction=old['model'].predict(old['scaler'].transform(E.F.beta_design(data).to_numpy(float)))
        assert np.allclose(beta,prediction,rtol=0,atol=1e-12)
    D.save_json(folder/'split_report.json',split_report)
    D.save_json(folder/'fixed_beta_coefficients.json',report)
    D.save_json(folder/'data_preparation_report.json',metadata)
    joblib.dump(dict(model=model,scaler=scaler),folder/'fixed_beta_model.joblib')
    joblib.dump(dict(datasets=datasets,roles=roles,lookup=E.sequence_lookup(data,roles),beta_oof=oof,beta=beta,
        beta_model=model,beta_scaler=scaler,settings=settings,metadata=metadata,reused=reused),folder/'prepared.joblib')
    return split_report


def prepare(output=OUTPUT):
    output.mkdir(parents=True,exist_ok=True)
    before={str(p):D.sha(p) for p in [TSUKUBA_SOURCE,D.SOURCE,D.OUTPUT/'predictions_all_models.csv']}
    tsukuba,report=prepare_tsukuba(output)
    naganuma,naga_report=E.load_datasets(D.OUTPUT)
    pairs={'Tsukuba':tsukuba,'Naganuma':naganuma}
    matched,joint=matched_site_frames(pairs)
    D.save_json(output/'data_preparation_report.json',dict(tsukuba=report,naganuma_reference=naga_report,
        common_hours=len(joint),cohort='valid in both sites, both surfaces and both input-source conditions'))
    contexts={}
    for site,datasets in pairs.items():
        for design in ['native','common_hours']:
            subset={s:df.copy() for s,df in (datasets if design=='native' else matched[site]).items()}
            folder=context_folder(output,design,site)
            reused=design=='native' and site=='Naganuma'
            meta=dict(site=site,design=design,common_hours=len(joint),reused_original_models=reused,
                      parent_preparation_report=str(output/'data_preparation_report.json'),source_sha256=before)
            contexts[f'{design}/{site}']=prepare_context(folder,subset,meta,reused)
    for path,digest in before.items():assert D.sha(path)==digest
    plan=[dict(design=d,site=s,name=name) for d in ['native','common_hours'] for s in ['Tsukuba','Naganuma']
          if not(d=='native' and s=='Naganuma') for name in MODELS]
    D.save_json(output/'execution_plan.json',dict(status='prepared',version=VERSION,new_training_jobs=plan,
        previous_naganuma_models_reused=MODELS,settings=asdict(E.F.Settings()),physical_features=D.PHYSICAL,
        contexts=contexts,input_sha256=before,
        environment=dict(python_executable=__import__('sys').executable,
            versions={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','scikit-learn','pandas','matplotlib']},
            note='Isolated scipy 1.17.1 runtime because primary scipy 1.18 native DLL is blocked by Windows application control; other analysis packages are reused unchanged. No security settings changed.'),
        primary_design='common_hours: both sites are retrained on identical UTC timestamps, samples and split assignments',
        reference_design='native: Tsukuba uses its own same 70/15/15 rule; existing Naganuma models reused unchanged',
        interpretation='two independently supervised local models, not cross-site transfer or unknown-site validation'))
    print(f'[PREPARED] Tsukuba {len(tsukuba["PUBLIC_POWER"])//2} h; Naganuma {len(naganuma["PUBLIC_POWER"])//2} h; joint {len(joint)} h; {len(plan)} new fits',flush=True)


if __name__=='__main__':prepare()
