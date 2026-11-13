"""Read-only, hourly Naganuma data and cached public NASA POWER inputs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

import naganuma_f_to3_hourly as N

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / '入力/長沼/Naganuma_DS.csv'
OUTPUT = ROOT / '入力/長沼/APE予測_10特徴量_MLP_TCN_Attention_20261028'
LATITUDE, LONGITUDE = 43.05, 141.75
COORDINATE_SOURCE = 'https://kaken.nii.ac.jp/ja/file/KAKENHI-PROJECT-21K05688/21K05688seika.pdf'
POWER_URL = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
POWER_PARAMETERS = ['ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DIFF', 'TOA_SW_DWN', 'T2M', 'RH2M']
EXTRAS = ['GHI', 'Kt', 'diffuse_fraction', 'temperature_C', 'RH_pct']
PHYSICAL = ['AM', 'omega', 'AOD550', 'TOTANGSTR', 'TO3', *EXTRAS]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    N.F.atomic_json(Path(path), obj)


def fetch_year(year: int, cache: Path):
    path = cache / f'NASA_POWER_naganuma_{year}_UTC.json'
    params = dict(parameters=','.join(POWER_PARAMETERS), community='RE', latitude=LATITUDE,
                  longitude=LONGITUDE, start=f'{year}0101', end=f'{year}1231',
                  format='JSON', **{'time-standard': 'UTC'})
    if path.exists():
        obj = json.loads(path.read_text(encoding='utf-8'))
    else:
        for attempt in range(4):
            try:
                response = requests.get(POWER_URL, params=params, timeout=(20, 120))
                response.raise_for_status()
                obj = response.json()
                if not set(POWER_PARAMETERS).issubset(obj.get('properties', {}).get('parameter', {})):
                    raise ValueError(f'POWER missing parameters: {obj.get("messages")}')
                path.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
                print(f'[NASA POWER] {year} downloaded {path.stat().st_size:,} bytes', flush=True)
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
    frame = pd.DataFrame(obj['properties']['parameter'])
    frame.index = pd.to_datetime(frame.index, format='%Y%m%d%H', utc=True) + pd.Timedelta(minutes=30)
    frame.index.name = 'datetime'
    fill = obj['header'].get('fill_value', -999)
    frame = frame.replace(fill, np.nan).astype(float)
    return frame, dict(year=year, request_parameters=params, header=obj['header'], parameters=obj['parameters'],
                       geometry=obj.get('geometry'), sha256=sha(path), file=str(path),
                       downloaded_or_checked_utc=datetime.now(timezone.utc).isoformat())


def prepare(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'public_data'
    cache.mkdir(exist_ok=True)
    before = sha(SOURCE)
    stat = SOURCE.stat()
    source = N.read_source(SOURCE)
    base, base_report = N.prepare_from_source(source, N.F.Settings())
    columns = {
        'horizontal_irradiance_w_m2': 'GHI',
        'diffuse_fraction_horizontal': 'diffuse_fraction',
        'air_temperature_c': 'temperature_C',
        'relative_humidity_pct': 'RH_pct',
    }
    flags = ['horizontal_irradiance_remark', 'diffuse_fraction_horizontal_remark',
             'air_temperature_remark', 'relative_humidity_remark']
    for c in [*columns, *flags]:
        source[c] = pd.to_numeric(source[c], errors='coerce')
    valid = np.isfinite(source[list(columns)]).all(axis=1) & source[flags].isin([1, 2]).all(axis=1)
    valid &= source['horizontal_irradiance_w_m2'].gt(0)
    valid &= source['relative_humidity_pct'].between(0, 100)
    valid &= source['diffuse_fraction_horizontal'].between(0, 1)
    weather = source.loc[valid, ['datetime', *columns]].rename(columns=columns).copy()
    weather['datetime'] = N.F.hourly_center_utc(weather['datetime'])
    weather['DHI_reconstructed'] = weather['GHI'] * weather['diffuse_fraction']
    weather = weather.groupby('datetime').agg(GHI=('GHI', 'mean'), DHI=('DHI_reconstructed', 'mean'),
                                              temperature_C=('temperature_C', 'mean'), RH_pct=('RH_pct', 'mean'),
                                              weather_samples=('GHI', 'size')).reset_index()
    weather = weather.loc[weather['weather_samples'].ge(3)].copy()
    with ThreadPoolExecutor(max_workers=3) as pool:
        retrieved = list(pool.map(lambda year: fetch_year(year, cache), range(2011, 2016)))
    power = pd.concat([r[0] for r in retrieved]).sort_index()
    assert power.index.is_unique
    power.to_csv(cache / 'NASA_POWER_hourly_2011_2015.csv', encoding='utf-8-sig')
    # Hourly Wh/m2 / one hour is numerically equal to the hourly mean W/m2.
    for _, meta in retrieved:
        for name in POWER_PARAMETERS[:3]:
            if meta['parameters'][name]['units'] not in ['Wh/m^2', 'Wh/m2', 'W/m^2', 'W/m2']:
                raise ValueError(f'Unrecognised hourly radiation unit: {meta["parameters"][name]}')
    public = power.rename(columns={'ALLSKY_SFC_SW_DWN': 'GHI', 'ALLSKY_SFC_SW_DIFF': 'DHI',
                                  'TOA_SW_DWN': 'G0', 'T2M': 'temperature_C', 'RH2M': 'RH_pct'}).reset_index()
    public['diffuse_fraction'] = public['DHI'] / public['GHI']
    public['Kt'] = public['GHI'] / public['G0']
    weather = weather.merge(public[['datetime', 'G0']], on='datetime', how='inner', validate='one_to_one')
    weather['Kt'] = weather['GHI'] / weather['G0']
    weather['diffuse_fraction'] = weather['DHI'] / weather['GHI']
    datasets = {}
    for name, met in [('NEDO_reference', weather), ('PUBLIC_POWER', public)]:
        ok = np.isfinite(met[EXTRAS]).all(axis=1)
        ok &= met['GHI'].gt(1) & met['G0'].gt(20) & met['Kt'].between(0, 2)
        ok &= met['diffuse_fraction'].between(0, 1.02) & met['RH_pct'].between(0, 100)
        met = met.loc[ok].copy()
        met['diffuse_fraction'] = met['diffuse_fraction'].clip(0, 1)
        datasets[name] = base.merge(met[['datetime', *EXTRAS]], on='datetime', how='inner', validate='many_to_one')
    common = sorted(set(datasets['NEDO_reference']['datetime']) & set(datasets['PUBLIC_POWER']['datetime']))
    for name in datasets:
        datasets[name] = datasets[name].loc[datasets[name]['datetime'].isin(common)].sort_values(['datetime', 'surface']).reset_index(drop=True)
        assert datasets[name].groupby('datetime')['surface'].nunique().eq(2).all()
        assert np.isfinite(datasets[name][['ape', *EXTRAS]].to_numpy(float)).all()
        datasets[name].to_csv(output / f'hourly_dataset_{name}.csv', index=False, encoding='utf-8-sig')
    assert datasets['NEDO_reference'][['datetime', 'surface', 'ape']].equals(datasets['PUBLIC_POWER'][['datetime', 'surface', 'ape']])
    after = sha(SOURCE)
    assert before == after and stat.st_mtime_ns == SOURCE.stat().st_mtime_ns
    report = dict(source=str(SOURCE), source_sha256_before=before, source_sha256_after=after,
                  source_unchanged=True, source_rows=len(source), base_hourly_report=base_report,
                  common_hours=len(common), common_surface_rows=2*len(common), coordinate=[LATITUDE, LONGITUDE],
                  coordinate_reference=COORDINATE_SOURCE, public_data_metadata=[r[1] for r in retrieved],
                  target='arithmetic mean of valid 10-minute APE, >=3 per surface and hour',
                  predictor_semantics={
                      'NEDO_reference': 'GHI/T/RH means over simultaneous valid 10min records; DHI=GHI*recorded diffuse fraction before averaging; Kt=hourly GHI/G0; diffuse fraction=hourly DHI/GHI',
                      'PUBLIC_POWER': 'hourly GHI/DHI/T/RH from NASA POWER; Kt=GHI/G0; diffuse fraction=DHI/GHI',
                      'G0': 'NASA POWER TOA_SW_DWN in BOTH source conditions to align clearness-index definition',
                      'timestamp': 'NASA POWER UTC hour HH is [HH:00,HH+1:00); relabel interval centre HH:30 for MERRA-2 exact join',
                      'atmosphere': 'existing MERRA-2 4 columns from Naganuma_DS; AM from stored astronomical calculation',
                      'angle': 'HSR 0 and TSR 32 are inherited surface codes, not verified installation geometry'},
                  missing_or_quality_filtered_10min_weather=int((~valid).sum()))
    save_json(output / 'data_preparation_report.json', report)
    print(f'[PREPARED] {len(common):,} common hours, {2*len(common):,} surface rows; source unchanged', flush=True)
    return datasets, report


if __name__ == '__main__':
    prepare()
