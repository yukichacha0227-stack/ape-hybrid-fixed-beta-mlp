"""Plot the saved, calibrated two-site MLP Test predictions without fitting."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / '比較実験_つくば長沼_10特徴量_MLP_TCN_Attention_20261029'
OUTPUT = EXPERIMENT / 'NEDO_reference_MLP_校正後散布図'
MODEL = 'NEDO_reference_MLP_cal'
SITES = {'Tsukuba': ('つくば', '#2475a7'), 'Naganuma': ('長沼', '#d96931')}
AXIS_LIMITS = {'HSR': (1.84, 2.02), 'TSR': (1.78, 2.01)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, stem):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUTPUT / f'{stem}.{extension}', dpi=300, bbox_inches='tight', pad_inches=.04, facecolor='white')
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    inputs = [EXPERIMENT / 'common_hours' / site / 'predictions_all_models.csv' for site in SITES]
    inputs += [EXPERIMENT / 'test_metrics_all_contexts.csv', ROOT / '入力/Tsukuba_DS.csv', ROOT / '入力/長沼/Naganuma_DS.csv']
    hashes = {str(path): digest(path) for path in inputs}
    for name in ['C:/Windows/Fonts/meiryo.ttc', 'C:/Windows/Fonts/YuGothM.ttc']:
        if Path(name).exists():
            font_manager.fontManager.addfont(name)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=name).get_name()
            break
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    expected = pd.read_csv(EXPERIMENT / 'test_metrics_all_contexts.csv')
    data, rows = {}, []
    keys = None
    for site in SITES:
        frame = pd.read_csv(EXPERIMENT / 'common_hours' / site / 'predictions_all_models.csv')
        frame = frame.loc[frame.split.eq('test')].copy()
        frame['datetime'] = pd.to_datetime(frame.datetime, utc=True)
        current = frame[['datetime', 'surface']].reset_index(drop=True)
        if keys is None:
            keys = current
        else:
            pd.testing.assert_frame_equal(keys, current)
        data[site] = frame
        for surface in ['HSR', 'TSR']:
            subset = frame.loc[frame.surface.eq(surface)]
            y, p = subset.ape_true.to_numpy(), subset[MODEL].to_numpy()
            assert len(y) == 1326 and np.isfinite(y).all() and np.isfinite(p).all()
            residual = y-p
            values = dict(n=len(y), r2=1-np.mean(residual**2)/np.var(y),
                          rmse_meV=1000*np.sqrt(np.mean(residual**2)),
                          mae_meV=1000*np.mean(np.abs(residual)),
                          mape_pct=100*np.mean(np.abs(residual)/y))
            reference = expected.loc[(expected.design == 'common_hours') & (expected.site == site)
                                     & (expected.surface == surface) & (expected.model == MODEL)].iloc[0]
            for key, value in values.items():
                assert abs(value-reference[key]) < 1e-9, (site, surface, key)
            rows.append(dict(site=site, surface=surface, **values))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / '散布図の評価値.csv', index=False, encoding='utf-8-sig')
    def panel(ax, site, surface):
        label, color = SITES[site]
        frame = data[site].loc[data[site].surface.eq(surface)]
        lower, upper = AXIS_LIMITS[surface]
        assert frame[['ape_true', MODEL]].ge(lower).all().all()
        assert frame[['ape_true', MODEL]].le(upper).all().all()
        ax.scatter(frame.ape_true, frame[MODEL], s=10, alpha=.38, color=color, edgecolors='none', rasterized=True)
        ax.plot([lower, upper], [lower, upper], '--', color='#444444', lw=1.15, zorder=0)
        ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel='正解APE（eV）', ylabel='予測APE（eV）',
               title=f'{label} / {surface}')
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.grid(alpha=.16, lw=.6)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.6), layout='compressed')
    fig.get_layout_engine().set(w_pad=.04, h_pad=.04, wspace=.045, hspace=.045)
    for row, surface in enumerate(['HSR', 'TSR']):
        for col, site in enumerate(SITES):
            panel(axes[row, col], site, surface)
    fig.suptitle('大気・気象10特徴量：NEDO_reference\n固定β＋残差MLP＋表面別校正', fontsize=13)
    save(fig, 'つくば長沼_NEDO_reference_β_MLP_校正後_正解予測散布図')
    for site, (label, _) in SITES.items():
        for surface in ['HSR', 'TSR']:
            fig, ax = plt.subplots(figsize=(6, 6.2), layout='compressed')
            panel(ax, site, surface)
            fig.suptitle('NEDO_reference：固定β＋残差MLP＋校正', fontsize=11)
            save(fig, f'{label}_{surface}_NEDO_reference_β_MLP_校正後_散布図')
    all_points = pd.concat([frame.assign(site=site)[['site', 'datetime', 'surface', 'ape_true', MODEL]]
                            for site, frame in data.items()], ignore_index=True)
    all_points.to_csv(OUTPUT / '散布図の全点_Test.csv', index=False, encoding='utf-8-sig')
    for path, before in hashes.items():
        assert digest(Path(path)) == before
    for path in OUTPUT.glob('*.png'):
        with Image.open(path) as img:
            img.verify()
    result = dict(status='pass', model=MODEL, design='common_hours', refitting=False,
                  same_test_timestamps=True, points_per_site_and_surface=1326,
                  plotted_points=len(all_points), all_points_retained=True,
                  axis_limits_by_surface_eV=AXIS_LIMITS, legend_displayed=False,
                  metric_annotations_displayed=False, layout='compressed', source_sha256=hashes,
                  original_inputs_unchanged=True, metrics_match_saved_evaluation=True)
    (OUTPUT / '作図検算.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT / '図の説明.md').write_text('''# NEDO参照・β＋MLP＋校正の正解―予測散布図

共通時刻比較版（common_hours）の独立Testだけを使用。つくば・長沼のHSR/TSRそれぞれ1,326時間、全5,304点。横軸は正解の時間平均APE、縦軸は固定β＋MLP残差＋表面別校正の最終予測APE（eV）。破線は予測＝正解。ユーザー指定によりHSRの縦横軸は1.84〜2.02 eV、TSRは1.78〜2.01 eV。同じ面は両地点で軸を統一し、全点が指定範囲内にあることを確認。凡例・図中の件数と評価値表示を削除し、レイアウトの余白を詰めた。評価値はCSVに保存している。

物理量10種類と時刻・季節・表面条件・β予測・過去残差等の補助情報を使う既存モデルの保存済み予測列NEDO_reference_MLP_calから作図した。固定β自体は大気5変数と面条件・交互作用のRidge回帰。β・MLPは地点別Train、校正は地点別CALで推定したものを使用し、作図時の再学習・再校正は行っていない。

2地点×2面の比較図と、各条件の個別図をPNG/PDF/SVGで保存。描画した全点は散布図の全点_Test.csv、指標は散布図の評価値.csv、元ファイルの保全と既存指標との一致は作図検算.json。native版（従来の地点別分割）とは件数・分割が異なる。
''', encoding='utf-8')
    print(metrics.round(6).to_string(index=False))
    print('Saved: '+str(OUTPUT))


if __name__ == '__main__':
    main()
