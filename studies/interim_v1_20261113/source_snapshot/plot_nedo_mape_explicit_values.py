"""Replot saved MAPE with explicit spoke ticks, MLP values, and coordinate audit."""
from pathlib import Path
import json
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import plot_nedo_five_model_radar as source


def draw(data, design, out, outer=0.0):
    angles = np.pi / 2 - np.arange(5) * 2 * np.pi / 5
    vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    fig = plt.figure(figsize=(12, 10.8))
    ax = fig.add_axes([.03, .26, .94, .67])
    ax.set_aspect('equal')
    ax.axis('off')
    ticks = [outer] + [value for value in [.2, .4, .6, .8] if value > outer]
    for error in ticks:
        radius = (1 - error) / (1 - outer)
        ring = vectors * radius
        ax.plot(*np.vstack((ring, ring[0])).T, color='#777777' if error == outer else '#b8b8b8', lw=1.2 if error == outer else .9, zorder=0)
        for vector in vectors:
            tangent = np.array([-vector[1], vector[0]])
            xy = vector * radius + tangent * .032
            ax.text(*xy, f'{error:.1f}%', fontsize=10.5, fontweight='bold', ha='center', va='center', color='#333333', bbox=dict(facecolor='white', edgecolor='none', pad=.3), zorder=4)
    ax.text(0, 0, '1.0%', ha='center', va='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1), zorder=4)
    for vector, label in zip(vectors, source.LABELS):
        ax.plot([0, vector[0]], [0, vector[1]], color='#b8b8b8', lw=.9, zorder=0)
        ax.text(*(vector * 1.15), label, ha='center' if abs(vector[0]) < .01 else 'left' if vector[0] > 0 else 'right', va='center', fontsize=17, fontweight='bold', color='#111111')
    series = [('Tsukuba', 'TSR', 'つくば TSR', '#17659c', '--', 's'),
              ('Tsukuba', 'HSR', 'つくば HSR', '#17659c', '-', 'o'),
              ('Naganuma', 'TSR', '長沼 TSR', '#d45d22', '--', 's'),
              ('Naganuma', 'HSR', '長沼 HSR', '#d45d22', '-', 'o')]
    audits, rows, row_labels, row_colors = [], [], [], []
    for number, (site, surface, name, color, style, marker) in enumerate(series):
        part = data.loc[data.site.eq(site) & data.surface.eq(surface)].set_index('model').loc[source.MODELS]
        errors = part.mape_pct.to_numpy()
        assert np.all((errors >= outer) & (errors <= 1.0))
        radii = (1.0 - errors) / (1.0 - outer)
        xy = vectors * radii[:, None]
        line, = ax.plot(*np.vstack((xy, xy[0])).T, color=color, ls=style, marker=marker, lw=3.6, ms=6, label=name, zorder=3)
        # Read back actual artist coordinates, rather than auditing only the source array.
        drawn = np.column_stack(line.get_data())[:5]
        recovered = 1 - (1 - outer) * np.sum(drawn * vectors, axis=1)
        np.testing.assert_allclose(recovered, errors, atol=1e-12, rtol=0)
        for model, error, point, actual in zip(source.MODELS, errors, drawn, recovered):
            audits.append(dict(design=design, site=site, surface=surface, model=model, mape_pct=float(error), x=float(point[0]), y=float(point[1]), mape_recovered_from_artist=float(actual)))
        ax.annotate(f'{name}  {errors[1]:.4f}%', xy=xy[1], xytext=(1.07, .20 - number * .135), ha='left', va='center', fontsize=11.5, fontweight='bold', color=color, arrowprops=dict(arrowstyle='-', color=color, lw=1.2, connectionstyle='angle,angleA=0,angleB=90,rad=3'), bbox=dict(facecolor='white', edgecolor='none', pad=2), zorder=6)
        rows.append([f'{v:.4f}' for v in errors]); row_labels.append(name); row_colors.append(color)
    ax.set_xlim(-1.6, 1.95); ax.set_ylim(-1.08, 1.24)
    ax.legend(loc='upper center', bbox_to_anchor=(.49, -.015), ncol=4, frameon=False, prop={'size': 12, 'weight': 'bold'})
    subtitle = '従来分割版：つくば各面1,748時間／長沼各面1,711時間（地点間でTest期間が異なる）' if design == 'native' else '共通時刻版：両地点・各面Test 1,326時間'
    fig.suptitle(f'NEDO参照：MAPE（%）  外周{outer:g}%・中心1%\n' + subtitle, fontsize=13, y=.98)
    table_ax = fig.add_axes([.18, .025, .76, .15]); table_ax.axis('off')
    table = table_ax.table(cellText=rows, rowLabels=row_labels, colLabels=source.LABELS, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10.5); table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0: cell.set_text_props(weight='bold'); cell.set_facecolor('#f0f0f0')
        elif col == -1: cell.set_text_props(color=row_colors[row - 1], weight='bold')
    stem = ('つくば長沼' if design == 'native' else '参考_共通時刻版') + '_5モデル_MAPE_外側高精度_目盛数値明示版'
    if outer != 0: stem += f'_最外周{outer:g}pct'
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'{stem}.{extension}', dpi=300, bbox_inches='tight', pad_inches=.1, facecolor='white')
    plt.close(fig)
    return audits


def main(outer=0.0):
    sys.stdout.reconfigure(encoding='utf-8')
    font_path = Path('C:/Windows/Fonts/meiryo.ttc')
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['pdf.fonttype'] = 42
    out = source.PARENT / 'NEDO_reference_5モデル_MAPEレーダー'
    hashes, audits = {}, []
    for design in ['native', 'common_hours']:
        path = out / f'{design}_評価値.csv'
        hashes[str(path)] = source.sha(path)
        data = pd.read_csv(path)
        recomputed = source.append_mape(data.copy(), design, hashes)
        np.testing.assert_allclose(data.mape_pct, recomputed.mape_pct, atol=1e-12, rtol=0)
        audits.extend(draw(recomputed, design, out, outer))
    for path, digest in hashes.items():
        assert source.sha(Path(path)) == digest
    prefix = '目盛数値明示版' + (f'_最外周{outer:g}pct' if outer else '')
    pd.DataFrame(audits).to_csv(out / f'{prefix}_描画座標検算.csv', index=False, encoding='utf-8-sig')
    (out / f'{prefix}_検算.json').write_text(json.dumps(dict(status='pass', vertices_verified=len(audits), outer_mape_pct=outer, center_mape_pct=1.0, check='MAPE recalculated from saved predictions; inverse transform of actual Matplotlib line coordinates matches MAPE', source_sha256=hashes, source_unchanged=True), ensure_ascii=False, indent=2), encoding='utf-8')
    (out / f'{prefix}_説明.md').write_text(f'各軸に同じMAPE目盛を表示し、MLPの4系列には実数値を直接付記した。表にも全20値を保存。最外周{outer:g}%、中心1%。描画半径=(1−MAPE[%])/(1−{outer:g})。保存予測からMAPEを再計算し、実際に描画した線の座標から逆算したMAPEとの一致を検算した。従来分割版と共通時刻版は別図とし、元データ・学習結果は変更していない。', encoding='utf-8')
    print(json.dumps(dict(status='pass', vertices=len(audits), output=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outer', type=float, default=0.0)
    main(parser.parse_args().outer)
