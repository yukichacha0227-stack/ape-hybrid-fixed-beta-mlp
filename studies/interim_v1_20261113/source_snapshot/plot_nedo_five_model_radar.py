"""Restyle saved calibrated R-squared values as a five-model radar chart."""
from pathlib import Path
import hashlib
import json
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'比較実験_つくば長沼_10特徴量_MLP_TCN_Attention_20261029'
SOURCE=PARENT/'NEDO_reference_6モデル_R2レーダー/従来分割版_つくばTimesNet未学習'
OUTPUT=PARENT/'NEDO_reference_5モデル_R2レーダー'
MODELS=['fixed_beta_5']+['NEDO_reference_'+kind+'_cal' for kind in ['MLP','TCN','SelfAttention','LSTM']]
LABELS=['β線形回帰のみ','＋MLP','＋TCN','＋SelfAttention','＋LSTM']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(data,design,metric='r2',reverse_error_axis=False):
    reverse=metric=='mape' and reverse_error_axis
    angles=np.pi/2-np.arange(5)*2*np.pi/5
    vectors=np.c_[np.cos(angles),np.sin(angles)]
    fig,ax=plt.subplots(figsize=(10,8.5));ax.set_aspect('equal');ax.axis('off')
    for value in np.arange(.2,1.001,.2):
        polygon=vectors*value
        ax.plot(*np.vstack([polygon,polygon[0]]).T,color='#555555' if value>.99 else '#b8b8b8',
                lw=1.3 if value>.99 else .9,zorder=0)
        tick=max(0.,1-value) if reverse else value
        ax.text(-.035,value,f'{tick:.1f}'+('%' if metric=='mape' else ''),ha='right',va='center',fontsize=12,
                color='#222222',fontweight='bold')
    center='1.0%' if reverse else '0.0%' if metric=='mape' else '0.0'
    ax.text(-.035,0,center,ha='right',va='center',fontsize=12,color='#222222',fontweight='bold')
    for vector,label in zip(vectors,LABELS):
        ax.plot([0,vector[0]],[0,vector[1]],color='#b8b8b8',lw=.9,zorder=0)
        x,y=vector*1.12
        ax.text(x,y,label,ha='center' if abs(x)<.01 else 'left' if x>0 else 'right',
                va='center',fontsize=17,color='#111111',fontweight='bold')
    for site,label,color in [('Tsukuba','つくば','#17659c'),('Naganuma','長沼','#d45d22')]:
        for surface,style,marker in [('HSR','-','o'),('TSR','--','s')]:
            part=data.loc[(data.site==site)&(data.surface==surface)].set_index('model').loc[MODELS]
            column='mape_pct' if metric=='mape' else 'r2'
            assert len(part)==5 and part[column].notna().all()
            values=part[column].to_numpy()
            radius=1-values if reverse else values
            if reverse:assert np.allclose(radius+values,1,atol=1e-12,rtol=0)
            points=vectors*radius[:,None];points=np.vstack([points,points[0]])
            ax.plot(*points.T,label=label+' '+surface,color=color,ls=style,marker=marker,
                    lw=3.6,ms=5.7,solid_capstyle='round',dash_capstyle='round')
    subtitle='従来分割版：地点ごとにTest期間・件数が異なる参考比較' if design=='native' else '共通時刻版：両地点・各面Test 1,326時間'
    heading='NEDO参照：MAPE（%）・内側ほど高精度（補正モデルは校正後）' if metric=='mape' else 'NEDO参照：決定係数 R²（補正モデルは校正後）'
    if reverse:heading=heading.replace('内側ほど','外側ほど')
    ax.set_title(heading+'\n'+subtitle,
                 fontsize=12.5,color='#111111',pad=14)
    ax.set_xlim(-1.65,1.65);ax.set_ylim(-1.10,1.22)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.005),ncol=4,frameon=False,
              prop={'size':12.5,'weight':'bold'},handlelength=2.5,columnspacing=1.4)
    fig.tight_layout()
    stem='つくば長沼_5モデル_R2レーダー' if design=='native' else '参考_共通時刻版_5モデル_R2レーダー'
    if metric=='mape':stem=stem.replace('R2','MAPE')
    if reverse:stem+='_外側高精度'
    for extension in ['png','pdf','svg']:
        fig.savefig(OUTPUT/f'{stem}.{extension}',dpi=300,bbox_inches='tight',pad_inches=.05,facecolor='white')
    plt.close(fig)


def append_mape(data,design,hashes):
    def read(path):
        hashes[str(path)]=sha(path)
        return pd.read_csv(path)
    expected=read(PARENT/'test_metrics_all_contexts.csv')
    for site in ['Tsukuba','Naganuma']:
        base=read(PARENT/design/site/'predictions_all_models.csv')
        if design=='native':
            folder=SOURCE/'Tsukuba/NEDO_reference_LSTM' if site=='Tsukuba' else ROOT/'入力/長沼/APE予測_10特徴量_追加5モデル比較_20261028/NEDO_reference_LSTM'
        else:folder=PARENT/'NEDO_reference_6モデル_R2レーダー'/site/'NEDO_reference_LSTM'
        lstm=read(folder/'predictions.csv');lstm_metrics=read(folder/'metrics.csv')
        assert np.array_equal(pd.to_datetime(base.datetime,utc=True),pd.to_datetime(lstm.datetime,utc=True))
        assert np.array_equal(base.surface,lstm.surface) and np.allclose(base.ape_true,lstm.ape,atol=1e-12,rtol=0)
        for index,row in data.loc[data.site.eq(site)].iterrows():
            mask=base.split.eq('test')&base.surface.eq(row.surface)
            y=base.loc[mask,'ape_true'].to_numpy()
            p=(lstm if row.model=='NEDO_reference_LSTM_cal' else base).loc[mask,row.model].to_numpy()
            assert len(y)==row.n and np.isfinite(p).all() and (y>0).all()
            value=float(100*np.mean(np.abs(p-y)/y))
            r2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2))
            assert abs(r2-row.r2)<1e-10
            if row.model=='NEDO_reference_LSTM_cal':
                saved=lstm_metrics.loc[(lstm_metrics.split=='test')&(lstm_metrics.surface==row.surface)&(lstm_metrics.model==row.model)].iloc[0]
            else:saved=expected.loc[(expected.design==design)&(expected.site==site)&(expected.surface==row.surface)&(expected.model==row.model)].iloc[0]
            assert abs(value-saved.mape_pct)<1e-10
            data.loc[index,'mape_pct']=value
    assert data.mape_pct.between(0,1).all()
    return data


def main(metric='r2',reverse_error_axis=False):
    global OUTPUT
    if metric=='mape':OUTPUT=PARENT/'NEDO_reference_5モデル_MAPEレーダー'
    OUTPUT.mkdir(exist_ok=True)
    for name in ['C:/Windows/Fonts/meiryo.ttc','C:/Windows/Fonts/YuGothM.ttc']:
        if Path(name).exists():
            font_manager.fontManager.addfont(name)
            plt.rcParams['font.family']=font_manager.FontProperties(fname=name).get_name()
            break
    plt.rcParams.update({'axes.unicode_minus':False,'pdf.fonttype':42})
    paths=[SOURCE/f'{design}_評価値.csv' for design in ['native','common_hours']]
    hashes={str(path):sha(path) for path in paths}
    for design,path in zip(['native','common_hours'],paths):
        source=pd.read_csv(path)
        data=source.loc[source.model.isin(MODELS)].copy()
        assert len(data)==20 and data.r2.between(0,1).all()
        assert data.groupby(['site','surface']).model.nunique().eq(5).all()
        assert not data.model.str.contains('TimesNet').any()
        if metric=='mape':data=append_mape(data,design,hashes)
        if metric=='mape' and reverse_error_axis and (OUTPUT/f'{design}_評価値.csv').exists():
            previous=pd.read_csv(OUTPUT/f'{design}_評価値.csv')
            pd.testing.assert_frame_equal(previous.reset_index(drop=True),data.reset_index(drop=True),check_exact=False,atol=1e-12,rtol=0)
        data.to_csv(OUTPUT/f'{design}_評価値.csv',index=False,encoding='utf-8-sig')
        draw(data,design,metric,reverse_error_axis)
    for path,digest in hashes.items():assert sha(Path(path))==digest
    for path in OUTPUT.glob('*.png'):
        with Image.open(path) as image:image.verify()
    description='''# NEDO参照・5モデルのR²レーダー

TimesNetを削除し、β線形回帰のみ、β＋MLP、β＋TCN、β＋Self-Attention、β＋LSTMの5角形に変更。系列はつくばHSR、つくばTSR、長沼HSR、長沼TSR。補正4モデルはすべて校正後、β単体は従来の未校正基準。R²の0〜1目盛を全角で統一した。

系列線を3.6pt、モデル名を17ptの濃い太字、R²目盛を12ptの濃い文字、凡例を12.5ptに変更した。学習や校正はやり直さず、検算済み6モデル図の保存値からTimesNetだけを除いて作図。

主図は直前の6モデル図と同じ従来分割版（native）。つくば各面Test1,748時間、長沼各面Test1,711時間で、地点間のTest期間・件数が異なる参考比較。同じ地点内のモデル比較は同じTest。厳密な同時刻比較には別保存した共通時刻版（各地点・各面1,326時間）を使用する。2つの分割の評価値は混ぜていない。

PNG/PDF/SVGと、描画した20値／分割のCSVを保存。元の評価値と原データは変更していない。
'''
    if metric=='mape':
        description=description.replace('R²レーダー','MAPEレーダー').replace('R²の0〜1目盛を全角で統一した。','MAPE＝100×平均(|予測APE−正解APE|/正解APE)。0〜1%目盛を全角で統一し、中心0%、内側ほど誤差が小さく高精度となる。逆数化や軸別の正規化はしていない。').replace('R²目盛','MAPE目盛').replace('検算済み6モデル図の保存値からTimesNetだけを除いて作図。','保存済みTest予測からMAPEを再計算し、保存済み評価値との一致を1e-10未満で確認して作図。')
        if reverse_error_axis:
            description=description.replace('中心0%、内側ほど誤差が小さく高精度となる。','中心1%、外周0%へ目盛を反転し、外側ほど誤差が小さく高精度となる。描画半径は(1%−MAPE)/1%。')
            description+='\n今回の図はファイル名末尾「_外側高精度」のPNG/PDF/SVG。MAPEの数値自体は前回から変更していない。\n'
    (OUTPUT/'図の説明.md').write_text(description,encoding='utf-8')
    (OUTPUT/'作図検算.json').write_text(json.dumps(dict(status='pass',source_sha256=hashes,
        vertices=5,series=4,metric='mape_pct' if metric=='mape' else 'r2',correction_models_calibrated=True,refitting=False,
        smaller_is_better=metric=='mape',radial_min=0,radial_max=1,
        error_axis_reversed=metric=='mape' and reverse_error_axis,
        metric_value_at_center=1 if metric=='mape' and reverse_error_axis else 0,
        metric_value_at_outer_edge=0 if metric=='mape' and reverse_error_axis else 1,
        model_order=MODELS,line_width_pt=3.6,axis_label_font_pt=17,radial_tick_font_pt=12,
        previous_values_unchanged=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print('[COMPLETE] '+str(OUTPUT),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--metric',choices=['r2','mape'],default='r2')
    parser.add_argument('--reverse-error-axis',action='store_true')
    args=parser.parse_args();main(args.metric,args.reverse_error_axis)
