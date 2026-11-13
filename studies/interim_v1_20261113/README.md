# 中間発表資料v1の解析成果物

研究内容・資料掲載図表は[リポジトリのREADME](../../README.md)に掲載。

| 成果物 | 内容 |
|---|---|
| [図表対応と相違](図表対応と相違.md) | スライドと成果物の対応、資料表と検算値の差、条件の区別 |
| [再現手順](再現手順.md) | 指標検算、保存モデル、再学習元コードの利用範囲 |
| [評価CSV](verified_test_metrics.csv) | 両地点・両面、公開/NEDO参照、校正前後のTest指標 |
| [β係数](beta_effective_coefficients.csv) | native/common_hoursの元単位の表面別係数 |
| [モデル一覧](model_inventory.json) | 22条件の保存重み・設定・学習履歴の出典 |
| [分割とTest期間](cohorts.csv) | nativeと共通時刻版の件数・Test開始終了 |
| [資料の定量表](slide13_table_as_presented.csv) | 資料を転記した値。検算済みの値とは区別 |
| [資料のβ表](slide23_table_as_presented.csv) | 資料を転記した概数。計算には正確な係数CSVを使う |
| [レーダーの値](radar_native_metrics.csv) | 資料掲載のnative比較・20値 |
| [共通時刻レーダーの値](radar_common_hours_metrics.csv) | 同一Test時刻による比較・20値 |
| [出典](source_provenance.json) | 元成果物の相対パスとSHA256 |
| [manifest](manifest.json) | この公開版のファイルSHA256 |

```bash
python studies/interim_v1_20261113/verify_release.py
```
