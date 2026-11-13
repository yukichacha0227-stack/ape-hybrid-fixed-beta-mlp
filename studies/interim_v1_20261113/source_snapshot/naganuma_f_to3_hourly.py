"""Read-only Naganuma_DS adapter for the existing hourly Experiment F+TO3.

The reference's training, features, OOF, recursive inference and calibration
functions are imported unchanged. Only data preparation and reporting differ.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "比較実験F/比較実験F_TO3追加_1時間整合版/比較実験F_TO3追加_1時間整合版.py"
DEFAULT_INPUT = ROOT / "入力/長沼/Naganuma_DS.csv"
DEFAULT_OUTPUT = ROOT / "入力/長沼/APE予測_F_TO3_1時間整合版"
ATMOSPHERE = {"aod": "aod550", "angstrom_exponent": "totangstr", "tqv": "omega", "to3": "to3"}


def load_reference():
    spec = importlib.util.spec_from_file_location("naganuma_hourly_reference_f_to3", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


F = load_reference()


def read_source(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    source.columns = source.columns.str.strip()
    required = {"datetime", "SiteNum", "am", *ATMOSPHERE,
                "ape_hsr", "ape_tsr", "remark_hsr", "remark_tsr"}
    if missing := sorted(required - set(source.columns)):
        raise ValueError(f"Missing Naganuma_DS columns: {missing}")
    source["datetime"] = pd.to_datetime(source["datetime"], utc=True, errors="raise").dt.as_unit("ns")
    for column in required - {"datetime"}:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source["datetime"].isna().any() or source.duplicated(["datetime", "SiteNum"]).any():
        raise ValueError("Missing or duplicate source timestamps.")
    if not source["SiteNum"].eq(101).all():
        raise ValueError("Naganuma_DS must contain only SiteNum=101.")
    return source.sort_values("datetime").reset_index(drop=True)


def prepare_from_source(source: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    """Aggregate all available 10-minute targets BEFORE joining hourly atmosphere."""
    keys = ["datetime", "SiteNum"]
    surfaces, reports, loads = {}, {}, {}
    for surface in ("HSR", "TSR"):
        target, remark = f"ape_{surface.lower()}", f"remark_{surface.lower()}"
        selected = source[remark].isin([1, 2]) & np.isfinite(source[target])
        raw = source.loc[selected, ["datetime", "SiteNum", target, remark]].copy()
        raw = raw.rename(columns={target: "APE", remark: "Remark"})
        raw["surface"] = surface
        hourly, report = F.aggregate_target_hourly(raw, settings.min_samples_per_hour)
        surfaces[surface], reports[surface] = hourly, report
        loads[surface] = {
            "surface": surface, "raw_rows": len(source), "quality_selected_rows": len(raw),
            "remark_counts": raw["Remark"].value_counts().sort_index().to_dict(),
            "start_utc": raw["datetime"].min(), "end_utc": raw["datetime"].max(),
        }
    common = surfaces["HSR"][keys].merge(surfaces["TSR"][keys], on=keys, validate="one_to_one")

    valid_am = np.isfinite(source["am"]) & source["am"].ne(-9999)
    am_raw = source.loc[valid_am, [*keys, "am"]].copy()
    am_raw["datetime"] = F.hourly_center_utc(am_raw["datetime"])
    am = am_raw.groupby(keys, as_index=False).agg(
        am=("am", "mean"), am_std_within_hour=("am", "std"), am_samples_in_hour=("am", "size")
    )

    complete = np.isfinite(source[list(ATMOSPHERE)].to_numpy(float)).all(axis=1)
    atmosphere = source.loc[complete, [*keys, *ATMOSPHERE]].rename(columns=ATMOSPHERE).copy()
    off_center = atmosphere["datetime"].dt.minute.ne(30) | atmosphere["datetime"].dt.second.ne(0)
    if off_center.any():
        raise ValueError("Atmospheric predictors must be at exact :30 UTC centers; no resampling permitted.")
    if not atmosphere["to3"].between(0, 1000, inclusive="neither").all():
        raise ValueError("TO3 is outside the reference's physical audit bounds (0, 1000) DU.")
    predictors = am.merge(atmosphere, on=keys, how="inner", validate="one_to_one")
    common_predictors = common.merge(predictors[keys], on=keys, how="inner", validate="one_to_one")
    predictors = predictors.merge(common_predictors, on=keys, how="inner", validate="one_to_one")
    parts = []
    for surface in ("HSR", "TSR"):
        part = surfaces[surface].merge(predictors, on=keys, how="inner", validate="one_to_one")
        part["surface"] = surface
        # This is the reference's binary surface encoding, not a verified Naganuma installation angle.
        part["tilt_angle_deg"] = F.SURFACE_TILT[surface]
        part["surface_tsr"] = float(surface == "TSR")
        parts.append(part)
    data = pd.concat(parts, ignore_index=True).sort_values(["datetime", "surface"]).reset_index(drop=True)
    if data.empty or not data.groupby("datetime")["surface"].nunique().eq(2).all():
        raise ValueError("Every selected hour must have both HSR and TSR.")
    if data.duplicated([*keys, "surface"]).any():
        raise ValueError("Duplicate hourly surface observations.")
    if not np.isfinite(data[["ape", "am", *ATMOSPHERE.values()]].to_numpy(float)).all():
        raise ValueError("Nonfinite model inputs.")
    report = {
        "experiment": "Naganuma F+TO3", "source_rows": len(source),
        "definition": "HSR/TSR arithmetic hourly means from read-only Naganuma_DS",
        "target_aggregation_warning": "APEの算術時間平均。スペクトルを積算してAPEを再計算した値ではない。",
        "source_cohort_warning": "Naganuma_DS is already paired HSR/TSR at 10-minute timestamps; unpaired original spectra are outside this input cohort.",
        "hsr_load": loads["HSR"], "tsr_load": loads["TSR"],
        "hsr_hourly": reports["HSR"], "tsr_hourly": reports["TSR"],
        "am_hourly": {"valid_10min_rows": len(am_raw), "hourly_rows": len(am),
                      "aggregation": "arithmetic mean within JST [hh:00, hh+1:00)"},
        "atmosphere": {"complete_center_rows": len(atmosphere), "source_to_model_columns": ATMOSPHERE,
                       "timestamp_semantics": "existing MERRA-2 tavg1 center; exact join, no imputation"},
        "exact_common_hsr_tsr_hours_before_predictors": len(common),
        "exact_common_hours_after_all_predictors": int(data["datetime"].nunique()),
        "final_surface_rows": len(data), "surface_row_counts": data["surface"].value_counts().to_dict(),
        "start_utc": data["datetime"].min(), "end_utc": data["datetime"].max(),
        "join": "exact center equality after target/AM aggregation; no interpolation or broadcasting",
        "surface_tilt_labels_deg": F.SURFACE_TILT,
        "surface_tilt_label_interpretation": "reference coding 0/32 retained; not verified physical installation angles; the standardized column duplicates binary surface_tsr",
    }
    return data, report


def md_table(frame: pd.DataFrame, columns=None, digits=4) -> str:
    frame = frame if columns is None else frame[columns]
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output, run_name, settings, data_report, split_report, metrics,
                 beta_report, model_report, data, prediction_frame):
    # MAPE is supplementary and does not alter any reference-model fitting.
    for index, row in metrics.iterrows():
        mask = prediction_frame["split"].eq(row["split"])
        if row["surface"] != "ALL":
            mask &= prediction_frame["surface"].eq(row["surface"])
        part = prediction_frame.loc[mask]
        metrics.loc[index, "mape_pct"] = 100 * np.mean(
            np.abs(part["ape_true"] - part[row["model"]]) / part["ape_true"].abs())
    metrics.to_csv(output / "metrics_all_splits.csv", index=False, encoding="utf-8-sig")
    test = metrics.loc[metrics["split"].eq("test") & metrics["surface"].isin(["HSR", "TSR"])].copy()
    display = ["surface", "model", "n", "r2", "rmse_meV", "mae_meV", "mbe_truth_minus_prediction_meV", "mape_pct"]
    split_rows = [{"split": role, "hours": details["unique_hours"],
                   "start_utc": details["start_utc"], "end_utc": details["end_utc"]}
                  for role, details in split_report.items() if role != "purge"]
    comparisons = []
    notes = []
    benchmark_path = REFERENCE.parent / "比較実験F_TO3追加_全評価指標.csv"
    benchmark = pd.read_csv(benchmark_path, encoding="utf-8-sig") if benchmark_path.exists() else pd.DataFrame()
    for surface in ("HSR", "TSR"):
        group = test.loc[test["surface"].eq(surface)].set_index("model")
        final, linear, raw = (group.loc[name] for name in [F.MODEL_HYBRID_CAL, F.MODEL_FIXED, F.MODEL_HYBRID_RAW])
        notes.append(f"- {surface}：主モデルR²={final['r2']:.4f}、RMSE={final['rmse_meV']:.3f} meV、"
                     f"MAE={final['mae_meV']:.3f} meV。線形βからΔR²={final['r2']-linear['r2']:+.4f}、"
                     f"ΔRMSE={final['rmse_meV']-linear['rmse_meV']:+.3f} meV。"
                     f"校正前から校正後のΔRMSE={final['rmse_meV']-raw['rmse_meV']:+.3f} meV。"
                     f"MBE（正解−予測）={final['mbe_truth_minus_prediction_meV']:+.3f} meV。")
        for site, frame in [("Naganuma", test), ("Tsukuba_reference", benchmark)]:
            if not frame.empty:
                selected = frame.loc[frame["split"].eq("test") & frame["surface"].eq(surface)
                                     & frame["model"].eq(F.MODEL_HYBRID_CAL)].copy()
                selected["site"] = site
                comparisons.append(selected)
    compared = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    compared.to_csv(output / "tsukuba_reference_comparison.csv", index=False, encoding="utf-8-sig")
    distribution = data.groupby("surface")["ape"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    bands = []
    for surface in ("HSR", "TSR"):
        part = prediction_frame.loc[prediction_frame["split"].eq("test") & prediction_frame["surface"].eq(surface)]
        q10, q90 = part["ape_true"].quantile([.1, .9])
        for label, mask in [("low10", part["ape_true"].le(q10)),
                            ("middle80", part["ape_true"].gt(q10) & part["ape_true"].lt(q90)),
                            ("high10", part["ape_true"].ge(q90))]:
            g = part.loc[mask]
            bands.append({"surface": surface, "band": label, **F.regression_metrics(g["ape_true"], g[F.MODEL_HYBRID_CAL])})
    bands = pd.DataFrame(bands)
    bands.to_csv(output / "test_residual_by_target_band.csv", index=False, encoding="utf-8-sig")
    history = pd.read_csv(output / "training_history_residual_mlp.csv")
    source_check = json.loads((output / "input_integrity.json").read_text(encoding="utf-8"))
    text = f"""# 長沼：比較実験F＋TO3と同じ1時間整合手法による予測・評価

## 1. 目的と入力の保全

実行日：{pd.Timestamp.now(tz='Asia/Tokyo').date()}（JST）。Naganuma_DSを読み取り専用で読み込み、参照実験と同じ学習・特徴量・再帰推論・校正関数をそのまま呼び出した。
元データSHA256：`{source_check['before_sha256']}`。実行後の同一性は`input_integrity.json`に記録する。

## 2. データ準備

- 入力総行数：{data_report['source_rows']:,}。HSR・TSRごとにRemark 1/2、有限APEを採用。
- JST [hh:00, hh+1:00) の10分APEを表面別に算術平均。有効APEが各表面3点以上の時間だけ採用。スペクトル積算後のAPE再計算ではない。
- AMも10分値から時間平均。先に大気変数の完全ケースへ絞ると10分観測を失うため、APE・AMの時間集約を先に行った。
- AOD・Å・TQV・TO3はDS内の毎時30分の既存値を使い、集約窓中心のUTC時刻と完全一致結合。補間・前方補完・10分行への反復付与は行わない。
- 最終共通時間数：{data_report['exact_common_hours_after_all_predictors']:,}、表面行数：{len(data):,}。
- DSはもともとHSR/TSRの共通10分時刻に限定されている。つくば元実験の各表面の全原観測と、採用母集団までは一致しない。

{md_table(distribution)}

## 3. 参照実験との一致・実装上の扱い

- 大気5変数：AM、AOD550、TQV（omega）、Å（TOTANGSTR）、TO3。
- 固定β：Ridge α=1、表面ラベルと各大気変数×TSRの固定交互作用。両表面をまとめて学習。
- MLP入力：{len(model_report['residual_feature_names'])}列。表面条件・β予測・時刻/年周期・大気交互作用・透過率proxy・1/2/3/6実時間前の残差と利用可否を含む。
- HSR=0、TSR=32という角度列は参照実装の固定符号化を維持。長沼の実設置角の確認値ではない。この列は標準化後に二値TSRラベルと重複する。
- OOFはTrain内の5分割expanding window。MLPはTrain OOF残差を教師値とし、学習時の残差ラグもTrain OOF残差から作成する。
- MLP：128–128–64、GELU、LayerNorm、Dropout 0.05、Huber(delta=1)、AdamW、lr=0.0002、batch=512、最大300 epoch、patience=100、seed=42。
- 参照実装どおり、早期停止Validationはラグ0・利用可否0で評価し、最終評価は表面別・split別に履歴をリセットした再帰推論を用いる。Testでは実測残差を入力せず、自分の過去の予測残差だけを参照する。
- 校正はValidation後半だけ。残差のアフィン傾きは[0,1.5]、clipは校正区間の真の残差のMADに基づくrobust sigma×3。つくば参照と同一。
- purge設定名は6時間だが、参照の処理は6個の採用時刻を除外する。一定の実時間幅への変更はしていない。
- 標準化・β学習・MLP学習・早期停止・校正にTest正解値を使わない。

{md_table(pd.DataFrame(split_rows))}

## 4. 独立Testの結果

{md_table(test, display)}

誤差単位はmeV、MAPEは%。MBEは正解−予測で、正が過小予測、負が過大予測。
`{F.MODEL_HYBRID_CAL}`が主モデル。Trainの残差MLP評価は学習内診断を含み、汎化性能ではない。

## 5. 評価・考察

{chr(10).join(notes)}

最良epoch：{model_report['best_epoch']}、実行epoch：{int(history['epoch'].max())}、Validation最良標準化Huber損失：{model_report['best_validation_loss']:.6f}。
同一のTest行で固定β・校正前主モデル・校正後主モデル・HGBを比較できる。旧5特徴量限定の長沼実験とは正解の時間集約・行数・分割・損失・入力が異なるため、数値差を単独要因の改善量とはしない。

## 6. 既存つくばとの参考比較

{md_table(compared, ['site','surface','n','r2','rmse_meV','mae_meV','mbe_truth_minus_prediction_meV'])}

学習手法は共通だが、地点、採用期間、時間数、観測の採用母集団が異なる。これは同一Test母集団を使う対応比較ではない。つくばから長沼へのモデル移植でもなく、長沼内で学習・Test評価した結果である。

## 7. 季節・時刻・APE帯域別の誤差

{md_table(pd.read_csv(output / 'test_residual_by_season.csv'))}

{md_table(bands, ['surface','band','n','r2','rmse_meV','mae_meV','mbe_truth_minus_prediction_meV'])}

上下10%は各表面のTest正解から診断用に設定し、学習や設定選択には使っていない。Testが含む月だけの季節評価であり、冬全体の評価とは限らない。

## 8. 予測図・再現・成果物

![主モデルの正解と予測](test_measured_vs_predicted_primary.png)

![学習履歴](residual_mlp_learning_history.png)

- `hourly_aligned_hsr_tsr_dataset.csv`：別途生成した1時間データ。
- `predictions_all_models.csv`、`metrics_all_splits.csv`：全split、全モデルの予測と指標。
- `split_report.json`、`fixed_beta_coefficients.json`、`model_and_calibration_report.json`：分割・係数・校正・30入力の情報。
- `non_torch_models_and_preprocessors.joblib`、`residual_mlp_model.pt`：保存モデル。
- `input_integrity.json`：DSの実行前後ハッシュと更新時刻。`verification.json`：独立した成果物検証。
- 再実行：`.venv/Scripts/python.exe naganuma_f_to3_hourly.py --output <新しい出力先>`。既存出力は上書きしない。
- 参照コード：`{REFERENCE.relative_to(ROOT)}`。読み込み部分と報告以外の学習関数は変更していない。

同時刻の気象再解析値が存在する条件での時間平均APE推定であり、先行時間付きの将来予報ではない。単一seed・単一時系列分割の結果として解釈する。
"""
    destination = output / "評価.md"
    destination.write_text(text, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    input_path, output = args.input.resolve(), args.output.resolve()
    if output == input_path or input_path.is_relative_to(output):
        raise ValueError("Output must not contain or replace the source file.")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    source_hash = F.sha256(input_path)
    source_stat = input_path.stat()
    source = read_source(input_path)
    import torch
    torch.set_num_threads(min(8, torch.get_num_threads()))
    original_prepare = F.prepare_data
    original_report = F.markdown_report
    provenance = {
        "input": str(input_path), "before_sha256": source_hash,
        "before_size": source_stat.st_size, "before_mtime_ns": source_stat.st_mtime_ns,
        "reference": str(REFERENCE), "reference_sha256": F.sha256(REFERENCE),
        "adapter_sha256": F.sha256(Path(__file__)), "source_written_by_adapter": False,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }

    def prepare(reference_args, reference_settings):
        data, report = prepare_from_source(source, reference_settings)
        F.atomic_json(output / "input_integrity.json", provenance)
        return data, report

    F.prepare_data = prepare
    F.markdown_report = write_report
    argv = ["--output-root", str(output.parent), "--run-name", output.name,
            "--hsr", str(input_path), "--tsr", str(input_path), "--aerosol", str(input_path),
            "--omega", str(input_path), "--ozone", str(input_path), "--met-root", str(input_path.parent)]
    if args.prepare_only:
        argv.append("--prepare-only")
    try:
        result = F.main(argv)
        if result:
            raise RuntimeError(f"Reference pipeline exited with status {result}")
    finally:
        F.prepare_data = original_prepare
        F.markdown_report = original_report
        after_stat = input_path.stat()
        provenance.update(after_sha256=F.sha256(input_path), after_size=after_stat.st_size,
                          after_mtime_ns=after_stat.st_mtime_ns)
        provenance["unchanged"] = (source_hash == provenance["after_sha256"]
                                   and source_stat.st_mtime_ns == after_stat.st_mtime_ns
                                   and source_stat.st_size == after_stat.st_size)
        if output.exists() and (output / "input_integrity.json").exists():
            F.atomic_json(output / "input_integrity.json", provenance)
        if not provenance["unchanged"]:
            raise RuntimeError("Naganuma_DS changed during the run; input integrity check failed.")
    print(f"Naganuma_DS unchanged: {source_hash}", flush=True)


if __name__ == "__main__":
    main()
