"""Independently audit published Test predictions with Python's standard library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(truth, prediction):
    n = len(truth)
    if not n or n != len(prediction):
        raise ValueError("Aligned, nonempty truth and prediction are required")
    if not all(math.isfinite(y) and y > 0 for y in truth):
        raise ValueError("APE truth must be positive and finite")
    if not all(math.isfinite(p) for p in prediction):
        raise ValueError("Predictions must be finite")
    error = [p-y for y, p in zip(truth, prediction)]
    mse = math.fsum(e*e for e in error)/n
    mean = math.fsum(truth)/n
    denominator = math.fsum((y-mean)**2 for y in truth)
    if denominator == 0:
        raise ValueError("R-squared is undefined for this constant target")
    return dict(n=n, r2=1-n*mse/denominator,
                mae_meV=1000*math.fsum(abs(e) for e in error)/n,
                rmse_meV=1000*math.sqrt(mse),
                mape_pct=100*math.fsum(abs(e)/y for e,y in zip(error,truth))/n,
                mbe_prediction_minus_truth_meV=1000*math.fsum(error)/n)


def verify(root=ROOT):
    repo=root.parents[1]
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path=repo/entry["path"]
        assert path.is_file(), entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry["sha256"], entry["path"]
    facts=json.loads((root/"build_facts.json").read_text(encoding="utf-8"))
    assert facts["no_refitting"] and facts["original_files_unchanged"]
    expected=read_csv(root/"verified_test_metrics.csv")
    source_tables={}
    keys={}
    for design in ["common_hours","native"]:
        for site in ["Tsukuba","Naganuma"]:
            rows=read_csv(root/"predictions"/f"{design}_{site}_test.csv")
            source_tables[design,site]=rows
            stamps=[(r["datetime"],r["surface"]) for r in rows]
            assert len(stamps)==len(set(stamps)), (design,site)
            keys[design,site]=stamps
    assert keys["common_hours","Tsukuba"]==keys["common_hours","Naganuma"]
    assert keys["native","Tsukuba"]!=keys["native","Naganuma"]
    checked=0
    for target in expected:
        rows=[r for r in source_tables[target["design"],target["site"]] if r["surface"]==target["surface"]]
        values=metrics([float(r["ape_true"]) for r in rows],[float(r[target["model"]]) for r in rows])
        for field,value in values.items():
            assert abs(value-float(target[field]))<1e-9, (target,field,value)
        checked+=1
    assert checked==facts["verified_metric_rows"]==96
    for design in ["native","common_hours"]:
        radar=read_csv(root/f"radar_{design}_metrics.csv")
        assert len(radar)==20
        for value in radar:
            found=[r for r in expected if r["design"]==design and r["site"]==value["site"] and r["surface"]==value["surface"] and r["model"]==value["model"]]
            assert len(found)==1, value
            for field in ["n","r2","mape_pct"]:
                assert abs(float(value[field])-float(found[0][field]))<1e-9,(value,field)
    provenance=json.loads((root/"source_provenance.json").read_text(encoding="utf-8"))
    for item in provenance:
        assert hashlib.sha256((repo/item["path"]).read_bytes()).hexdigest()==item["source_sha256"], item["path"]
    image_count=0
    for path in root.rglob("*.png"):
        assert path.read_bytes()[:8]==b"\x89PNG\r\n\x1a\n",str(path)
        image_count+=1
    for path in [repo/"README.md",root/"再現手順.md",root/"図表対応と相違.md",repo/"docs/historical/README_tsukuba_five_feature_main.md"]:
        text=path.read_text(encoding="utf-8")
        assert not any(ord(ch)<32 and ch not in "\r\n\t" for ch in text), str(path)
        for target in re.findall(r"\]\(([^)]+)\)",text):
            target=target.strip("<>").split("#")[0]
            if target and not re.match(r"[A-Za-z]+://",target):
                assert (path.parent/unquote(target)).exists(), (str(path),target)
    assert facts["model_count"]==facts["models_with_weights"]==22
    weights=list((root/"models").rglob("model.pt"))
    assert len(weights)==22
    print(json.dumps(dict(status="pass",verified_metric_rows=checked,
        radar_values_checked=40,png_files=image_count,model_weights=len(weights),
        common_hours_same_test=True,native_different_tests=True,
        hash_checked_files=len(manifest["files"]),bias_definition="prediction - truth"),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args()
    verify(args.root.resolve())
