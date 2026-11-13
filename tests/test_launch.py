import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launch"))
import kaggle_job


def test_build_writes_valid_job(tmp_path):
    d = kaggle_job.build(tmp_path / "j", "kqt-e1-qwen", "abc123", ["--model", "Qwen/Qwen2.5-3B-Instruct", "--grid", "e1"])
    meta = json.loads((d / "kernel-metadata.json").read_text())
    assert meta["id"] == "dainjuser/kqt-e1-qwen"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    src = (d / "job.py").read_text()
    compile(src, "job.py", "exec")
    assert "abc123" in src and "Qwen/Qwen2.5-3B-Instruct" in src and "/kaggle/working/out" in src


def test_parse_status_extracts_quoted_status():
    assert kaggle_job.parse_status('dainjuser/x has status "KernelWorkerStatus.COMPLETE"') == "COMPLETE"
    assert kaggle_job.parse_status('dainjuser/x has status "KernelWorkerStatus.ERROR"') == "ERROR"
    assert kaggle_job.parse_status('dainjuser/x has status "KernelWorkerStatus.RUNNING"') == "RUNNING"
    assert kaggle_job.parse_status('dainjuser/kqt-error-run has status "KernelWorkerStatus.RUNNING"') == "RUNNING"
    assert kaggle_job.parse_status("no status here") == ""
