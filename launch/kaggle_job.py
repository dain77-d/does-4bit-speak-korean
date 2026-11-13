"""Push one private Kaggle GPU job that runs run.py at a pinned commit, wait for it, and download its results."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

USER = "dainjuser"
REPO = "https://github.com/dain77-d/does-4bit-speak-korean"
ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """import os, subprocess, sys
subprocess.run(["git", "clone", "-q", "{repo}", "/tmp/src"], check=True)
subprocess.run(["git", "-C", "/tmp/src", "checkout", "-q", "{commit}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/src/requirements-kaggle.txt"], check=True)
subprocess.run([sys.executable, "run.py", {args}, "--out", "/kaggle/working/out"], cwd="/tmp/src", check=True,
               env={{**os.environ, "KQT_COMMIT": "{commit}"}})
"""


def build(job_dir, slug, commit, run_args):
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.py").write_text(SCRIPT.format(repo=REPO, commit=commit, args=", ".join(repr(a) for a in run_args)))
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "job.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (job_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return job_dir


def kaggle(*args):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                       env={**os.environ, "PYTHONUTF8": "1"})
    if r.returncode:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


STATUS_RE = re.compile(r'"(?:KernelWorkerStatus\.)?(\w+)"')


def parse_status(line):
    m = STATUS_RE.search(line)
    return m.group(1) if m else ""


def wait(slug, poll=60, timeout=6 * 3600, max_fails=5):
    t0 = time.time()
    fails = 0
    while time.time() - t0 < timeout:
        try:
            line = kaggle("kernels", "status", f"{USER}/{slug}").strip().splitlines()[-1]
        except RuntimeError as e:
            fails += 1
            print(e, flush=True)
            if fails >= max_fails:
                raise
            time.sleep(poll)
            continue
        fails = 0
        status = parse_status(line)
        print(time.strftime("%H:%M:%S"), status or line, flush=True)
        if status == "COMPLETE":
            return True
        if status == "ERROR" or status.startswith("CANCEL"):
            return False
        time.sleep(poll)
    raise TimeoutError(slug)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("run_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    commit = a.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_args = [x for x in a.run_args if x != "--"]
    job = build(ROOT / "jobs" / a.slug, a.slug, commit, run_args)
    pushed = kaggle("kernels", "push", "-p", str(job))
    if "error" in pushed.lower():
        raise RuntimeError(f"kernels push reported an error:\n{pushed}")
    print(pushed)
    ok = wait(a.slug)
    dest = ROOT / "results" / a.slug
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{USER}/{a.slug}", "-p", str(dest))
    print("ok" if ok else "FAILED", "->", dest)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
