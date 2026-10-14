import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np

N_CIRCLES = 26
TOL = 1e-9

_SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)"


@dataclass
class EvalResult:
    score: float
    feedback: str
    valid: bool
    runtime: float


def validate(centers, radii, n=N_CIRCLES, tol=TOL):
    try:
        centers = np.asarray(centers, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
    except (TypeError, ValueError):
        return 0.0, "centers/radii are not numeric arrays"
    if centers.shape != (n, 2) or radii.shape != (n,):
        return 0.0, f"bad shapes: centers {centers.shape}, radii {radii.shape}; expected ({n}, 2) and ({n},)"
    if not (np.isfinite(centers).all() and np.isfinite(radii).all()):
        return 0.0, "non-finite values (NaN/inf) in centers or radii"
    if (radii < 0).any():
        return 0.0, f"negative radius for circle {int(np.argmax(radii < 0))}"
    out = ((centers - radii[:, None]) < -tol) | ((centers + radii[:, None]) > 1 + tol)
    if out.any():
        i = int(np.argwhere(out.any(axis=1))[0, 0])
        return 0.0, f"circle {i} (center {centers[i].round(4).tolist()}, r={radii[i]:.4f}) extends outside the square"
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    gap = d - (radii[:, None] + radii[None])
    np.fill_diagonal(gap, np.inf)
    if (gap < -tol).any():
        i, j = np.argwhere(gap < -tol)[0]
        return 0.0, f"circles {i} and {j} overlap by {-gap[i, j]:.2e}"
    return float(radii.sum()), "valid"


def run_program(code: str, timeout: float = 120.0) -> EvalResult:
    start = time.time()
    workdir = tempfile.mkdtemp(prefix="cand_")
    try:
        prog_path = os.path.join(workdir, "program.py")
        out_path = os.path.join(workdir, "result.json")
        with open(prog_path, "w") as f:
            f.write(code)
        cmd = [sys.executable, "-I", prog_path, out_path]
        if shutil.which("sandbox-exec"):
            cmd = ["sandbox-exec", "-p", _SANDBOX_PROFILE] + cmd
        try:
            proc = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout,
                env={"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "1"},
            )
        except subprocess.TimeoutExpired:
            return EvalResult(0.0, f"timeout after {timeout:.0f}s", False, time.time() - start)

        runtime = time.time() - start
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
            return EvalResult(0.0, f"program crashed:\n{tail}", False, runtime)
        try:
            with open(out_path) as f:
                data = json.load(f)
            centers, radii = data["centers"], data["radii"]
        except Exception as e:
            return EvalResult(0.0, f"could not read program output: {e}", False, runtime)

        score, msg = validate(centers, radii)
        return EvalResult(score, msg, score > 0, runtime)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "seed.py"
    with open(path) as f:
        res = run_program(f.read())
    print(f"score={res.score:.6f} valid={res.valid} runtime={res.runtime:.2f}s\nfeedback: {res.feedback}")
