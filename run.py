"""Step 3: the bare evolution loop (no database).

best program -> prompt LLM -> apply edit -> evaluate -> keep if better

Usage:
  python run.py --backend openai --model Qwen/Qwen3-8B --iterations 100
  python run.py --backend anthropic --model claude-opus-5-5 --iterations 100
"""
import argparse
import json
import os
import time

from diff import DiffError, apply_diffs, apply_full_rewrite, get_block, parse_diffs
from evaluate import N_CIRCLES, run_program
from llm import make_policy

PROBLEM = f"""Pack {N_CIRCLES} non-overlapping circles inside the unit square [0,1]x[0,1] so that the
SUM of their radii is as large as possible. The best known result is about 2.635.

The program defines `construct_packing()` which returns `(centers, radii)`:
  - centers: numpy array of shape ({N_CIRCLES}, 2)
  - radii:   numpy array of shape ({N_CIRCLES},)
The function may run its own optimizer (numpy and scipy are available). It must finish
within the time limit. The packing is checked strictly (tolerance 1e-9): every circle must
lie fully inside the square and no two circles may overlap; any violation scores 0."""

DIFF_INSTRUCTIONS = """Propose an improvement. Output one or more edits in exactly this format:

<<<<<<< SEARCH
<exact lines copied from the current program>
=======
<replacement lines>
>>>>>>> REPLACE

The SEARCH text must match the current program exactly (including indentation).
Briefly explain your idea first, then give the edits."""

REWRITE_INSTRUCTIONS = """Your previous edit could not be applied: {error}

Instead, output the COMPLETE new version of the program in a single ```python code block.
It must define construct_packing() and include any helper functions and imports it needs."""


def build_prompt(program: dict) -> str:
    return f"""{PROBLEM}

# Current program (score = {program['score']:.6f})
```python
{get_block(program['code'])}
```

# Evaluator feedback
{program['feedback']}

{DIFF_INSTRUCTIONS}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible server URL (vLLM)")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=120.0, help="per-program eval timeout (s)")
    ap.add_argument("--seed", default="seed.py")
    ap.add_argument("--out", default=None, help="run directory (default runs/<timestamp>)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    policy = make_policy(args.backend, args.model, args.base_url)

    seed_code = open(args.seed).read()
    res = run_program(seed_code, args.timeout)
    best = {"code": seed_code, "score": res.score, "feedback": res.feedback}
    print(f"seed score {best['score']:.6f}")

    t0 = time.time()
    n_samples = n_diff_ok = n_fallback = n_valid = 0

    for it in range(1, args.iterations + 1):
        prompt = build_prompt(best)
        rec = {"iter": it, "parent_score": best["score"], "prompt": prompt}
        try:
            comp = policy.generate(prompt)
        except Exception as e:  # network/API errors shouldn't kill a long run
            print(f"[{it}] LLM error: {e}")
            continue
        n_samples += 1
        rec.update(completion=comp.text, input_tokens=comp.input_tokens, output_tokens=comp.output_tokens)

        # 1) try SEARCH/REPLACE diffs; 2) fall back to a full rewrite of the evolve block
        child, diffs = None, parse_diffs(comp.text)
        rec["n_diffs"] = len(diffs)
        try:
            if not diffs:
                raise DiffError("no SEARCH/REPLACE blocks found")
            child = apply_diffs(best["code"], diffs)
            rec["diff_applied"] = True
            n_diff_ok += 1
        except DiffError as e:
            rec["diff_applied"] = False
            rec["diff_error"] = str(e)
            n_fallback += 1
            fb_prompt = prompt + "\n\n" + REWRITE_INSTRUCTIONS.format(error=e)
            try:
                fb = policy.generate(fb_prompt)
                n_samples += 1
                rec.update(fallback_prompt=fb_prompt, fallback_completion=fb.text)
                child = apply_full_rewrite(best["code"], fb.text)
            except Exception as e2:
                rec["fallback_error"] = str(e2)

        if child is None:
            rec.update(score=0.0, feedback="edit could not be applied", valid=False)
        else:
            res = run_program(child, args.timeout)
            rec.update(child_code=child, score=res.score, feedback=res.feedback,
                       valid=res.valid, eval_runtime=res.runtime)
            n_valid += res.valid
            if res.score > best["score"]:
                best = {"code": child, "score": res.score, "feedback": res.feedback}
                with open(os.path.join(out_dir, "best.py"), "w") as f:
                    f.write(child)

        rec.update(best_score=best["score"], n_samples=n_samples, wall_time=time.time() - t0)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[{it:3d}] score {rec['score']:.6f}  best {best['score']:.6f}  "
              f"diff_ok {n_diff_ok}/{it}  valid {n_valid}/{it}  "
              f"{'' if rec['valid'] else rec['feedback'].splitlines()[0][:60]}")

    print(f"\nbest score {best['score']:.6f} after {n_samples} LLM samples, {time.time() - t0:.0f}s")
    print(f"diff-apply success {n_diff_ok}/{args.iterations}, full-rewrite fallbacks {n_fallback}, "
          f"valid programs {n_valid}/{args.iterations}")
    print(f"logs: {out_dir}/log.jsonl   best program: {out_dir}/best.py")


if __name__ == "__main__":
    main()
