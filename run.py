import argparse
import json
import os
import random
import time

from diff import DiffError, apply_diffs, apply_full_rewrite, get_block, parse_diffs
from database import Program, ProgramDatabase
from evaluate import N_CIRCLES, run_program
from llm import make_policy

PROBLEM = f"""Pack {N_CIRCLES} non-overlapping circles inside the unit square [0,1]x[0,1] so that the
SUM of their radii is as large as possible. The best known result is about 2.635.

The program defines `construct_packing()` which returns `(centers, radii)`:
  - centers: numpy array of shape ({N_CIRCLES}, 2)
  - radii:   numpy array of shape ({N_CIRCLES},)
The function may run its own optimizer (numpy and scipy are available). Only numpy is
imported (as `np`); import anything else yourself inside your code, e.g.
`from scipy.optimize import minimize`. It must finish within the time limit. The packing is checked strictly (tolerance 1e-9): every circle must
lie fully inside the square and no two circles may overlap; any violation scores 0."""

DIFF_INSTRUCTIONS = """Propose an improvement. Output one or more edits in exactly this format:

<<<<<<< SEARCH
<exact lines copied from the current program>
=======
<replacement lines>
>>>>>>> REPLACE

The SEARCH text must match the CURRENT program exactly (including indentation), not one of
the other programs shown for inspiration.
Briefly explain your idea first, then give the edits."""

REWRITE_INSTRUCTIONS = """Your previous edit could not be applied: {error}

Instead, output the COMPLETE new version of the program in a single ```python code block.
It must define construct_packing() and include any helper functions and imports it needs."""


def build_prompt(program: Program, last_failure: str | None = None,
                 inspirations: list[Program] = ()) -> str:
    insp = ""
    if inspirations:
        insp = "\n# Other programs from the population (for inspiration; borrow ideas freely)\n"
        for i, p in enumerate(inspirations, 1):
            insp += f"\n## Program {chr(64 + i)} (score = {p.score:.6f})\n```python\n{get_block(p.code)}\n```\n"
    failure = ""
    if last_failure:
        failure = f"""
# Your previous attempt did not improve the program
{last_failure}
Avoid repeating this mistake.
"""
    return f"""{PROBLEM}
{insp}
# Current program to improve (score = {program.score:.6f})
```python
{get_block(program.code)}
```

# Evaluator feedback
{program.feedback}
{failure}
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
    ap.add_argument("--no-think", action="store_true", help="disable Qwen3 thinking mode (much faster)")
    ap.add_argument("--islands", type=int, default=4)
    ap.add_argument("--exploit", type=float, default=0.7, help="probability of picking the island best as parent")
    ap.add_argument("--migration-interval", type=int, default=20, help="iterations between migrations (0 = never)")
    ap.add_argument("--rng-seed", type=int, default=None)
    ap.add_argument("--inspirations", type=int, default=3, help="other programs shown in the prompt (0 = off)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    policy = make_policy(args.backend, args.model, args.base_url, no_think=args.no_think)
    db = ProgramDatabase(n_islands=args.islands, exploit_ratio=args.exploit,
                         migration_interval=args.migration_interval,
                         path=os.path.join(out_dir, "programs.jsonl"),
                         rng=random.Random(args.rng_seed))

    seed_code = open(args.seed).read()
    res = run_program(seed_code, args.timeout)
    db.add_seed(seed_code, res.score, res.feedback)
    best_score = res.score
    print(f"seed score {best_score:.6f}")

    t0 = time.time()
    n_samples = n_diff_ok = n_fallback = n_valid = 0
    last_failure = {}  #island -> failure message fed into that island's next prompt

    for it in range(1, args.iterations + 1):
        island = (it - 1) % args.islands
        parent, mode = db.sample_parent(island)
        inspirations = db.sample_inspirations(island, parent, args.inspirations)
        prompt = build_prompt(parent, last_failure.get(island), inspirations)
        rec = {"iter": it, "island": island, "selection": mode, "parent_id": parent.id,
               "parent_score": parent.score, "parent_generation": parent.generation,
               "inspiration_ids": [p.id for p in inspirations],
               "inspiration_scores": [p.score for p in inspirations], "prompt": prompt}
        try:
            comp = policy.generate(prompt)
        except Exception as e:
            print(f"[{it}] LLM error: {e}")
            continue
        n_samples += 1
        rec.update(completion=comp.text, reasoning=comp.reasoning,
                   input_tokens=comp.input_tokens, output_tokens=comp.output_tokens)

        child, diffs = None, parse_diffs(comp.text)
        rec["n_diffs"] = len(diffs)
        try:
            if not diffs:
                raise DiffError("no SEARCH/REPLACE blocks found")
            child = apply_diffs(parent.code, diffs)
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
                child = apply_full_rewrite(parent.code, fb.text)
            except Exception as e2:
                rec["fallback_error"] = str(e2)

        if child is None:
            rec.update(score=0.0, feedback="edit could not be applied", valid=False)
            last_failure[island] = f"Your edit could not be applied: {rec.get('fallback_error') or rec.get('diff_error')}"
        else:
            res = run_program(child, args.timeout)
            prog, is_new = db.add(child, res.score, res.feedback, res.valid, parent.id, island, it)
            rec.update(child_code=child, child_id=prog.id, duplicate=not is_new, score=res.score,
                       feedback=res.feedback, valid=res.valid, eval_runtime=res.runtime)
            n_valid += res.valid
            if res.score > parent.score:
                last_failure.pop(island, None)
            else:
                last_failure[island] = (f"Attempt scored {res.score:.6f} (parent scored {parent.score:.6f})."
                                        f" Evaluator feedback:\n{res.feedback}")
            if res.score > best_score:
                best_score = res.score
                with open(os.path.join(out_dir, "best.py"), "w") as f:
                    f.write(child)

        migrated = db.maybe_migrate(it)
        rec.update(best_score=best_score, n_samples=n_samples, wall_time=time.time() - t0, migrated=migrated)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[{it:3d}] I{island} {mode:7s} parent {parent.score:.4f} -> {rec['score']:.6f}  best {best_score:.6f}  "
              f"diff_ok {n_diff_ok}/{it}  valid {n_valid}/{it}  "
              f"{'(dup)' if rec.get('duplicate') else ''}"
              f"{'' if rec['valid'] else rec['feedback'].splitlines()[0][:50]}")
        if migrated:
            print(f"      migration -> {db.summary()}")

    print(f"\nbest score {best_score:.6f} after {n_samples} LLM samples, {time.time() - t0:.0f}s")
    print(f"diff-apply success {n_diff_ok}/{args.iterations}, full-rewrite fallbacks {n_fallback}, "
          f"valid programs {n_valid}/{args.iterations}")
    print(f"islands (best/size): {db.summary()}")
    print(f"logs: {out_dir}/log.jsonl  programs: {out_dir}/programs.jsonl  best: {out_dir}/best.py")


if __name__ == "__main__":
    main()
