Step 0: Pin down the problem

Pack n = 26 circles in the unit square, non-overlapping, and maximize the sum of their radii. AlphaEvolve reported about 2.635 here, slightly above the previous best of about 2.634. That number is your north star, but a small model getting to about 2.6 is already a good result.

The program you evolve doesn’t output coordinates directly. It’s a function construct_packing() that returns (centers, radii), and it’s allowed to run its own optimizer (numpy/scipy) inside. That’s how AlphaEvolve gets its gains: it evolves the search procedure, not the answer.

Step 1: Build the evaluator first

Everything else depends on this, so make it strict and fast.

python
def evaluate(centers, radii, n=26, tol=1e-9):
    if centers.shape != (n, 2) or radii.shape != (n,) or (radii < 0).any():
        return 0.0, "bad shapes or negative radii"
    # inside the square
    if ((centers - radii[:, None]) < -tol).any() or ((centers + radii[:, None]) > 1 + tol).any():
        return 0.0, "circle outside square"
    # pairwise overlap
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    gap = d - (radii[:, None] + radii[None])
    np.fill_diagonal(gap, np.inf)
    if (gap < -tol).any():
        i, j = np.argwhere(gap < -tol)[0]
        return 0.0, f"circles {i} and {j} overlap"
    return float(radii.sum()), "valid"

Two practical details:

Run each candidate in a subprocess with a timeout (say 60–120 s) and no network. Evolved programs will produce infinite loops and huge allocations.
Return the error message too. It becomes feedback in the next prompt (“circles 3 and 7 overlap”).

Watch for reward hacking. If your tolerance is loose, the LLM will find a way to exploit it. Keep tol tiny and fixed.

Step 2: Write a seed program

Start with something deliberately mediocre so you can see evolution make progress:

python
import numpy as np
# EVOLVE-BLOCK-START
def construct_packing():
    n = 26
    # naive: points on a jittered grid, equal radii
    g = np.linspace(0.1, 0.9, 6)
    centers = np.array([(x, y) for x in g for y in g])[:n]
    radii = np.full(n, 0.05)
    return centers, radii
# EVOLVE-BLOCK-END

This should score about 1.3. Fixed harness code (imports, the __main__ that calls the function and prints JSON) goes outside the markers so the LLM can’t break it.

Step 3: Get the LLM loop working with no database

Before adding anything clever, run the bare loop:

take the best program → prompt the LLM to improve it → apply the edit → evaluate → keep it if better

Model: serve something like Qwen3-4B or 8B with vLLM, or use an API model to prototype quickly.
Output format: ask for SEARCH/REPLACE diff blocks:
  <<<<<<< SEARCH
  radii = np.full(n, 0.05)
  =======
  radii = compute_max_radii(centers)
  >>>>>>> REPLACE

Parse these with a regex and apply them as exact string replacements. If a SEARCH block doesn’t match, fall back to asking for a full rewrite of the evolve block. Small models are bad at exact diffs, so log how often this happens.

Milestone: a curve of best score going up over about 100 iterations. If it’s flat, the issue is the prompt or the diff parsing, not the algorithm.

Step 4: Add the program database

This is what separates AlphaEvolve from greedy hill-climbing.

Store everything: code, score, feedback, parent id, and generation number.
Islands: keep about 4–5 subpopulations that evolve separately. Every so often, copy the best program from one island into another (migration). This prevents the whole run from collapsing onto one idea.
MAP-Elites grid (optional, add it later): bin programs by a couple of features, such as code length and score, and keep the best program per cell. This preserves structurally different solutions.
Parent selection: usually pick a top program (exploit), sometimes a random one (explore). Start with roughly 70/30.
Step 5: Build a richer prompt

Each prompt should include:

A problem description, with the target score if you want.
The parent program and its score.
2–3 “inspiration” programs from the database with their scores, both high-scoring and diverse ones.
The evaluator feedback for the parent.
The instruction to output diffs.

Point 3 is the in-context learning part: the model can see which programs scored well and borrow their ideas.

Step 6: Scale it up
Batch and async: generate many candidates in parallel with vLLM and evaluate them in a process pool. Throughput matters more than anything else here.
Evaluation cascade: a quick syntax check, then a short-timeout run, then the full evaluation. Most junk dies cheaply.
Model ensemble (optional): a small model for most samples and a larger one occasionally. This mirrors AlphaEvolve’s Flash/Pro split.
Step 7: Instrument it for the later comparison

Since ThetaEvolve and TTT-Discover will reuse this harness, log the following from day one:

best score vs. number of LLM samples,
best score vs. wall-clock time,
valid-program rate and diff-apply success rate,
every (prompt, completion, score) tuple.

The last item becomes your RL training data once you add ThetaEvolve. Keep the whole harness swappable: the environment, the database, and the “policy” (the LLM call) should be separate modules.