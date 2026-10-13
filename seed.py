import json
import sys

import numpy as np

# EVOLVE-BLOCK-START
def construct_packing():
    n = 26
    # naive: points on a grid, equal radii
    g = np.linspace(0.1, 0.9, 6)
    centers = np.array([(x, y) for x in g for y in g])[:n]
    radii = np.full(n, 0.05)
    return centers, radii
# EVOLVE-BLOCK-END


# Fixed harness (outside the evolve block, never edited by the LLM).
# Scoring happens in the parent process; this only reports the packing.
if __name__ == "__main__":
    centers, radii = construct_packing()
    with open(sys.argv[1], "w") as f:
        json.dump({"centers": np.asarray(centers).tolist(), "radii": np.asarray(radii).tolist()}, f)
