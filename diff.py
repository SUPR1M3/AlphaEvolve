import re

START = "# EVOLVE-BLOCK-START"
END = "# EVOLVE-BLOCK-END"

_DIFF_RE = re.compile(
    r"<<<<<<< SEARCH[ \t]*\n(.*?)\n?=======[ \t]*\n(.*?)\n?>>>>>>> REPLACE", re.DOTALL
)
_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


class DiffError(Exception):
    pass


def split_program(code: str):
    s, e = code.find(START), code.find(END)
    if s == -1 or e == -1 or e < s:
        raise DiffError("evolve-block markers not found")
    s += len(START)
    return code[:s], code[s:e], code[e:]


def get_block(code: str) -> str:
    return split_program(code)[1].strip("\n")


def parse_diffs(text: str):
    return [(m.group(1), m.group(2)) for m in _DIFF_RE.finditer(text)]


def _find_lenient(block: str, search: str):
    norm = lambda s: "\n".join(line.rstrip() for line in s.split("\n"))
    nblock, nsearch = norm(block), norm(search)
    idx = nblock.find(nsearch)
    if idx == -1 or not nsearch.strip():
        return None
    start_line = nblock[:idx].count("\n")
    n_lines = nsearch.count("\n") + 1
    lines = block.split("\n")
    start = sum(len(l) + 1 for l in lines[:start_line])
    end = start + sum(len(l) + 1 for l in lines[start_line:start_line + n_lines]) - 1
    return start, end


def apply_diffs(code: str, diffs) -> str:
    prefix, block, suffix = split_program(code)
    for i, (search, replace) in enumerate(diffs):
        if search and search in block:
            block = block.replace(search, replace, 1)
            continue
        span = _find_lenient(block, search)
        if span is None:
            raise DiffError(f"SEARCH block {i + 1} not found in program:\n{search[:300]}")
        block = block[:span[0]] + replace + block[span[1]:]
    return prefix + block + suffix


def apply_full_rewrite(code: str, text: str) -> str:
    blocks = _CODE_RE.findall(text)
    if not blocks:
        raise DiffError("no ```python code block in response")
    new = blocks[-1]
    if START in new and END in new:
        new = split_program(new)[1]
    if "def construct_packing" not in new:
        raise DiffError("rewrite does not define construct_packing()")
    prefix, _, suffix = split_program(code)
    return prefix + "\n" + new.strip("\n") + "\n" + suffix
