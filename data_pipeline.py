"""
Shared data pipeline for both models (ReChecker goc va Phuong an C).
Reuses the ORIGINAL repo's parse_file() logic exactly, for maximum fidelity
to the real ReChecker training data. clean_fragment() normalization is
applied downstream by build_dataset.py, not here -- this module returns raw
(unnormalized) lines so build_dataset_raw.py can also consume them as-is.
"""
DATA_FILE = "reentrancy_1671.txt"


def parse_file(filename):
    """Verbatim copy of SmConVulDetector.py's parse_file()."""
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 33 in line and fragment:
                yield fragment, fragment_val
                fragment = []
            elif stripped.split()[0].isdigit():
                if fragment:
                    if stripped.isdigit():
                        fragment_val = int(stripped)
                    else:
                        fragment.append(stripped)
            else:
                fragment.append(stripped)


def split_into_functions(fragment):
    """Re-derive function-block boundaries within an already-flattened gadget,
    using the same heuristic as AutoExtractCode.py's split_function(): a new
    function starts wherever the first token of a line is 'function' or
    'constructor'. Returns a list of (start_idx, end_idx) blocks. Lines before
    the first such marker (rare) are treated as part of block 0.
    """
    starts = []
    for i, line in enumerate(fragment):
        toks = line.split()
        if toks and toks[0] in ("function", "constructor"):
            starts.append(i)
    if not starts:
        return [(0, len(fragment))]
    blocks = []
    if starts[0] != 0:
        blocks.append((0, starts[0]))
    for k in range(len(starts)):
        s = starts[k]
        e = starts[k + 1] if k + 1 < len(starts) else len(fragment)
        blocks.append((s, e))
    return blocks


def load_gadgets():
    """Returns list of dicts: {lines, label, w_lines, c_lines} where
    w_lines = lines of the FIRST function block (the W-function, per
    AutoExtractCode.py's construction order), c_lines = all subsequent
    function-block lines concatenated (the C-function(s), if any)."""
    gadgets = []
    for fragment, label in parse_file(DATA_FILE):
        blocks = split_into_functions(fragment)
        w_lines = fragment[blocks[0][0]:blocks[0][1]]
        c_lines = []
        for (s, e) in blocks[1:]:
            c_lines.extend(fragment[s:e])
        gadgets.append({
            "raw_lines": fragment,
            "label": label,
            "w_lines": w_lines,
            "c_lines": c_lines,
            "has_c": len(c_lines) > 0,
        })
    return gadgets


if __name__ == "__main__":
    gs = load_gadgets()
    print("total gadgets:", len(gs))
    labels = [g["label"] for g in gs]
    print("label=1 (vulnerable):", sum(labels), "label=0:", len(labels) - sum(labels))
    has_c = sum(1 for g in gs if g["has_c"])
    print("gadgets with a C-function block:", has_c, "/", len(gs))
    lens = sorted(len(g["raw_lines"]) for g in gs)
    print("line-count percentiles (0/25/50/75/90/99/100):",
          [lens[int(p / 100 * (len(lens) - 1))] for p in [0, 25, 50, 75, 90, 99, 100]])
