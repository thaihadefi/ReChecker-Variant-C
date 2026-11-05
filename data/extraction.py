"""W/C region extraction and diagnostics."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .tokenizer import tokenize_lines
from .types import ExtractionDiagnostics


FUNCTION_MARKERS = {"function", "constructor"}
BLOCK_MARKERS = {"function", "constructor", "fallback", "receive", "modifier"}
EXTRACTION_STRATEGIES = ("first", "call_anchored", "call_flow")
EXTRACTION_STRATEGY_DESCRIPTIONS = {
    "first": "first-function-declaration-block",
    "call_anchored": "call-anchored-function-blocks",
    "call_flow": "call-flow-sliced-w-and-c",
}

_DECL_NAME_RE = re.compile(
    r"^\s*(?:(function|modifier)\s+([A-Za-z_]\w*)|(constructor|fallback|receive)\b)"
)
_CONTAINER_RE = re.compile(r"^\s*(contract|library|interface)\s+([A-Za-z_]\w*)")
_LOW_LEVEL_CALL_RE = re.compile(
    r"(?:\.\s*(?:call|delegatecall|callcode|send|transfer|"
    r"functionCall|functionCallWithValue|uncheckedFunctionCall|sendValue|"
    r"onERC721Received|onERC1155Received|tokensReceived|notifyFillOrder)\b"
    r"|\bcall\s*\.\s*value\b"
    r"|\b(?:_safeMint|safeMint)\s*\()"
)
_HIGH_LEVEL_EXT_CALL_RE = re.compile(
    r"(?:\.\s*(?:transferFrom|safeTransfer|safeTransferFrom|"
    r"flashLoan|executeOperation|burn|mint|swap|withdraw|deposit|claim|borrow|repay|liquidate)\s*\("
    r"|\bI[A-Z]\w*\s*\([^)]*\)\s*\.\s*[A-Za-z_]\w*\s*\()"
)
_MEMBER_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*\(")
_BARE_CALL_RE = re.compile(r"(?<!\.)\b([A-Za-z_]\w*)\s*\(")
_VIEW_PURE_RE = re.compile(r"\b(?:view|pure)\b")
_NONREENTRANT_RE = re.compile(
    r"\b(?:nonReentrant|lock|mutex|noReentrancy|reentrancyGuard)\b", re.IGNORECASE
)
_PUBLIC_EXTERNAL_RE = re.compile(r"\b(?:public|external)\b")
_PRIVATE_INTERNAL_RE = re.compile(r"\b(?:internal|private)\b")
_LOCAL_DECL_RE = re.compile(
    r"^\s*(?:uint\d*|int\d*|bool|address|bytes\d*|string|mapping|"
    r"[A-Za-z_]\w*\s+(?:memory|storage|calldata))\b"
)
_ASSIGN_OP_RE = re.compile(
    r"(?:(?<![=!<>])=(?!=)|[+\-*/%&|^]=|\+\+|--|\bdelete\s+[A-Za-z_]\w*)"
)
_STATE_VAR_DECL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:=\s*[^;]+)?;\s*$")
_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")

_SAFE_RECEIVERS = {
    "abi",
    "math",
    "SafeMath",
    "SignedSafeMath",
    "Address",
    "Strings",
    "ECDSA",
    "EnumerableSet",
    "SafeERC20",
    "SafeCast",
    "console",
    "super",
}
_SAFE_METHODS = {
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "mulDiv",
    "tryAdd",
    "trySub",
    "tryMul",
    "tryDiv",
    "min",
    "max",
    "sqrt",
    "push",
    "pop",
    "length",
    "encode",
    "encodePacked",
    "encodeWithSelector",
    "encodeWithSignature",
    "decode",
    "toUint256",
    "toInt256",
    "toString",
    "recover",
}
_SOLIDITY_BUILTINS = {
    "if",
    "for",
    "while",
    "require",
    "assert",
    "revert",
    "emit",
    "return",
    "returns",
    "function",
    "modifier",
    "constructor",
    "fallback",
    "receive",
    "new",
    "delete",
    "keccak256",
    "sha256",
    "ripemd160",
    "ecrecover",
    "addmod",
    "mulmod",
    "selfdestruct",
    "suicide",
    "address",
    "uint",
    "uint256",
    "uint128",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
    "int",
    "int256",
    "bool",
    "bytes",
    "bytes32",
    "bytes4",
    "string",
    "payable",
    "type",
    "super",
    "this",
}


def split_first_function_regions(
    lines: Sequence[str],
) -> tuple[list[str], list[str], ExtractionDiagnostics]:
    """Preserve the historical first-function heuristic and expose its limitations.

    The source dataset does not carry explicit candidate metadata. A region begins at
    the first function/constructor declaration and ends at the next such declaration.
    Intervening modifier or contract declarations therefore remain visible in diagnostics.
    """
    starts = [
        index
        for index, line in enumerate(lines)
        if line.split() and line.split()[0] in FUNCTION_MARKERS
    ]
    if not starts:
        return list(lines), [], ExtractionDiagnostics()

    boundaries = starts + [len(lines)]
    blocks = [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:])
        if start < end
    ]
    w_start, w_end = blocks[0]
    w_lines = list(lines[w_start:w_end])
    c_lines = [line for start, end in blocks[1:] for line in lines[start:end]]
    first_tokens = [line.split()[0] for line in w_lines if line.split()]
    diagnostics = ExtractionDiagnostics(
        dropped_preamble_lines=w_start,
        w_contains_modifier_declaration="modifier" in first_tokens,
        w_contains_contract_declaration="contract" in first_tokens,
    )
    return w_lines, c_lines, diagnostics


@dataclass
class SolidityBlock:
    index: int
    kind: str
    name: str
    start: int
    end: int
    header: str
    container_kind: str
    is_interface: bool
    is_view_pure: bool
    is_nonreentrant: bool
    is_public_or_external: bool
    lines: list[str]
    token_est: int
    direct_low_level_calls: list[int] = field(default_factory=list)
    direct_high_level_calls: list[int] = field(default_factory=list)
    direct_member_calls: list[int] = field(default_factory=list)
    state_write_lines: list[int] = field(default_factory=list)
    callees: set[str] = field(default_factory=set)
    callers: set[str] = field(default_factory=set)
    state_vars_used: set[str] = field(default_factory=set)
    score: float = 0.0


def _build_diagnostics(w_lines: Sequence[str], dropped_preamble_lines: int) -> ExtractionDiagnostics:
    first_tokens = [line.split()[0] for line in w_lines if line.split()]
    return ExtractionDiagnostics(
        dropped_preamble_lines=dropped_preamble_lines,
        w_contains_modifier_declaration="modifier" in first_tokens,
        w_contains_contract_declaration="contract" in first_tokens,
    )


def _extract_state_var_names(state_lines: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for line in state_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(
            ("pragma ", "import ", "event ", "using ", "struct ", "enum ", "contract ", "library ", "interface ")
        ):
            continue
        match = _STATE_VAR_DECL_RE.search(stripped)
        if match:
            candidate = match.group(1)
            if candidate not in _SOLIDITY_BUILTINS and len(candidate) > 1:
                names.add(candidate)
    return names


def _parse_solidity_structure(
    lines: Sequence[str],
) -> tuple[list[SolidityBlock], list[str], int]:
    """Parse brace-matched function/modifier blocks and contract state lines."""
    blocks: list[SolidityBlock] = []
    covered = [False] * len(lines)

    # Track container kind (contract / library / interface) per line
    container_at_line = ["contract"] * len(lines)
    current_container = "contract"
    container_depth = 0
    for idx, line in enumerate(lines):
        m_cont = _CONTAINER_RE.match(line)
        if m_cont and container_depth <= 0:
            current_container = m_cont.group(1)
            container_depth = 0
        container_at_line[idx] = current_container
        container_depth += line.count("{") - line.count("}")

    idx = 0
    n_lines = len(lines)
    first_func_start = n_lines
    while idx < n_lines:
        line = lines[idx]
        tokens = line.split()
        first_tok = tokens[0] if tokens else ""
        if first_tok in BLOCK_MARKERS:
            if first_tok in FUNCTION_MARKERS and first_func_start == n_lines:
                first_func_start = idx
            m_decl = _DECL_NAME_RE.match(line)
            if m_decl:
                if m_decl.group(1):
                    kind = m_decl.group(1)
                    name = m_decl.group(2)
                else:
                    kind = m_decl.group(3)
                    name = m_decl.group(3)
            else:
                kind = first_tok
                name = first_tok

            depth = 0
            opened = False
            is_interface = False
            header_parts: list[str] = []
            end_idx = idx
            for scan in range(idx, n_lines):
                scan_line = lines[scan]
                if not opened:
                    header_parts.append(scan_line.strip())
                open_braces = scan_line.count("{")
                close_braces = scan_line.count("}")
                if open_braces > 0:
                    opened = True
                depth += open_braces - close_braces
                if opened and depth <= 0:
                    end_idx = scan + 1
                    break
                if not opened and ";" in scan_line:
                    is_interface = True
                    end_idx = scan + 1
                    break
                if not opened and scan > idx and scan_line.split() and scan_line.split()[0] in BLOCK_MARKERS:
                    is_interface = True
                    end_idx = scan
                    break
            else:
                end_idx = n_lines

            header = " ".join(header_parts)
            if "{" in header:
                header = header.split("{", 1)[0]
            cont_kind = container_at_line[idx]
            if cont_kind == "interface":
                is_interface = True

            block_lines = list(lines[idx:end_idx])
            for c_i in range(idx, end_idx):
                covered[c_i] = True

            is_view_pure = bool(_VIEW_PURE_RE.search(header))
            is_nonreentrant = bool(_NONREENTRANT_RE.search(header))
            is_pub_ext = (
                kind in ("constructor", "fallback", "receive")
                or bool(_PUBLIC_EXTERNAL_RE.search(header))
                or not bool(_PRIVATE_INTERNAL_RE.search(header))
            )
            tok_est = max(1, len(tokenize_lines(block_lines)))
            blocks.append(
                SolidityBlock(
                    index=len(blocks),
                    kind=kind,
                    name=name,
                    start=idx,
                    end=end_idx,
                    header=header,
                    container_kind=cont_kind,
                    is_interface=is_interface,
                    is_view_pure=is_view_pure,
                    is_nonreentrant=is_nonreentrant,
                    is_public_or_external=is_pub_ext,
                    lines=block_lines,
                    token_est=tok_est,
                )
            )
            idx = end_idx
            continue
        idx += 1

    state_lines: list[str] = []
    for i, line in enumerate(lines):
        if covered[i]:
            continue
        stripped = line.strip()
        if not stripped or stripped == "}":
            continue
        if stripped.startswith(("pragma ", "import ", "using ", "event ")):
            continue
        if container_at_line[i] == "interface":
            continue
        state_lines.append(line)

    if first_func_start == n_lines:
        first_func_start = 0
    return blocks, state_lines, first_func_start


def _analyze_and_score_blocks(
    blocks: list[SolidityBlock],
    state_lines: Sequence[str],
) -> None:
    """Populate call-graph edges, external call indicators, state writes, and scores."""
    state_var_names = _extract_state_var_names(state_lines)
    known_names = {
        b.name
        for b in blocks
        if not b.is_interface and b.name not in ("function", "constructor", "fallback", "receive")
    }
    has_concrete_contract_func = any(
        not b.is_interface and b.kind in ("function", "constructor", "fallback", "receive") and b.container_kind == "contract"
        for b in blocks
    )

    for block in blocks:
        if block.is_interface:
            continue
        # Detect custom modifiers in header
        for ident in _IDENT_RE.findall(block.header):
            if ident in known_names and ident != block.name:
                block.callees.add(ident)

        # Scan body lines (skipping line 0 if single-line header, or scanning all lines)
        for rel_idx, raw_line in enumerate(block.lines):
            stripped = raw_line.strip()
            # Combine with previous line when line starts with '.' for multiline member calls
            joined_line = (
                f"{block.lines[rel_idx - 1].strip()} {stripped}"
                if rel_idx > 0 and stripped.startswith(".")
                else stripped
            )

            if _LOW_LEVEL_CALL_RE.search(joined_line):
                block.direct_low_level_calls.append(rel_idx)
            elif _HIGH_LEVEL_EXT_CALL_RE.search(joined_line):
                block.direct_high_level_calls.append(rel_idx)
            else:
                for m_call in _MEMBER_CALL_RE.finditer(joined_line):
                    recv, method = m_call.group(1), m_call.group(2)
                    if recv not in _SAFE_RECEIVERS and method not in _SAFE_METHODS:
                        block.direct_member_calls.append(rel_idx)
                        break

            # Internal function calls
            for b_call in _BARE_CALL_RE.finditer(stripped):
                callee = b_call.group(1)
                if callee in known_names and callee != block.name and callee not in _SOLIDITY_BUILTINS:
                    block.callees.add(callee)

            # State variable usage
            if state_var_names:
                for ident in _IDENT_RE.findall(stripped):
                    if ident in state_var_names:
                        block.state_vars_used.add(ident)

            # State writes (excluding function header and require/assert/if conditions)
            if rel_idx > 0 and not stripped.startswith(("require(", "require (", "assert(", "assert (", "if(", "if (", "for(", "for (", "while(", "while (", "return ", "emit ")):
                if not _LOCAL_DECL_RE.match(stripped) and _ASSIGN_OP_RE.search(stripped):
                    block.state_write_lines.append(rel_idx)

    # Populate reverse callers map
    blocks_by_name: dict[str, list[SolidityBlock]] = {}
    for b in blocks:
        if not b.is_interface:
            blocks_by_name.setdefault(b.name, []).append(b)

    for b in blocks:
        for callee_name in b.callees:
            for target in blocks_by_name.get(callee_name, ()):
                target.callers.add(b.name)

    def _has_direct_ext(b: SolidityBlock) -> bool:
        return bool(
            b.direct_low_level_calls or b.direct_high_level_calls or b.direct_member_calls
        )

    def _has_strong_ext(b: SolidityBlock) -> bool:
        return bool(b.direct_low_level_calls or b.direct_high_level_calls)

    # Score each block (including modifiers so call-flow can rank modifiers with external calls)
    for b in blocks:
        if b.is_interface:
            b.score = -100.0
            continue

        # 1-hop and 2-hop callees
        callee_blocks: list[SolidityBlock] = []
        for c_name in b.callees:
            callee_blocks.extend(blocks_by_name.get(c_name, ()))
        two_hop_blocks: list[SolidityBlock] = list(callee_blocks)
        for cb in callee_blocks:
            for c2_name in cb.callees:
                two_hop_blocks.extend(blocks_by_name.get(c2_name, ()))

        transitive_strong = any(_has_strong_ext(cb) for cb in two_hop_blocks)
        transitive_any = any(_has_direct_ext(cb) for cb in two_hop_blocks)
        transitive_writes = any(bool(cb.state_write_lines) for cb in two_hop_blocks)

        has_ext = _has_direct_ext(b) or transitive_any
        if not has_ext:
            b.score = -10.0 if b.is_view_pure else 0.0
            continue

        score = 0.0
        if b.direct_low_level_calls:
            score += 40.0
        elif b.direct_high_level_calls:
            score += 30.0
        elif transitive_strong:
            score += 26.0
        elif b.direct_member_calls:
            score += 18.0
        elif transitive_any:
            score += 15.0

        all_call_lines = (
            b.direct_low_level_calls + b.direct_high_level_calls + b.direct_member_calls
        )
        # Also locate lines where a transitive external-calling helper is invoked
        if not all_call_lines and transitive_any:
            ext_callee_names = {
                cb.name for cb in callee_blocks if _has_direct_ext(cb)
            }
            for rel_idx, raw_line in enumerate(b.lines):
                if any(re.search(rf"\b{re.escape(cn)}\s*\(", raw_line) for cn in ext_callee_names):
                    all_call_lines.append(rel_idx)

        total_ext_ops = (
            len(b.direct_low_level_calls)
            + len(b.direct_high_level_calls)
            + len(b.direct_member_calls)
            + sum(1 for cb in callee_blocks if _has_direct_ext(cb))
        )

        has_writes = bool(b.state_write_lines) or transitive_writes
        if b.state_write_lines and all_call_lines and max(b.state_write_lines) > min(all_call_lines):
            # Classic Check-Effects-Interactions violation (state write after call)
            score += 25.0
        elif has_writes and total_ext_ops >= 2:
            # Cross-call / multi-external-call state update pattern (e.g. _safeMint + safeTransferFrom)
            score += 18.0
        elif has_writes:
            score += 10.0

        if b.callees:
            score += 5.0

        if b.is_public_or_external and not b.name.startswith("_"):
            score += 6.0
        elif b.name.startswith("_") and b.callers:
            score -= 5.0

        if b.is_view_pure:
            score -= 50.0
        if b.is_nonreentrant:
            score -= 5.0
        if has_concrete_contract_func and b.container_kind == "library":
            score -= 25.0

        b.score = score


def split_call_anchored_regions(
    lines: Sequence[str],
    w_budget: int = 299,
) -> tuple[list[str], list[str], ExtractionDiagnostics]:
    """Anchor W on the highest-scoring external-call function(s); keep C in contract order."""
    blocks, state_lines, first_func_start = _parse_solidity_structure(lines)
    concrete_funcs = [
        b for b in blocks if not b.is_interface and b.kind in ("function", "constructor", "fallback", "receive")
    ]
    if not concrete_funcs:
        return split_first_function_regions(lines)

    _analyze_and_score_blocks(blocks, state_lines)
    ranked = sorted(concrete_funcs, key=lambda b: (-b.score, b.index))
    if ranked[0].score <= 0:
        return split_first_function_regions(lines)

    w_indices: set[int] = set()
    w_lines: list[str] = []
    used_tokens = 0
    for cand in ranked:
        if cand.score <= 0:
            break
        if not w_indices or used_tokens + cand.token_est <= w_budget:
            w_indices.add(cand.index)
            w_lines.extend(cand.lines)
            used_tokens += cand.token_est
        if used_tokens >= w_budget:
            break

    c_lines: list[str] = []
    for b in blocks:
        if b.index not in w_indices and not b.is_interface:
            c_lines.extend(b.lines)
    if not c_lines and state_lines:
        c_lines.extend(state_lines)

    return w_lines, c_lines, _build_diagnostics(w_lines, first_func_start)


def split_call_flow_regions(
    lines: Sequence[str],
    w_budget: int = 299,
    c_budget: int = 200,
) -> tuple[list[str], list[str], ExtractionDiagnostics]:
    """Anchor W on top reentrancy-candidate functions (A) and slice C along A <-> B call/state flows."""
    blocks, state_lines, first_func_start = _parse_solidity_structure(lines)
    concrete_funcs = [
        b for b in blocks if not b.is_interface and b.kind in ("function", "constructor", "fallback", "receive")
    ]
    if not concrete_funcs:
        return split_first_function_regions(lines)

    _analyze_and_score_blocks(blocks, state_lines)
    ranked = sorted(concrete_funcs, key=lambda b: (-b.score, b.index))
    if ranked[0].score <= 0:
        return split_first_function_regions(lines)

    # 1. Pack W with top-scoring external-call entry functions
    w_indices: set[int] = set()
    w_blocks: list[SolidityBlock] = []
    w_lines: list[str] = []
    used_w_tokens = 0

    for cand in ranked:
        if cand.score <= 0:
            break
        if not w_indices or used_w_tokens + cand.token_est <= w_budget:
            w_indices.add(cand.index)
            w_blocks.append(cand)
            w_lines.extend(cand.lines)
            used_w_tokens += cand.token_est
        if used_w_tokens >= w_budget:
            break

    # 2. Prioritize C by call-flow neighborhood (A -> B callees, callers, modifiers)
    #    and shared state variables (A and B touching the same storage vars).
    w_callees: set[str] = set()
    w_callers: set[str] = set()
    w_state_vars: set[str] = set()
    for wb in w_blocks:
        w_callees |= wb.callees
        w_callers |= wb.callers
        w_state_vars |= wb.state_vars_used

    # Include 2-hop callees of W
    for b in blocks:
        if b.name in w_callees and not b.is_interface:
            w_callees |= b.callees

    c_Direct_flow: list[SolidityBlock] = []
    c_other_candidates: list[SolidityBlock] = []
    c_shared_state: list[SolidityBlock] = []
    c_rest: list[SolidityBlock] = []

    for b in blocks:
        if b.index in w_indices or b.is_interface:
            continue
        if (b.name in w_callees or b.name in w_callers) and not b.is_view_pure:
            c_Direct_flow.append(b)
        elif b.score > 0:
            c_other_candidates.append(b)
        elif w_state_vars and (b.state_vars_used & w_state_vars) and not b.is_view_pure:
            c_shared_state.append(b)
        else:
            c_rest.append(b)

    # Sort call-flow neighbors and high-score candidates so callback/external-call flows come first
    c_Direct_flow.sort(
        key=lambda b: (
            -(1 if b.score > 0 else 0),
            -(1 if b.kind == "modifier" and b.name in w_callees else 0),
            -b.score,
            -len(b.state_write_lines),
            b.index,
        )
    )
    c_other_candidates.sort(key=lambda b: (-b.score, b.index))
    c_shared_state.sort(
        key=lambda b: (-len(b.state_vars_used & w_state_vars), -len(b.state_write_lines), b.index)
    )

    c_lines: list[str] = []
    used_c_tokens = 0
    for group in (c_Direct_flow, c_other_candidates, c_shared_state):
        for b in group:
            c_lines.extend(b.lines)
            used_c_tokens += b.token_est

    # Include state variable declarations before low-priority boilerplate if C budget has room
    if state_lines and used_c_tokens < c_budget:
        c_lines.extend(state_lines)
        used_c_tokens += len(tokenize_lines(state_lines))

    for b in c_rest:
        c_lines.extend(b.lines)

    if state_lines and used_c_tokens >= c_budget and not any(line in c_lines for line in state_lines):
        c_lines.extend(state_lines)

    return w_lines, c_lines, _build_diagnostics(w_lines, first_func_start)


def split_regions(
    lines: Sequence[str],
    strategy: str = "first",
    w_budget: int = 299,
    c_budget: int = 200,
) -> tuple[list[str], list[str], ExtractionDiagnostics]:
    """Dispatch W/C extraction according to the chosen strategy."""
    if strategy == "first":
        return split_first_function_regions(lines)
    if strategy == "call_anchored":
        return split_call_anchored_regions(lines, w_budget=w_budget)
    if strategy == "call_flow":
        return split_call_flow_regions(lines, w_budget=w_budget, c_budget=c_budget)
    raise ValueError(
        f"unknown extraction strategy {strategy!r}; expected one of {EXTRACTION_STRATEGIES}"
    )
