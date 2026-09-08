"""Identifier normalization inherited from the documented ReChecker pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence


KEYWORDS = frozenset({
    "bool", "break", "case", "catch", "const", "continue", "default", "do",
    "double", "struct", "else", "enum", "payable", "function", "modifier",
    "emit", "export", "extern", "false", "constructor", "float", "if",
    "contract", "int", "long", "string", "super", "or", "private",
    "protected", "noReentrancy", "public", "return", "returns", "assert",
    "event", "indexed", "using", "require", "uint", "onlyDaoChallenge",
    "transfer", "Transfer", "Transaction", "switch", "pure", "view", "this",
    "throw", "true", "try", "revert", "bytes", "bytes4", "bytes32",
    "internal", "external", "union", "constant", "while", "for",
    "notExecuted", "NULL", "uint256", "uint128", "uint8", "uint16",
    "address", "call", "msg", "value", "sender", "notConfirmed", "onlyOwner",
    "onlyGovernor", "onlyCommittee", "onlyAdmin", "onlyPlayers", "ownerExists",
    "onlyManager", "onlyHuman", "only_owner", "onlyCongressMembers",
    "preventReentry", "noEther", "onlyMembers", "onlyProxyOwner", "confirmed",
    "mapping",
})
MAIN_DECLARATIONS = frozenset({"function", "constructor", "modifier", "contract"})
MAIN_ARGUMENTS = frozenset({"argc", "argv"})

COMMENT_CLOSE = re.compile(r"\*/\s*$")
FUNCTION_NAME = re.compile(r"\b([_A-Za-z]\w*)\b(?=\s*\()")
VARIABLE_NAME = re.compile(r"\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()")


def clean_fragment(fragment: Sequence[str]) -> list[str]:
    """Normalize user-defined function and variable names deterministically."""
    function_symbols: dict[str, str] = {}
    variable_symbols: dict[str, str] = {}
    cleaned_fragment: list[str] = []

    for line in fragment:
        if COMMENT_CLOSE.search(line) is not None:
            continue
        normalized = re.sub(r'".*?"', '""', line)
        normalized = re.sub(r"'.*?'", "''", normalized)
        normalized = re.sub(r"[^\x00-\x7f]", "", normalized)

        # Both candidate lists are collected before substitutions, matching the
        # released normalizer's order-dependent behavior.
        function_names = FUNCTION_NAME.findall(normalized)
        variable_names = VARIABLE_NAME.findall(normalized)
        for function_name in function_names:
            if function_name not in MAIN_DECLARATIONS and function_name not in KEYWORDS:
                symbol = function_symbols.setdefault(
                    function_name, f"FUN{len(function_symbols) + 1}"
                )
                normalized = re.sub(
                    rf"\b({function_name})\b(?=\s*\()", symbol, normalized
                )

        for variable_name in variable_names:
            if variable_name not in KEYWORDS and variable_name not in MAIN_ARGUMENTS:
                symbol = variable_symbols.setdefault(
                    variable_name, f"VAR{len(variable_symbols) + 1}"
                )
                normalized = re.sub(
                    rf"\b({variable_name})\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()",
                    symbol,
                    normalized,
                )

        cleaned_fragment.append(normalized)
    return cleaned_fragment
