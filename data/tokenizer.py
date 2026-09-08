"""Tokenizer ported from ReChecker's FragmentVectorizer."""

from __future__ import annotations

from collections.abc import Iterable


OPERATORS_3 = {"<<=", ">>="}
OPERATORS_2 = {
    "->", "++", "--", "!~", "<<", ">>", "<=", ">=", "==", "!=", "&&",
    "||", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
}
OPERATORS_1 = {
    "(", ")", "[", "]", ".", "+", "-", "*", "&", "/", "%", "<", ">",
    "^", "|", "=", ",", "?", ":", ";", "{", "}",
}


def tokenize_line(line: str) -> list[str]:
    pieces: list[str] = []
    word: list[str] = []
    index = 0
    while index < len(line):
        if line[index] == " ":
            pieces.extend(("".join(word), line[index]))
            word = []
            index += 1
        elif line[index:index + 3] in OPERATORS_3:
            pieces.extend(("".join(word), line[index:index + 3]))
            word = []
            index += 3
        elif line[index:index + 2] in OPERATORS_2:
            pieces.extend(("".join(word), line[index:index + 2]))
            word = []
            index += 2
        elif line[index] in OPERATORS_1:
            pieces.extend(("".join(word), line[index]))
            word = []
            index += 1
        else:
            word.append(line[index])
            index += 1
    # Preserve the original tokenizer behavior: trailing text without a delimiter is dropped.
    return [piece for piece in pieces if piece not in {"", " "}]


def tokenize_lines(lines: Iterable[str]) -> list[str]:
    return [token for line in lines for token in tokenize_line(line)]
