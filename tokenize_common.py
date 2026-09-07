"""Tokenizer ported from the original ReChecker's vectorize_fragment.py
(FragmentVectorizer.tokenize), shared by both the baseline (Word2Vec) and
Phuong an C (FastText + segment) pipelines.
"""
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}'
}


def tokenize_line(line):
    tmp, w = [], []
    i = 0
    while i < len(line):
        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 3])
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        else:
            w.append(line[i])
            i += 1
    res = list(filter(lambda c: c != '', tmp))
    return list(filter(lambda c: c != ' ', res))


def tokenize_lines(lines):
    """Tokenize a list of cleaned code lines into one flat token list."""
    out = []
    for line in lines:
        out.extend(tokenize_line(line))
    return out
