import sys

from nltk import tokenize

remove_chars = "aeiouy"
sub_char = "_"

charfreq = {}
X = []
Y = []

for line in sys.stdin:
    sents = tokenize.sent_tokenize(line)
    for sent in sents:
        x = []
        y = []
        for c in sent:
            if c not in charfreq:
                charfreq[c] = 0
            charfreq[c] += 1
            if c in remove_chars:
                y.append(c)
                x.append(sub_char)
            else:
                x.append(c)

