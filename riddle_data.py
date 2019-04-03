import sys

language = "english"
remove_chars = "aeiouy"
sub_char = "°"

lower = True

file_prefix = sys.argv[1]
X_file = file_prefix + ".X.txt"
Y_file = file_prefix + ".Y.txt"
freq_file = file_prefix + ".freq.txt"

charfreq = {}

with open(X_file, "w") as xout, open(Y_file, "w") as yout:
    for line in sys.stdin:
        line = line.strip()
        if lower:
            line = line.lower()
        x = []
        y = []
        for c in line:
            if c not in charfreq:
                charfreq[c] = 0
            charfreq[c] += 1
            if c in remove_chars:
                y.append(c)
                x.append(sub_char)
            else:
                x.append(c)
        print("".join(x), file=xout)
        print("".join(y), file=yout)

freqlist = [(k, charfreq[k]) for k in sorted(charfreq, key=charfreq.get, reverse=True)]

with open(freq_file, "w") as out:
    for c, f in freqlist:
        if c not in remove_chars:
            print(f"{c}\t{f}", file=out)
    print("###", file=out)
    for c, f in freqlist:
        if c in remove_chars:
            print(f"{c}\t{f}", file=out)

