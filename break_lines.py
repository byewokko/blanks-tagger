import sys

char_limit = int(sys.argv[1])

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    chunk = []
    chunk_len = 0
    for word in line.split(" "):
        if len(word) + chunk_len > char_limit:
            print(" ".join(chunk))
            chunk = []
            chunk_len = 0
        chunk.append(word)
        chunk_len += len(word)
    splitpar = " ".join(chunk)
    if splitpar:
        print(splitpar)
