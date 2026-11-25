import collections, os, sys
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
c = collections.Counter()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep  # the engine folder
for l in lines:
    if "error" in l and not l.startswith("FAILED"):
        c[l.replace(root, "")[:240]] += 1
for k, v in c.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 80):
    print(v, k)
