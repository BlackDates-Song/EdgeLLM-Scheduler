import os

SRC = "data/training_data.txt"
DST = "data/training_data_cleaned.txt"

os.makedirs(os.path.dirname(DST), exist_ok=True)

n_in, n_out = 0, 0
with open(SRC, 'r', encoding='utf-8') as f_in, open(DST, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        n_in += 1
        s = line.strip()
        if not s or "->" not in s:
            continue
        s = s.strip('"').replace('"', '')
        if not s.endswith("<END>"):
            s = s + " <END>"
        f_out.write(s + "\n")
        n_out += 1

print(f"Cleaned {n_in} lines, wrote {n_out} lines to {DST}")