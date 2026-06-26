from pathlib import Path

path = Path("frame0_dump.txt")

rows = []
with path.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([int(x) for x in line.split()])

missing_rows = [i for i in range(512) if i >= len(rows)]
rows_with_no_nonzero = []
rows_with_not_exactly_one = []
nonzero_values = set()

for i in range(min(512, len(rows))):
    nz = [v for v in rows[i] if v != 0]
    if not nz:
        rows_with_no_nonzero.append(i)
    if len(nz) != 1:
        rows_with_not_exactly_one.append((i, len(nz)))
    if len(nz) == 1:
        nonzero_values.add(nz[0])

print("missing_rows:", missing_rows[:20], "count", len(missing_rows))
print("rows_with_no_nonzero:", rows_with_no_nonzero[:20], "count", len(rows_with_no_nonzero))
print("rows_with_not_exactly_one:", rows_with_not_exactly_one[:20], "count", len(rows_with_not_exactly_one))
print("distinct_single_nonzero_values:", sorted(nonzero_values), "count", len(nonzero_values))
