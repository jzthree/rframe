# rframe

**R-style DataFrame operations in Python.**

`rframe` brings the familiar syntax of R's `data.frame` and `data.table` to Python. If you've ever missed R's bracket notation, `match()` function, or `data.table`'s `[i, j, by]` syntax while working in Python, this library is for you.

Built on top of pandas for performance and compatibility, but with an interface that feels like home for R users.

## Installation

```bash
pip install rframe
```

Or from source:
```bash
git clone https://github.com/jzthree/rframe.git
cd rframe
pip install -e .
```

## Quick Start

```python
from rframe import data_frame, match, which, by

# Create a data frame (just like R)
df = data_frame(
    x = [1, 2, 3, 4, 5],
    y = ['a', 'b', 'a', 'b', 'c'],
    z = [10, 20, 30, 40, 50]
)

# R-style bracket notation
df[0, :]              # First row
df[:, 'x']            # Column x
df[df.x > 2, :]       # Filter: rows where x > 2
df[:, ['x', 'z']]     # Select columns

# Grouped operations (data.table style)
df[:, {'mean_z': 'mean(z)'}, by('y')]

# R's match function
match([1, 2, 5], [5, 4, 3, 2, 1])  # Returns: [4, 3, 0]
```

## Why rframe?

| What you want | pandas | rframe |
|---------------|--------|--------|
| First row | `df.iloc[0]` | `df[0, :]` |
| Column 'x' | `df['x']` or `df.loc[:, 'x']` | `df[:, 'x']` or `df.x` |
| Filter rows | `df[df['x'] > 2]` | `df[df.x > 2, :]` |
| Group + aggregate | `df.groupby('y')['z'].mean()` | `df[:, {'mean_z': 'mean(z)'}, by('y')]` |
| Find positions | *(no direct equivalent)* | `match(needle, haystack)` |

---

## Core Concepts

### 1. The `[i, j]` Bracket Notation

The fundamental operation in R is `df[rows, cols]`. rframe brings this to Python:

```python
from rframe import data_frame

df = data_frame(x=[1,2,3,4,5], y=['a','b','a','b','c'], z=[10,20,30,40,50])

# Select rows (i)
df[0, :]           # Single row by index
df[0:3, :]         # Slice of rows
df[[0, 2, 4], :]   # Specific rows by index

# Select columns (j)
df[:, 'x']         # Single column (returns Series)
df[:, ['x', 'y']]  # Multiple columns (returns RFrame)
df[:, 0:2]         # Columns by position

# Both together
df[0:3, ['x', 'z']]  # First 3 rows, columns x and z
```

### 2. Filtering Rows

Filter using boolean conditions, just like R:

```python
# Single condition
df[df.x > 2, :]

# Multiple conditions (use & for AND, | for OR)
df[(df.x > 1) & (df.y == 'b'), :]
df[(df.x < 2) | (df.x > 4), :]

# Using isin (R's %in%)
from rframe import isin
df[isin(df.y, ['a', 'c']), :]
```

### 3. The `match()` Function

One of R's most useful functions, finally in Python:

```python
from rframe import match, NOMATCH

# Find positions of first argument in second argument
match([1, 2, 3], [3, 2, 1])
# Returns: [2, 1, 0]  (0-indexed)

# Elements not found get NOMATCH (max int64)
result = match(['a', 'b', 'z'], ['a', 'b', 'c'])
# Returns: [0, 1, 9223372036854775807]

# Filter to valid matches
valid_idx = result[result < NOMATCH]

# Use for table lookups
names = ['alice', 'bob', 'charlie']
scores = [95, 87, 92]
lookup = ['bob', 'alice']
idx = match(lookup, names)
[scores[i] for i in idx]  # [87, 95]
```

### 4. Grouped Operations (data.table style)

The powerful `[i, j, by]` syntax from data.table:

```python
from rframe import data_frame, by, N

df = data_frame(
    group = ['A', 'A', 'B', 'B', 'B'],
    value = [10, 20, 30, 40, 50]
)

# Count per group (like .N in data.table)
df[:, N, by('group')]
#   group  N
# 0     A  2
# 1     B  3

# Aggregate per group
df[:, {'total': 'sum(value)', 'avg': 'mean(value)'}, by('group')]
#   group  total   avg
# 0     A     30  15.0
# 1     B    120  40.0

# Filter + group
df[df.value > 15, {'n': 'n()'}, by('group')]
```

### 5. Column Operations

Create and modify columns:

```python
from rframe import col, assign, delete

df = data_frame(x=[1, 2, 3], y=[10, 20, 30])

# Compute new columns using expressions
df[:, {'z': 'x + y', 'w': 'x * 2'}]

# Using col() for explicit references
df[:, {'ratio': col('y') / col('x')}]

# Modify in place with assign (like := in data.table)
df[:, assign(z=col('x') + col('y'))]

# Delete columns
df[:, delete('z')]
```

---

## Essential R Functions

rframe includes 30+ R utility functions. Here are the most commonly used:

### Finding & Matching

```python
from rframe import match, which, which_max, which_min, isin, NOMATCH

# match: find positions in a lookup table
match([1, 2, 5], [5, 4, 3, 2, 1])  # [4, 3, 0]

# which: indices where condition is True
which([True, False, True, False])  # [0, 2]
which(df.x > 2)                     # indices where x > 2

# which_max/which_min: index of max/min value
which_max([1, 5, 3, 5, 2])  # 1 (first occurrence)
which_min([3, 1, 2, 1, 4])  # 1

# isin: R's %in% operator
isin([1, 2, 5], [1, 2, 3, 4])  # [True, True, False]
```

### Sequences & Repetition

```python
from rframe import seq, seq_len, rep, c

# seq: generate sequences
seq(1, 5)                # [1, 2, 3, 4, 5]
seq(0, 1, by=0.2)        # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
seq(1, 10, length_out=5) # [1, 3.25, 5.5, 7.75, 10]

# seq_len: sequence from 0 to n-1
seq_len(5)  # [0, 1, 2, 3, 4]

# rep: replicate elements
rep([1, 2], times=3)  # [1, 2, 1, 2, 1, 2]
rep([1, 2], each=3)   # [1, 1, 1, 2, 2, 2]

# c: combine values (like R's c())
c(1, 2, 3)           # array([1, 2, 3])
c([1, 2], [3, 4], 5) # array([1, 2, 3, 4, 5])
```

### Set Operations

```python
from rframe import unique, duplicated, setdiff, intersect, union

x = [1, 2, 2, 3, 1, 4]

unique(x)      # [1, 2, 3, 4] - preserves order
duplicated(x)  # [False, False, True, False, True, False]

a, b = [1, 2, 3], [2, 3, 4]
setdiff(a, b)   # [1]     - in a but not b
intersect(a, b) # [2, 3]  - in both
union(a, b)     # [1, 2, 3, 4]
```

### NA Handling

```python
from rframe import is_na, na_omit, complete_cases, NA
import numpy as np

x = [1, np.nan, 3, None, 5]

is_na(x)       # [False, True, False, True, False]
na_omit(x)     # [1, 3, 5]

# Check complete cases across multiple vectors
complete_cases([1, np.nan, 3], [4, 5, 6])  # [True, False, True]
```

### Conditionals & Transformations

```python
from rframe import ifelse, order, rank, cut

# ifelse: vectorized if-else
ifelse([True, False, True], 'yes', 'no')  # ['yes', 'no', 'yes']
ifelse(df.x > 2, 'high', 'low')

# order: indices that would sort the array
order([3, 1, 2])  # [1, 2, 0]

# rank: compute ranks
rank([30, 10, 20])  # [3, 1, 2]

# cut: bin continuous values
cut([1, 5, 10, 15], breaks=[0, 5, 10, 20])
# ['(0, 5]', '(0, 5]', '(5, 10]', '(10, 20]']
```

---

## DataFrame Methods

RFrame objects have familiar methods:

```python
df = data_frame(x=[3,1,4,1,5], y=['b','a','c','a','d'])

# Inspection
df.nrow          # 5
df.ncol          # 2
df.colnames      # ['x', 'y']
df.shape         # (5, 2)
df.head(3)       # First 3 rows
df.tail(2)       # Last 2 rows
df.str()         # Structure summary (like R's str())

# Sorting
df.order('x')           # Sort by x ascending
df.order('-x')          # Sort by x descending
df.order('y', '-x')     # Sort by y asc, then x desc

# Deduplication
df.unique()             # Unique rows
df.unique('y')          # Unique by column y

# Combining
df1.rbind(df2)          # Row bind (stack vertically)
df1.cbind(df2)          # Column bind (stack horizontally)
df1.merge(df2, on='id') # Join on column

# Transformation
df.transform(z=lambda d: d.x * 2)
df.within(z=lambda d: d.x * 2, w=lambda d: d.z + 1)  # Can reference new cols

# Conversion
df.to_pandas()          # Get underlying pandas DataFrame
df.to_dict()            # Convert to dictionary
df.to_csv('file.csv')   # Write to CSV
```

---

## Comparison with R

| R | rframe | Notes |
|---|--------|-------|
| `df[1, ]` | `df[0, :]` | Python is 0-indexed |
| `df[, "x"]` | `df[:, 'x']` | |
| `df$x` | `df.x` | |
| `df[df$x > 2, ]` | `df[df.x > 2, :]` | |
| `match(x, table)` | `match(x, table)` | Returns 0-indexed positions |
| `x %in% y` | `isin(x, y)` | |
| `which(cond)` | `which(cond)` | 0-indexed |
| `dt[, .N, by=g]` | `df[:, N, by('g')]` | |
| `dt[, .(sum=sum(x)), by=g]` | `df[:, {'sum': 'sum(x)'}, by('g')]` | |
| `dt[, x := y*2]` | `df[:, assign(x=col('y')*2)]` | |
| `c(1, 2, 3)` | `c(1, 2, 3)` | |
| `seq(1, 10, by=2)` | `seq(1, 10, by=2)` | |
| `rep(1:3, times=2)` | `rep([1,2,3], times=2)` | |
| `NA` | `NA` or `np.nan` | |

---

## Tips for R Users

1. **Indexing is 0-based**: Python uses 0-based indexing. `df[0, :]` is the first row.

2. **Use `:` not nothing**: In R, `df[, "x"]` works. In Python, use `df[:, 'x']`.

3. **Parentheses for conditions**: Python needs explicit `&` and `|` with parentheses:
   ```python
   # R: df[x > 1 & y == "a", ]
   # Python:
   df[(df.x > 1) & (df.y == 'a'), :]
   ```

4. **`match()` returns NOMATCH, not NA**: We use max int64 instead of NA for non-matches. This causes an IndexError if accidentally used, making bugs obvious.

5. **by() is a function**: Use `by('col')` not `by=col`.

---

## License

MIT

---

## Contributing

Issues and PRs welcome at [github.com/jzthree/rframe](https://github.com/jzthree/rframe).
