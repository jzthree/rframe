# rframe API Reference

Complete reference for all rframe functions and classes.

---

## Table of Contents

1. [RFrame Class](#rframe-class)
2. [Constructors](#constructors)
3. [Bracket Notation](#bracket-notation)
4. [Special Symbols](#special-symbols)
5. [Column Operations](#column-operations)
6. [R Utility Functions](#r-utility-functions)
   - [Matching & Finding](#matching--finding)
   - [Sequences](#sequences)
   - [Set Operations](#set-operations)
   - [Array Utilities](#array-utilities)
   - [String Operations](#string-operations)
   - [NA Handling](#na-handling)
   - [Binning & Tables](#binning--tables)

---

## RFrame Class

The core class that wraps pandas DataFrame with R-like syntax.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `df` | `pd.DataFrame` | The underlying pandas DataFrame |
| `nrow` | `int` | Number of rows |
| `ncol` | `int` | Number of columns |
| `shape` | `tuple` | (nrow, ncol) |
| `colnames` | `list` | Column names (settable) |
| `rownames` | `list` | Row names/index (settable) |
| `names` | `list` | Alias for colnames |

### Methods

#### Inspection

##### `head(n=6)`
Return first n rows.

```python
df.head(3)
```

##### `tail(n=6)`
Return last n rows.

```python
df.tail(3)
```

##### `str()`
Return structure description (like R's `str()`).

```python
print(df.str())
# RFrame: 5 obs. of 3 variables:
#  $ x: int64 - 1, 2, 3, 4, 5
#  $ y: object - a, b, a, b, c
#  $ z: float64 - 10.0, 20.0, 30.0, 40.0, 50.0
```

##### `summary()`
Return summary statistics (transposed describe).

```python
df.summary()
```

#### Sorting

##### `order(*cols, decreasing=False)`
Sort rows by columns. Prefix column name with `-` for descending.

```python
df.order('x')           # Ascending by x
df.order('-x')          # Descending by x
df.order('group', '-x') # By group asc, then x desc
```

#### Deduplication

##### `unique(*cols)`
Return unique rows. If cols specified, uniqueness based on those columns only.

```python
df.unique()        # Unique across all columns
df.unique('x')     # Unique values of x
df.unique('x','y') # Unique combinations of x and y
```

#### Merging & Binding

##### `merge(other, on=None, how='inner', suffixes=('_x', '_y'))`
Merge with another RFrame (like R's `merge()`).

```python
df1.merge(df2, on='id')
df1.merge(df2, on=['id', 'date'], how='left')
df1.merge(df2, on='id', how='outer')
```

**Parameters:**
- `other`: RFrame or DataFrame to merge with
- `on`: Column(s) to join on
- `how`: 'inner', 'left', 'right', 'outer'
- `suffixes`: Suffixes for overlapping column names

##### `rbind(*others)`
Row bind (stack vertically). Like R's `rbind()`.

```python
df1.rbind(df2)
df1.rbind(df2, df3, df4)
```

##### `cbind(*others)`
Column bind (stack horizontally). Like R's `cbind()`.

```python
df1.cbind(df2)
df1.cbind(df2, df3)
```

#### Transformation

##### `transform(**kwargs)`
Add/modify columns. Each kwarg is `column_name=expression`.

```python
df.transform(
    z=lambda d: d.x + d.y,
    w=lambda d: d.x * 2
)
```

##### `within(**kwargs)`
Like transform, but expressions can reference newly created columns.

```python
df.within(
    z=lambda d: d.x * 2,
    w=lambda d: d.z + 100  # Can use z!
)
```

##### `subset(condition=None, select=None, drop=False)`
Subset rows and/or columns (like R's `subset()`).

```python
df.subset(lambda d: d.x > 2)
df.subset(lambda d: d.x > 2, select=['x', 'y'])
```

#### Aggregation

##### `aggregate(formula=None, by=None, func=None, **kwargs)`
Aggregate data (like R's `aggregate()`).

```python
df.aggregate(by='group', x='mean', y='sum')
df.aggregate('value ~ group', func=np.mean)
```

#### Apply Functions

##### `apply(func, axis=0, **kwargs)`
Apply function along axis (like R's `apply()`).

```python
df.apply(np.mean, axis=0)  # Column means
df.apply(sum, axis=1)       # Row sums
```

##### `lapply(func)`
Apply function to each column (like R's `lapply()`).

```python
df.lapply(lambda x: x.mean())
```

##### `sapply(func)`
Apply function to each column and simplify (like R's `sapply()`).

```python
df.sapply(lambda x: x.mean())  # Returns Series
```

##### `mapply(func, *cols)`
Apply function element-wise across multiple columns.

```python
df.mapply(lambda x, y: x + y, 'a', 'b')
```

#### Chaining

##### `chain(*operations)`
Chain multiple operations (like data.table's `DT[...][...]`).

```python
df.chain(
    lambda d: d[d.x > 0, :],
    lambda d: d.order('-x'),
    lambda d: d.head(10)
)
```

#### I/O

##### `to_pandas()`
Return a copy of the underlying pandas DataFrame.

##### `to_dict(orient='list')`
Convert to dictionary.

##### `to_csv(path, **kwargs)`
Write to CSV file.

##### `RFrame.read_csv(path, **kwargs)` (classmethod)
Read from CSV file.

```python
df = RFrame.read_csv('data.csv')
```

##### `RFrame.from_pandas(df)` (classmethod)
Create from pandas DataFrame.

---

## Constructors

### `data_frame(**kwargs)`
Create an RFrame from keyword arguments (like R's `data.frame()`).

```python
df = data_frame(
    x=[1, 2, 3],
    y=['a', 'b', 'c'],
    z=[10.0, 20.0, 30.0]
)
```

### `data_table(**kwargs)`
Alias for `data_frame()`. For users who prefer data.table naming.

```python
dt = data_table(x=[1, 2, 3], y=[4, 5, 6])
```

### `RFrame(data=None, **kwargs)`
Create an RFrame from various inputs.

```python
RFrame({'x': [1,2,3], 'y': [4,5,6]})  # From dict
RFrame(pandas_df)                      # From pandas DataFrame
RFrame(x=[1,2,3], y=[4,5,6])          # From kwargs
```

### Aliases

- `DF` = `data_frame`
- `DT` = `data_table`
- `DataFrame` = `RFrame`
- `DataTable` = `RFrame`

---

## Bracket Notation

### Basic Syntax: `df[i, j]`

```python
# i = row selection
# j = column selection

df[i, j]
```

### Row Selection (i)

| Syntax | Description | Example |
|--------|-------------|---------|
| `int` | Single row by index | `df[0, :]` |
| `slice` | Range of rows | `df[0:5, :]` |
| `list` | Specific rows | `df[[0, 2, 4], :]` |
| `bool array` | Filter by condition | `df[df.x > 2, :]` |
| `None` or `:` | All rows | `df[:, 'x']` |

### Column Selection (j)

| Syntax | Description | Example |
|--------|-------------|---------|
| `str` | Single column (returns Series) | `df[:, 'x']` |
| `list[str]` | Multiple columns | `df[:, ['x', 'y']]` |
| `slice` | Columns by position | `df[:, 0:2]` |
| `int` | Single column by position | `df[:, 0]` |
| `dict` | Compute new columns | `df[:, {'z': 'x + y'}]` |
| `None` or `:` | All columns | `df[0, :]` |

### Grouped Operations: `df[i, j, by]`

```python
from rframe import by, N

# Count per group
df[:, N, by('group')]

# Aggregate per group
df[:, {'sum': 'sum(x)', 'mean': 'mean(x)'}, by('group')]

# Multiple grouping columns
df[:, {'n': 'n()'}, by('group1', 'group2')]
```

### Expression Syntax in `j`

When `j` is a dict, values can be:

| Type | Example | Description |
|------|---------|-------------|
| String expression | `'x + y'` | Evaluated with column names as variables |
| String aggregation | `'mean(x)'` | Aggregation function on column |
| `col()` expression | `col('x') + col('y')` | Explicit column references |
| Lambda | `lambda d: d.x.mean()` | Function receiving RFrame |
| Scalar | `42` | Constant value |

**Available functions in string expressions:**
`mean`, `sum`, `min`, `max`, `sd`/`std`, `var`, `median`, `abs`, `sqrt`, `log`, `log10`, `exp`, `n`/`N` (count)

---

## Special Symbols

Import from `rframe`:

```python
from rframe import N, SD, I, GRP, BY, NGRP
```

### `N`
Number of rows in the current group (or total if ungrouped).

```python
df[:, N, by('group')]  # Count per group
```

### `SD`
Subset of Data - all columns for the current group.

```python
df[:, SD, by('group')]
```

### `I`
Row indices (0-indexed) in the original data.

### `GRP`
Current group number (0-indexed).

### `BY`
Current grouping values.

### `NGRP`
Total number of groups.

---

## Column Operations

### `col(name)`
Create a column reference for expressions.

```python
from rframe import col

df[:, {'z': col('x') + col('y')}]
df[:, {'ratio': col('y') / col('x')}]
```

**Supported operations:** `+`, `-`, `*`, `/`, `//`, `%`, `**`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `&`, `|`, `~`

### `assign(**kwargs)`
Create assignment expressions (like `:=` in data.table). Modifies in place.

```python
from rframe import assign, col

df[:, assign(z=col('x') + col('y'))]
df[:, assign(
    w=col('x') * 2,
    v=100
)]
```

### `delete(*cols)`
Delete columns (like `:= NULL` in data.table).

```python
from rframe import delete

df[:, delete('temp_col')]
df[:, delete('a', 'b', 'c')]
```

---

## R Utility Functions

### Matching & Finding

#### `match(x, table, nomatch=NOMATCH)`
Find positions of `x` values in `table`. Returns 0-indexed positions.

```python
from rframe import match, NOMATCH

match([1, 2, 3], [3, 2, 1])  # [2, 1, 0]
match([1, 5], [1, 2, 3])     # [0, NOMATCH]

# Custom nomatch value
match([1, 5], [1, 2, 3], nomatch=-1)  # [0, -1]
```

**Parameters:**
- `x`: Values to find
- `table`: Values to search in
- `nomatch`: Value for non-matches (default: max int64)

#### `NOMATCH`
Constant for non-matches (max int64 = 9223372036854775807). Causes IndexError if used as index.

```python
result = match([1, 99], [1, 2, 3])
valid = result[result < NOMATCH]  # Filter valid matches
```

#### `which(condition)`
Return indices where condition is True.

```python
which([True, False, True])  # [0, 2]
which(df.x > 2)             # [2, 3, 4]
```

#### `which_max(x)`
Return index of first maximum value.

```python
which_max([1, 5, 3, 5, 2])  # 1
```

#### `which_min(x)`
Return index of first minimum value.

```python
which_min([3, 1, 2, 1, 4])  # 1
```

#### `isin(x, table)`
Test if elements of x are in table (R's `%in%`).

```python
isin([1, 2, 5], [1, 2, 3, 4])  # [True, True, False]
```

---

### Sequences

#### `seq(from_, to, by=1.0, length_out=None)`
Generate regular sequences.

```python
seq(1, 5)               # [1, 2, 3, 4, 5]
seq(0, 1, by=0.2)       # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
seq(1, 10, length_out=5) # [1, 3.25, 5.5, 7.75, 10]
```

#### `seq_len(length_out)`
Generate sequence from 0 to length_out-1.

```python
seq_len(5)  # [0, 1, 2, 3, 4]
```

#### `seq_along(x)`
Generate sequence along the length of x.

```python
seq_along(['a', 'b', 'c'])  # [0, 1, 2]
```

#### `rep(x, times=1, each=1, length_out=None)`
Replicate elements.

```python
rep([1, 2, 3], times=2)      # [1, 2, 3, 1, 2, 3]
rep([1, 2, 3], each=2)       # [1, 1, 2, 2, 3, 3]
rep([1, 2], length_out=5)    # [1, 2, 1, 2, 1]
rep(0, times=5)              # [0, 0, 0, 0, 0]
```

#### `rev(x)`
Reverse a vector.

```python
rev([1, 2, 3])  # [3, 2, 1]
```

#### `c(*args, **kwargs)`
Combine values into a vector (like R's `c()`).

```python
c(1, 2, 3)              # array([1, 2, 3])
c([1, 2], [3, 4], 5)    # array([1, 2, 3, 4, 5])
c(a=1, b=2, c=3)        # {'a': 1, 'b': 2, 'c': 3}
```

---

### Set Operations

#### `unique(x, return_index=False)`
Return unique values preserving order of first occurrence.

```python
unique([3, 1, 2, 1, 3])  # [3, 1, 2]

# With indices
values, indices = unique([3, 1, 2, 1], return_index=True)
```

#### `duplicated(x, from_last=False)`
Identify duplicate elements.

```python
duplicated([1, 2, 1, 3, 2])  # [False, False, True, False, True]
duplicated([1, 2, 1], from_last=True)  # [True, False, False]
```

#### `setdiff(x, y)`
Set difference: elements in x but not in y.

```python
setdiff([1, 2, 3, 4], [2, 4])  # [1, 3]
```

#### `intersect(x, y)`
Set intersection: elements in both x and y.

```python
intersect([1, 2, 3], [2, 3, 4])  # [2, 3]
```

#### `union(x, y)`
Set union: all unique elements from x and y.

```python
union([1, 2, 3], [2, 3, 4])  # [1, 2, 3, 4]
```

---

### Array Utilities

#### `head(x, n=6)`
Return first n elements.

```python
head([1,2,3,4,5,6,7,8], 3)  # [1, 2, 3]
```

#### `tail(x, n=6)`
Return last n elements.

```python
tail([1,2,3,4,5,6,7,8], 3)  # [6, 7, 8]
```

#### `length(x)`
Return length of x.

```python
length([1, 2, 3])  # 3
```

#### `order(*args, decreasing=False, na_last=True)`
Return indices that would sort the input.

```python
order([3, 1, 2])                    # [1, 2, 0]
order([3, 1, 2], decreasing=True)   # [0, 2, 1]
order([2, 1, 2], [3, 2, 1])         # Sort by first, then second
```

#### `rank(x, ties_method='average', na_last=True)`
Compute ranks of values.

```python
rank([30, 10, 20])           # [3, 1, 2]
rank([30, 10, 10, 20])       # [4, 1.5, 1.5, 3] (average ties)
rank([30, 10, 10], ties_method='first')  # [3, 1, 2]
```

**ties_method options:** 'average', 'first', 'min', 'max', 'dense'

---

### String Operations

#### `paste(*args, sep=' ', collapse=None)`
Concatenate strings with separator.

```python
paste(['a', 'b'], [1, 2], sep='-')     # ['a-1', 'b-2']
paste(['a', 'b', 'c'], collapse=',')   # 'a,b,c'
paste('x', [1, 2, 3], sep='_')         # ['x_1', 'x_2', 'x_3']
```

#### `paste0(*args, collapse=None)`
Concatenate strings with no separator.

```python
paste0(['col'], [1, 2, 3])  # ['col1', 'col2', 'col3']
```

---

### NA Handling

#### `NA`
Constant representing missing value (alias for `np.nan`).

```python
from rframe import NA
x = [1, NA, 3]
```

#### `is_na(x)`
Test for NA/NaN/None values.

```python
is_na([1, np.nan, None, 3])  # [False, True, True, False]
```

#### `na_omit(x)`
Remove NA values.

```python
na_omit([1, np.nan, 3, None, 5])  # [1, 3, 5]
```

#### `complete_cases(*args)`
Find indices with no NA values across all inputs.

```python
complete_cases([1, np.nan, 3], [4, 5, 6])  # [True, False, True]
```

---

### Conditionals

#### `ifelse(condition, yes, no)`
Vectorized conditional.

```python
ifelse([True, False, True], 'yes', 'no')  # ['yes', 'no', 'yes']
ifelse(df.x > 2, 'high', 'low')
ifelse(df.x > 2, df.x, 0)  # Vectorized values
```

---

### Binning & Tables

#### `cut(x, breaks, labels=None, right=True, include_lowest=False)`
Divide continuous variable into intervals.

```python
cut([1, 5, 10, 15, 20], breaks=[0, 5, 10, 20])
# ['(0, 5]', '(0, 5]', '(5, 10]', '(10, 20]', '(10, 20]']

cut([1, 5, 10], breaks=3)  # 3 equal-width bins

cut([1, 5, 10, 15], breaks=[0, 5, 10, 20], labels=['low', 'med', 'high'])
# ['low', 'low', 'med', 'high']
```

#### `table(*args)`
Build frequency table.

```python
table(['a', 'b', 'a', 'c', 'b', 'a'])
# a    3
# b    2
# c    1

# Cross-tabulation
table(['a', 'a', 'b', 'b'], [1, 2, 1, 2])
```

---

## by() Function

Create a grouping clause for `[i, j, by]` syntax.

```python
from rframe import by

df[:, N, by('group')]
df[:, {'sum': 'sum(x)'}, by('g1', 'g2')]
df[:, {'n': 'n()'}, by(['g1', 'g2'])]  # List also works
```

---

## Type Conversions

### RFrame to other types

```python
df.to_pandas()       # pandas DataFrame
df.to_dict()         # dict
df.to_dict('records') # list of dicts
```

### Other types to RFrame

```python
RFrame(pandas_df)          # From pandas
RFrame({'x': [1,2], 'y': [3,4]})  # From dict
RFrame.read_csv('file.csv')      # From CSV
```

---

## Index

### Classes
- `RFrame` - Main DataFrame class
- `ByClause` - Grouping clause (created by `by()`)
- `ColRef` - Column reference (created by `col()`)
- `AssignmentExpr` - Assignment expression (created by `assign()`)

### Constructors
`data_frame`, `data_table`, `RFrame`, `DF`, `DT`

### Special Symbols
`N`, `SD`, `I`, `GRP`, `BY`, `NGRP`, `NOMATCH`

### Column Operations
`col`, `assign`, `delete`, `by`

### R Functions
`match`, `which`, `which_max`, `which_min`, `isin`, `seq`, `seq_len`, `seq_along`, `rep`, `rev`, `c`, `unique`, `duplicated`, `setdiff`, `intersect`, `union`, `head`, `tail`, `length`, `order`, `rank`, `paste`, `paste0`, `ifelse`, `is_na`, `na_omit`, `complete_cases`, `NA`, `cut`, `table`
