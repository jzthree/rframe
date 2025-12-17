#!/usr/bin/env python3
"""
RFrame Usage Examples

Demonstrates how to use RFrame for R-like data manipulation in Python.
"""

import numpy as np
from rframe import (
    RFrame, data_frame, data_table, by,
    col, assign, delete, N, SD,
    match, which, which_max, which_min, isin,
    seq, seq_len, rep, rev, unique, duplicated,
    ifelse, is_na, na_omit, paste, paste0, c, order, table,
    head, tail, setdiff, intersect, union
)


def main():
    print("=" * 60)
    print("RFrame: R-like DataFrame operations in Python")
    print("=" * 60)

    # =========================================================================
    # 1. Creating DataFrames
    # =========================================================================
    print("\n1. CREATING DATAFRAMES")
    print("-" * 40)

    # Using data_frame() - like R's data.frame()
    df = data_frame(
        x=[1, 2, 3, 4, 5],
        y=['a', 'b', 'a', 'b', 'c'],
        z=[10.0, 20.0, 30.0, 40.0, 50.0]
    )
    print("df = data_frame(x=[1,2,3,4,5], y=['a','b','a','b','c'], z=[10,20,30,40,50])")
    print(df)

    # Using data_table() - same thing, just different name
    dt = data_table(id=[1, 2, 3], value=[100, 200, 300])
    print("\ndt = data_table(id=[1,2,3], value=[100,200,300])")
    print(dt)

    # =========================================================================
    # 2. Basic Properties (R-style)
    # =========================================================================
    print("\n2. R-STYLE PROPERTIES")
    print("-" * 40)

    print(f"nrow: {df.nrow}")           # R's nrow()
    print(f"ncol: {df.ncol}")           # R's ncol()
    print(f"colnames: {df.colnames}")   # R's colnames()
    print(f"names: {df.names}")         # R's names()
    print(f"shape: {df.shape}")

    # =========================================================================
    # 3. Column Access ($ style)
    # =========================================================================
    print("\n3. COLUMN ACCESS (df$column style)")
    print("-" * 40)

    print("df.x (like R's df$x):")
    print(df.x)

    print("\ndf['z'] (like R's df[['z']]):")
    print(df['z'])

    # =========================================================================
    # 4. Bracket Notation [i, j]
    # =========================================================================
    print("\n4. R-STYLE BRACKET NOTATION")
    print("-" * 40)

    # Row selection
    print("df[0, :] - First row:")
    print(df[0, :])

    print("\ndf[0:3, :] - First 3 rows:")
    print(df[0:3, :])

    # Column selection
    print("\ndf[:, 'x'] - Column x:")
    print(df[:, 'x'])

    print("\ndf[:, ['x', 'z']] - Columns x and z:")
    print(df[:, ['x', 'z']])

    # Row and column
    print("\ndf[0:2, ['x', 'y']] - First 2 rows, columns x and y:")
    print(df[0:2, ['x', 'y']])

    # =========================================================================
    # 5. Filtering Rows
    # =========================================================================
    print("\n5. FILTERING ROWS")
    print("-" * 40)

    print("df[df.x > 2, :] - Rows where x > 2:")
    print(df[df.x > 2, :])

    print("\ndf[df.y == 'a', :] - Rows where y == 'a':")
    print(df[df.y == 'a', :])

    # Combined conditions
    print("\ndf[(df.x > 1) & (df.y == 'b'), :] - x > 1 AND y == 'b':")
    print(df[(df.x > 1) & (df.y == 'b'), :])

    # =========================================================================
    # 6. data.table-style Grouped Operations
    # =========================================================================
    print("\n6. DATA.TABLE-STYLE GROUPED OPERATIONS")
    print("-" * 40)

    # Count per group (like dt[, .N, by=y])
    print("df[:, N, by('y')] - Count per group:")
    result = df[:, N, by('y')]
    print(result)

    # Aggregation per group
    print("\ndf[:, {'mean_z': 'mean(z)', 'sum_x': 'sum(x)'}, by('y')]:")
    result = df[:, {'mean_z': 'mean(z)', 'sum_x': 'sum(x)'}, by('y')]
    print(result)

    # =========================================================================
    # 7. Column Expressions
    # =========================================================================
    print("\n7. COLUMN EXPRESSIONS")
    print("-" * 40)

    # Using dict for new columns
    print("df[:, {'x_squared': 'x**2', 'z_half': 'z/2'}]:")
    result = df[:, {'x_squared': 'x**2', 'z_half': 'z/2'}]
    print(result)

    # Using col() for explicit column references
    print("\ndf[:, {'x_plus_z': col('x') + col('z')}]:")
    result = df[:, {'x_plus_z': col('x') + col('z')}]
    print(result)

    # =========================================================================
    # 8. Assignment by Reference (:=)
    # =========================================================================
    print("\n8. ASSIGNMENT BY REFERENCE (:=)")
    print("-" * 40)

    df2 = df.copy()
    print("Original df2:")
    print(df2)

    # Add new column
    df2[:, assign(w=col('x') * 10)]
    print("\nAfter df2[:, assign(w=col('x') * 10)]:")
    print(df2)

    # Delete column
    df2[:, delete('w')]
    print("\nAfter df2[:, delete('w')]:")
    print(df2)

    # =========================================================================
    # 9. R Utility Functions
    # =========================================================================
    print("\n9. R UTILITY FUNCTIONS")
    print("-" * 40)

    # match() - The function you specifically asked for!
    print("match([1, 2, 5], [5, 4, 3, 2, 1]):")
    result = match([1, 2, 5], [5, 4, 3, 2, 1])
    print(f"  Result: {result}")  # [4, 3, 0]

    print("\nmatch(['a', 'b', 'z'], ['a', 'b', 'c']):")
    result = match(['a', 'b', 'z'], ['a', 'b', 'c'])
    print(f"  Result: {result}")  # [0, 1, -1]

    # which()
    print("\nwhich([True, False, True, False, True]):")
    result = which([True, False, True, False, True])
    print(f"  Result: {result}")  # [0, 2, 4]

    print("\nwhich(df.x > 2):")
    result = which(df.x > 2)
    print(f"  Result: {result}")

    # which_max / which_min
    print("\nwhich_max([1, 5, 3, 5, 2]):")
    print(f"  Result: {which_max([1, 5, 3, 5, 2])}")  # 1

    print("\nwhich_min([3, 1, 2, 1, 4]):")
    print(f"  Result: {which_min([3, 1, 2, 1, 4])}")  # 1

    # isin (%in% operator)
    print("\nisin([1, 2, 5], [1, 2, 3, 4]):")
    result = isin([1, 2, 5], [1, 2, 3, 4])
    print(f"  Result: {result}")  # [True, True, False]

    # =========================================================================
    # 10. Sequence Functions
    # =========================================================================
    print("\n10. SEQUENCE FUNCTIONS")
    print("-" * 40)

    print("seq(1, 5):")
    print(f"  {seq(1, 5)}")

    print("\nseq(0, 1, by=0.2):")
    print(f"  {seq(0, 1, by=0.2)}")

    print("\nseq(1, 10, length_out=5):")
    print(f"  {seq(1, 10, length_out=5)}")

    print("\nseq_len(5):")
    print(f"  {seq_len(5)}")

    print("\nrep([1, 2], times=3):")
    print(f"  {rep([1, 2], times=3)}")

    print("\nrep([1, 2, 3], each=2):")
    print(f"  {rep([1, 2, 3], each=2)}")

    # =========================================================================
    # 11. Set Operations
    # =========================================================================
    print("\n11. SET OPERATIONS")
    print("-" * 40)

    a = [1, 2, 3, 4]
    b = [3, 4, 5, 6]

    print(f"a = {a}, b = {b}")
    print(f"setdiff(a, b): {setdiff(a, b)}")
    print(f"intersect(a, b): {intersect(a, b)}")
    print(f"union(a, b): {union(a, b)}")

    # =========================================================================
    # 12. String Functions
    # =========================================================================
    print("\n12. STRING FUNCTIONS")
    print("-" * 40)

    print("paste(['a', 'b'], [1, 2], sep='-'):")
    print(f"  {paste(['a', 'b'], [1, 2], sep='-')}")

    print("\npaste0('col', [1, 2, 3]):")
    print(f"  {paste0('col', [1, 2, 3])}")

    print("\npaste(['a', 'b', 'c'], collapse=','):")
    print(f"  {paste(['a', 'b', 'c'], collapse=',')}")

    # =========================================================================
    # 13. Other R Functions
    # =========================================================================
    print("\n13. OTHER R FUNCTIONS")
    print("-" * 40)

    # unique and duplicated
    x = [1, 2, 2, 3, 1, 4]
    print(f"x = {x}")
    print(f"unique(x): {unique(x)}")
    print(f"duplicated(x): {duplicated(x)}")

    # order
    print("\norder([3, 1, 4, 1, 5]):")
    print(f"  {order([3, 1, 4, 1, 5])}")

    # ifelse
    print("\nifelse([True, False, True], 'yes', 'no'):")
    print(f"  {ifelse([True, False, True], 'yes', 'no')}")

    # table (frequency counts)
    print("\ntable(['a', 'b', 'a', 'c', 'b', 'a']):")
    print(table(['a', 'b', 'a', 'c', 'b', 'a']))

    # c() function
    print("\nc(1, 2, 3):")
    print(f"  {c(1, 2, 3)}")

    print("\nc([1, 2], [3, 4], 5):")
    print(f"  {c([1, 2], [3, 4], 5)}")

    # =========================================================================
    # 14. NA Handling
    # =========================================================================
    print("\n14. NA HANDLING")
    print("-" * 40)

    x_with_na = [1, np.nan, 3, None, 5]
    print(f"x = {x_with_na}")
    print(f"is_na(x): {is_na(x_with_na)}")
    print(f"na_omit(x): {na_omit(x_with_na)}")

    # =========================================================================
    # 15. DataFrame Methods
    # =========================================================================
    print("\n15. DATAFRAME METHODS")
    print("-" * 40)

    print("df.head(3):")
    print(df.head(3))

    print("\ndf.tail(2):")
    print(df.tail(2))

    print("\ndf.order('z', decreasing=True):")
    print(df.order('-z'))

    print("\ndf.unique('y'):")
    print(df.unique('y'))

    print("\ndf.str():")
    print(df.str())

    # =========================================================================
    # 16. Merging (like R's merge)
    # =========================================================================
    print("\n16. MERGING")
    print("-" * 40)

    df1 = data_frame(id=[1, 2, 3], x=[10, 20, 30])
    df2 = data_frame(id=[2, 3, 4], y=[200, 300, 400])

    print("df1:")
    print(df1)
    print("\ndf2:")
    print(df2)

    print("\ndf1.merge(df2, on='id'):")
    print(df1.merge(df2, on='id'))

    print("\ndf1.merge(df2, on='id', how='left'):")
    print(df1.merge(df2, on='id', how='left'))

    # =========================================================================
    # 17. rbind and cbind
    # =========================================================================
    print("\n17. RBIND AND CBIND")
    print("-" * 40)

    df_a = data_frame(x=[1, 2], y=[3, 4])
    df_b = data_frame(x=[5, 6], y=[7, 8])

    print("df_a.rbind(df_b):")
    print(df_a.rbind(df_b))

    df_c = data_frame(z=[10, 20])
    print("\ndf_a.cbind(df_c):")
    print(df_a.cbind(df_c))

    # =========================================================================
    # 18. Transform and Within
    # =========================================================================
    print("\n18. TRANSFORM AND WITHIN")
    print("-" * 40)

    df = data_frame(x=[1, 2, 3, 4], y=[10, 20, 30, 40])
    print("Original df:")
    print(df)

    print("\ndf.transform(z=lambda d: d.x + d.y):")
    print(df.transform(z=lambda d: d.x + d.y))

    print("\ndf.within(z=lambda d: d.x * 2, w=lambda d: d.z + 100):")
    print(df.within(z=lambda d: d.x * 2, w=lambda d: d.z + 100))

    # =========================================================================
    # 19. Chaining Operations
    # =========================================================================
    print("\n19. CHAINING OPERATIONS")
    print("-" * 40)

    df = data_frame(
        x=[5, 2, 8, 1, 9, 3],
        y=['a', 'b', 'a', 'b', 'a', 'b']
    )

    result = (df
              .chain(
                  lambda d: d[d.x > 2, :],      # Filter
                  lambda d: d.order('-x'),      # Sort
                  lambda d: d.head(3)           # Take top 3
              ))

    print("Chain: filter x>2 -> order by -x -> head(3):")
    print(result)

    print("\n" + "=" * 60)
    print("That's the RFrame library! Enjoy R-like data manipulation.")
    print("=" * 60)


if __name__ == '__main__':
    main()
