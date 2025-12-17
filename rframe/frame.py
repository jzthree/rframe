"""
RFrame - An R-like DataFrame class built on pandas.

Provides data.frame and data.table-like syntax and operations.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from functools import wraps
import warnings

from .special import (
    N, SD, I, GRP, BY, NGRP,
    _N, _SD, _I, _GRP, _BY, _NGRP,
    ColRef, BinaryOp, UnaryOp, AssignmentExpr, Exclude
)


class RFrame:
    """
    An R-like DataFrame class with data.table syntax.

    Wraps pandas DataFrame but provides R-style operations:
    - Bracket notation: df[i, j] for row/column selection
    - data.table syntax: df[i, j, by=...] for grouped operations
    - Assignment by reference: df[:, assign(new_col=expr)]
    - Special symbols: .N, .SD, .I, .GRP

    Parameters
    ----------
    data : dict, pd.DataFrame, or array-like, optional
        Data to initialize the frame
    **kwargs
        Column name/value pairs

    Examples
    --------
    >>> from rframe import RFrame, col
    >>> df = RFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    >>> df[0, :]  # First row
    >>> df[:, 'x']  # Column x
    >>> df[df.x > 1, :]  # Filter rows
    >>> df[:, {'mean_x': 'mean(x)'}, by='group']  # Grouped aggregation
    """

    def __init__(self, data=None, **kwargs):
        if data is None and kwargs:
            data = kwargs
        elif data is not None and kwargs:
            if isinstance(data, dict):
                data = {**data, **kwargs}
            else:
                raise ValueError("Cannot combine positional data with keyword arguments")

        if isinstance(data, pd.DataFrame):
            self._df = data.copy()
        elif isinstance(data, RFrame):
            self._df = data._df.copy()
        elif data is None:
            self._df = pd.DataFrame()
        else:
            self._df = pd.DataFrame(data)

    # =========================================================================
    # Core Properties
    # =========================================================================

    @property
    def df(self) -> pd.DataFrame:
        """Access the underlying pandas DataFrame."""
        return self._df

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (nrow, ncol)."""
        return self._df.shape

    @property
    def nrow(self) -> int:
        """Number of rows (R-style)."""
        return len(self._df)

    @property
    def ncol(self) -> int:
        """Number of columns (R-style)."""
        return len(self._df.columns)

    @property
    def colnames(self) -> List[str]:
        """Column names (R-style)."""
        return list(self._df.columns)

    @colnames.setter
    def colnames(self, names: List[str]):
        """Set column names."""
        self._df.columns = names

    @property
    def rownames(self) -> List:
        """Row names/index (R-style)."""
        return list(self._df.index)

    @rownames.setter
    def rownames(self, names: List):
        """Set row names."""
        self._df.index = names

    @property
    def names(self) -> List[str]:
        """Alias for colnames (R-style)."""
        return self.colnames

    @names.setter
    def names(self, value: List[str]):
        self.colnames = value

    # =========================================================================
    # Column Access ($ style)
    # =========================================================================

    def __getattr__(self, name: str) -> pd.Series:
        """
        Allow df.column_name access (like R's df$column).

        Examples
        --------
        >>> df.x  # Equivalent to df['x'] or R's df$x
        """
        if name.startswith('_') or name in ('df', 'shape', 'nrow', 'ncol'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if name in self._df.columns:
            return self._df[name]
        raise AttributeError(f"Column '{name}' not found")

    def __setattr__(self, name: str, value: Any):
        """Allow df.column_name = value assignment."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        elif hasattr(self, '_df') and not name.startswith('_'):
            self._df[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name: str):
        """Allow del df.column_name."""
        if name in self._df.columns:
            del self._df[name]
        else:
            raise AttributeError(f"Column '{name}' not found")

    # =========================================================================
    # Main Bracket Notation [i, j, by]
    # =========================================================================

    def __getitem__(self, key) -> Union['RFrame', pd.Series, Any]:
        """
        R-style bracket notation: df[i, j] or df[i, j, by=...]

        In R/data.table:
        - df[i] selects rows
        - df[i, j] selects rows i and columns j
        - df[i, j, by=...] groups by columns and applies j

        In Python, we use:
        - df[i, :] for row selection
        - df[:, j] for column selection
        - df[i, j] for both
        - df[:, j, by] is handled via the `by()` method or tuple syntax

        Parameters
        ----------
        key : various
            - Single value: column selection
            - Tuple of (i, j): row and column selection
            - Tuple of (i, j, by): grouped operation

        Examples
        --------
        >>> df[0, :]  # First row
        >>> df[:, 'x']  # Column 'x'
        >>> df[:, ['x', 'y']]  # Multiple columns
        >>> df[df.x > 1, :]  # Filter rows where x > 1
        >>> df[0:5, 'x']  # First 5 rows of column x
        """
        # Handle tuple (i, j) or (i, j, by)
        if isinstance(key, tuple):
            return self._bracket_select(key)

        # Single key - treat as column selection (like df['col'] in pandas)
        # But also allow boolean arrays for row filtering (like df[mask])
        if isinstance(key, str):
            return self._df[key]
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            return RFrame(self._df[key])
        elif isinstance(key, (pd.Series, np.ndarray)) and key.dtype == bool:
            return RFrame(self._df[key])
        elif isinstance(key, slice):
            return RFrame(self._df.iloc[key])
        else:
            # Try as column selection
            return self._df[key]

    def _bracket_select(self, key: tuple) -> Union['RFrame', pd.Series, Any]:
        """Handle df[i, j] and df[i, j, by] syntax."""
        if len(key) == 2:
            i, j = key
            by = None
        elif len(key) == 3:
            i, j, by = key
        else:
            raise ValueError(f"Expected 2 or 3 elements, got {len(key)}")

        # Handle 'by' - accept ByClause, string, or list of strings directly
        if isinstance(by, ByClause):
            by = by.cols
        elif isinstance(by, str):
            by = [by]
        elif isinstance(by, (list, tuple)) and by and isinstance(by[0], str):
            by = list(by)

        # Process row selection (i)
        df_filtered = self._process_rows(i)

        # If we have a by clause, do grouped operation
        if by is not None:
            return self._grouped_operation(df_filtered, j, by)

        # Process column selection (j)
        return self._process_cols(df_filtered, j)

    def _process_rows(self, i) -> pd.DataFrame:
        """Process row selection."""
        if i is None or (isinstance(i, slice) and i == slice(None)):
            return self._df

        if isinstance(i, slice):
            return self._df.iloc[i]
        elif isinstance(i, (list, np.ndarray, pd.Series)):
            if len(i) > 0:
                if isinstance(i[0], bool) or (hasattr(i, 'dtype') and i.dtype == bool):
                    return self._df[i]
                else:
                    return self._df.iloc[i]
            return self._df.iloc[i]
        elif isinstance(i, int):
            return self._df.iloc[[i]]
        elif callable(i):
            mask = i(self)
            return self._df[mask]
        else:
            return self._df.iloc[i]

    def _process_cols(self, df: pd.DataFrame, j) -> Union['RFrame', pd.Series, Any]:
        """Process column selection."""
        if j is None or (isinstance(j, slice) and j == slice(None)):
            return RFrame(df)

        # String column name
        if isinstance(j, str):
            return df[j]

        # List of column names
        if isinstance(j, list):
            if all(isinstance(c, str) for c in j):
                return RFrame(df[j])
            # Could be list of AssignmentExpr
            if all(isinstance(c, AssignmentExpr) for c in j):
                return self._apply_assignments(df, j)

        # Exclude columns (like R's -c("col1", "col2"))
        if isinstance(j, Exclude):
            keep_cols = [c for c in df.columns if c not in j.cols]
            return RFrame(df[keep_cols])

        # Dict for renaming/computing columns
        if isinstance(j, dict):
            return self._compute_columns(df, j)

        # Single AssignmentExpr or list
        if isinstance(j, AssignmentExpr):
            return self._apply_assignments(df, [j])
        if isinstance(j, list) and j and isinstance(j[0], AssignmentExpr):
            return self._apply_assignments(df, j)

        # Slice for column indices
        if isinstance(j, slice):
            return RFrame(df.iloc[:, j])

        # Integer for column index
        if isinstance(j, int):
            return df.iloc[:, j]

        # ColRef or expression
        if isinstance(j, (ColRef, BinaryOp, UnaryOp)):
            return self._eval_expr(df, j)

        # Special symbols
        if isinstance(j, _N):
            return len(df)
        if isinstance(j, _I):
            return np.arange(len(df))
        if isinstance(j, _SD):
            return RFrame(df)

        # Callable
        if callable(j):
            return j(RFrame(df))

        return RFrame(df[j])

    def _grouped_operation(self, df: pd.DataFrame, j, by) -> 'RFrame':
        """Handle grouped operations (data.table's DT[i, j, by])."""
        if isinstance(by, str):
            by = [by]

        grouped = df.groupby(by, sort=False)

        # Handle different j expressions
        if isinstance(j, _N):
            # Count per group
            result = grouped.size().reset_index(name='N')
            return RFrame(result)

        if isinstance(j, _SD):
            # Return all columns per group (basically the same data)
            return RFrame(df)

        if isinstance(j, dict):
            return self._grouped_compute(grouped, j, by)

        if isinstance(j, str):
            # Simple aggregation expression
            return self._grouped_agg_expr(grouped, j, by)

        if j is None or (isinstance(j, slice) and j == slice(None)):
            # Just grouping info
            result = grouped.size().reset_index(name='N')
            return RFrame(result)

        # Callable
        if callable(j):
            results = []
            for name, group in grouped:
                res = j(RFrame(group))
                if isinstance(res, RFrame):
                    res = res._df
                elif isinstance(res, pd.Series):
                    res = res.to_frame().T
                elif isinstance(res, dict):
                    res = pd.DataFrame([res])
                elif np.isscalar(res):
                    res = pd.DataFrame([{by[0] if isinstance(by, list) else by: name, 'V1': res}])
                results.append(res)
            return RFrame(pd.concat(results, ignore_index=True))

        return RFrame(df)

    def _grouped_compute(self, grouped, j: dict, by) -> 'RFrame':
        """Compute expressions per group."""
        results = []

        for group_key, group_df in grouped:
            row = {}
            # Add group key(s)
            if isinstance(by, str):
                row[by] = group_key
            else:
                for k, v in zip(by, group_key if isinstance(group_key, tuple) else [group_key]):
                    row[k] = v

            # Compute each expression
            for col_name, expr in j.items():
                row[col_name] = self._eval_agg_expr(group_df, expr)

            results.append(row)

        return RFrame(pd.DataFrame(results))

    def _grouped_agg_expr(self, grouped, expr: str, by) -> 'RFrame':
        """Parse and apply simple aggregation expression."""
        # Parse expressions like "mean(x)", "sum(y)", etc.
        import re
        match = re.match(r'(\w+)\((\w+)\)', expr)
        if match:
            func_name, col_name = match.groups()
            agg_result = grouped[col_name].agg(func_name).reset_index()
            agg_result.columns = list(agg_result.columns[:-1]) + [f'{func_name}_{col_name}']
            return RFrame(agg_result)

        # Just return the expression as-is
        return RFrame(grouped.agg(expr).reset_index())

    def _eval_agg_expr(self, df: pd.DataFrame, expr) -> Any:
        """Evaluate an aggregation expression on a DataFrame."""
        if callable(expr):
            return expr(RFrame(df))

        if isinstance(expr, str):
            import re

            # Check for no-argument functions first: n(), count()
            match_no_arg = re.match(r'(\w+)\(\s*\)', expr)
            if match_no_arg:
                func_name = match_no_arg.group(1)
                if func_name in ('n', 'count', 'nrow'):
                    return len(df)

            # Parse "func(col)" pattern
            match = re.match(r'(\w+)\((\w+)\)', expr)
            if match:
                func_name, col_name = match.groups()
                if func_name == 'mean':
                    return df[col_name].mean()
                elif func_name == 'sum':
                    return df[col_name].sum()
                elif func_name == 'min':
                    return df[col_name].min()
                elif func_name == 'max':
                    return df[col_name].max()
                elif func_name == 'sd' or func_name == 'std':
                    return df[col_name].std()
                elif func_name == 'var':
                    return df[col_name].var()
                elif func_name == 'median':
                    return df[col_name].median()
                elif func_name == 'first':
                    return df[col_name].iloc[0]
                elif func_name == 'last':
                    return df[col_name].iloc[-1]
                elif func_name == 'n' or func_name == 'count':
                    return len(df)
                elif func_name == 'n_distinct' or func_name == 'nunique':
                    return df[col_name].nunique()

            # Check for special .N
            if expr == '.N' or expr == 'N':
                return len(df)

            # Try as column name
            if expr in df.columns:
                return df[expr].iloc[0] if len(df) == 1 else df[expr].tolist()

        if isinstance(expr, _N):
            return len(df)

        if isinstance(expr, (ColRef, BinaryOp, UnaryOp)):
            result = self._eval_expr(df, expr)
            if isinstance(result, pd.Series):
                return result.iloc[0] if len(result) == 1 else result.tolist()
            return result

        return expr

    def _eval_expr(self, df: pd.DataFrame, expr) -> Any:
        """Evaluate a column expression."""
        if isinstance(expr, ColRef):
            return df[expr.name]

        if isinstance(expr, BinaryOp):
            left = self._eval_expr(df, expr.left)
            right = self._eval_expr(df, expr.right)

            ops = {
                '+': lambda a, b: a + b,
                '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a / b,
                '//': lambda a, b: a // b,
                '%': lambda a, b: a % b,
                '**': lambda a, b: a ** b,
                '==': lambda a, b: a == b,
                '!=': lambda a, b: a != b,
                '<': lambda a, b: a < b,
                '<=': lambda a, b: a <= b,
                '>': lambda a, b: a > b,
                '>=': lambda a, b: a >= b,
                '&': lambda a, b: a & b,
                '|': lambda a, b: a | b,
            }
            return ops[expr.op](left, right)

        if isinstance(expr, UnaryOp):
            operand = self._eval_expr(df, expr.operand)
            if expr.op == '~':
                return ~operand
            elif expr.op == '-':
                return -operand

        return expr

    def _compute_columns(self, df: pd.DataFrame, j: dict) -> 'RFrame':
        """Compute new columns from dict specification."""
        result = df.copy()

        for col_name, expr in j.items():
            if isinstance(expr, str):
                # Parse expression string
                result[col_name] = self._eval_string_expr(df, expr)
            elif isinstance(expr, (ColRef, BinaryOp, UnaryOp)):
                result[col_name] = self._eval_expr(df, expr)
            elif callable(expr):
                result[col_name] = expr(RFrame(df))
            else:
                result[col_name] = expr

        return RFrame(result)

    def _eval_string_expr(self, df: pd.DataFrame, expr: str) -> Any:
        """Evaluate a string expression."""
        # Create local namespace with columns
        local_ns = {col: df[col] for col in df.columns}
        local_ns['np'] = np
        local_ns['pd'] = pd

        # Add common functions
        local_ns.update({
            'mean': lambda x: x.mean() if hasattr(x, 'mean') else np.mean(x),
            'sum': lambda x: x.sum() if hasattr(x, 'sum') else np.sum(x),
            'min': lambda x: x.min() if hasattr(x, 'min') else np.min(x),
            'max': lambda x: x.max() if hasattr(x, 'max') else np.max(x),
            'sd': lambda x: x.std() if hasattr(x, 'std') else np.std(x),
            'std': lambda x: x.std() if hasattr(x, 'std') else np.std(x),
            'var': lambda x: x.var() if hasattr(x, 'var') else np.var(x),
            'median': lambda x: x.median() if hasattr(x, 'median') else np.median(x),
            'abs': np.abs,
            'sqrt': np.sqrt,
            'log': np.log,
            'log10': np.log10,
            'exp': np.exp,
            'n': lambda: len(df),
            'N': len(df),
        })

        try:
            return eval(expr, {"__builtins__": {}}, local_ns)
        except Exception as e:
            raise ValueError(f"Could not evaluate expression '{expr}': {e}")

    def _apply_assignments(self, df: pd.DataFrame, assignments: List[AssignmentExpr]) -> 'RFrame':
        """Apply := assignments (modifies in place and returns self)."""
        for assign in assignments:
            if assign.is_delete:
                if assign.col_name in self._df.columns:
                    del self._df[assign.col_name]
            else:
                value = assign.value
                if isinstance(value, (ColRef, BinaryOp, UnaryOp)):
                    value = self._eval_expr(self._df, value)
                elif callable(value):
                    value = value(self)
                self._df[assign.col_name] = value

        return self

    def __setitem__(self, key, value):
        """Allow df['col'] = value or df[i, 'col'] = value."""
        if isinstance(key, str):
            self._df[key] = value
        elif isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(j, str):
                if i is None or (isinstance(i, slice) and i == slice(None)):
                    self._df[j] = value
                else:
                    df_view = self._process_rows(i)
                    self._df.loc[df_view.index, j] = value
        else:
            self._df[key] = value

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"RFrame ({self.nrow} x {self.ncol})\n{self._df.__repr__()}"

    def __str__(self) -> str:
        return self.__repr__()

    # =========================================================================
    # Chaining Support (data.table's DT[...][...])
    # =========================================================================

    def chain(self, *operations) -> 'RFrame':
        """
        Chain multiple operations (like data.table's DT[...][...]).

        Parameters
        ----------
        *operations : callable
            Functions that take RFrame and return RFrame

        Examples
        --------
        >>> df.chain(
        ...     lambda d: d[d.x > 0, :],
        ...     lambda d: d[:, ['x', 'y']]
        ... )
        """
        result = self
        for op in operations:
            result = op(result)
        return result

    # =========================================================================
    # data.table-style methods
    # =========================================================================

    def by(self, *cols) -> 'ByClause':
        """
        Create a by clause for grouped operations.

        Examples
        --------
        >>> df[:, {'mean_x': 'mean(x)'}, df.by('group')]
        """
        return ByClause(cols)

    def order(self, *cols, decreasing: bool = False) -> 'RFrame':
        """
        Order/sort rows (like data.table's setorder or order()).

        Parameters
        ----------
        *cols : str
            Column names to sort by (prefix with '-' for descending)
        decreasing : bool, default False
            Sort in descending order

        Examples
        --------
        >>> df.order('x')  # Ascending by x
        >>> df.order('-x')  # Descending by x
        >>> df.order('group', '-value')  # By group asc, value desc
        """
        ascending = []
        sort_cols = []

        for col in cols:
            if isinstance(col, str) and col.startswith('-'):
                sort_cols.append(col[1:])
                ascending.append(False)
            else:
                sort_cols.append(col)
                ascending.append(not decreasing)

        sorted_df = self._df.sort_values(by=sort_cols, ascending=ascending)
        return RFrame(sorted_df)

    def unique(self, *cols) -> 'RFrame':
        """
        Get unique rows (like data.table's unique()).

        Parameters
        ----------
        *cols : str, optional
            Columns to consider for uniqueness

        Examples
        --------
        >>> df.unique()  # All columns
        >>> df.unique('x', 'y')  # Only consider x and y
        """
        if cols:
            return RFrame(self._df.drop_duplicates(subset=cols))
        return RFrame(self._df.drop_duplicates())

    def merge(self, other: 'RFrame', on: Optional[Union[str, List[str]]] = None,
              how: str = 'inner', suffixes: Tuple[str, str] = ('_x', '_y')) -> 'RFrame':
        """
        Merge with another RFrame (like R's merge()).

        Parameters
        ----------
        other : RFrame
            Frame to merge with
        on : str or list of str, optional
            Columns to merge on
        how : str, default 'inner'
            Type of merge: 'inner', 'left', 'right', 'outer'
        suffixes : tuple, default ('_x', '_y')
            Suffixes for overlapping columns

        Examples
        --------
        >>> df1.merge(df2, on='id')
        >>> df1.merge(df2, on=['id', 'date'], how='left')
        """
        other_df = other._df if isinstance(other, RFrame) else other
        result = self._df.merge(other_df, on=on, how=how, suffixes=suffixes)
        return RFrame(result)

    def rbind(self, *others) -> 'RFrame':
        """
        Row bind (like R's rbind()).

        Examples
        --------
        >>> df1.rbind(df2, df3)
        """
        dfs = [self._df]
        for other in others:
            if isinstance(other, RFrame):
                dfs.append(other._df)
            else:
                dfs.append(pd.DataFrame(other))
        return RFrame(pd.concat(dfs, ignore_index=True))

    def cbind(self, *others) -> 'RFrame':
        """
        Column bind (like R's cbind()).

        Examples
        --------
        >>> df1.cbind(df2, df3)
        """
        dfs = [self._df]
        for other in others:
            if isinstance(other, RFrame):
                dfs.append(other._df)
            elif isinstance(other, pd.Series):
                dfs.append(other.to_frame())
            elif isinstance(other, dict):
                dfs.append(pd.DataFrame(other))
            else:
                dfs.append(pd.DataFrame(other))
        return RFrame(pd.concat(dfs, axis=1))

    def melt(self, id_vars=None, measure_vars=None,
             var_name: str = 'variable', value_name: str = 'value') -> 'RFrame':
        """
        Unpivot from wide to long format (like R's melt/pivot_longer).

        Parameters
        ----------
        id_vars : str or list, optional
            Column(s) to use as identifier variables (kept as-is)
        measure_vars : str or list, optional
            Column(s) to unpivot. If None, uses all columns not in id_vars
        var_name : str, default 'variable'
            Name for the variable column
        value_name : str, default 'value'
            Name for the value column

        Returns
        -------
        RFrame
            Long-format data

        Examples
        --------
        >>> df = data_frame(id=[1, 2], A=[10, 20], B=[30, 40])
        >>> df.melt(id_vars='id')
        #    id variable  value
        # 0   1        A     10
        # 1   2        A     20
        # 2   1        B     30
        # 3   2        B     40

        >>> df.melt(id_vars='id', var_name='metric', value_name='score')
        """
        result = pd.melt(
            self._df,
            id_vars=id_vars,
            value_vars=measure_vars,
            var_name=var_name,
            value_name=value_name
        )
        return RFrame(result)

    def dcast(self, index, columns, values=None,
              aggfunc='mean', fill_value=None) -> 'RFrame':
        """
        Pivot from long to wide format (like R's dcast/pivot_wider).

        Parameters
        ----------
        index : str or list
            Column(s) to use as row identifiers
        columns : str
            Column whose unique values become new column names
        values : str, optional
            Column to use for values. If None, uses remaining columns
        aggfunc : str or callable, default 'mean'
            Aggregation function if there are duplicate index/column pairs
            Options: 'mean', 'sum', 'min', 'max', 'first', 'last', 'count'
        fill_value : scalar, optional
            Value to use for missing combinations

        Returns
        -------
        RFrame
            Wide-format data

        Examples
        --------
        >>> df = data_frame(
        ...     id=[1, 1, 2, 2],
        ...     variable=['A', 'B', 'A', 'B'],
        ...     value=[10, 20, 30, 40]
        ... )
        >>> df.dcast(index='id', columns='variable', values='value')
        #    id   A   B
        # 0   1  10  20
        # 1   2  30  40

        >>> # With aggregation
        >>> df.dcast(index='id', columns='var', values='val', aggfunc='sum')
        """
        result = pd.pivot_table(
            self._df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=fill_value
        ).reset_index()

        # Flatten column names if MultiIndex
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [
                '_'.join(str(c) for c in col).strip('_')
                if isinstance(col, tuple) else col
                for col in result.columns
            ]
        else:
            # Remove the columns name
            result.columns.name = None

        return RFrame(result)

    def pivot_longer(self, cols=None, names_to: str = 'name',
                     values_to: str = 'value', cols_exclude=None) -> 'RFrame':
        """
        Pivot from wide to long (tidyr-style alias for melt).

        Parameters
        ----------
        cols : str or list, optional
            Columns to pivot (if None, all non-id columns)
        names_to : str, default 'name'
            Name of the new column for variable names
        values_to : str, default 'value'
            Name of the new column for values
        cols_exclude : str or list, optional
            Columns to exclude from pivoting (used as id_vars)

        Examples
        --------
        >>> df.pivot_longer(cols=['A', 'B'], names_to='metric', values_to='score')
        >>> df.pivot_longer(cols_exclude='id')  # Pivot all except 'id'
        """
        if cols_exclude is not None:
            if isinstance(cols_exclude, str):
                cols_exclude = [cols_exclude]
            id_vars = cols_exclude
            measure_vars = cols
        else:
            id_vars = None
            measure_vars = cols

        return self.melt(
            id_vars=id_vars,
            measure_vars=measure_vars,
            var_name=names_to,
            value_name=values_to
        )

    def pivot_wider(self, names_from: str, values_from: str,
                    id_cols=None, aggfunc='first', fill_value=None) -> 'RFrame':
        """
        Pivot from long to wide (tidyr-style alias for dcast).

        Parameters
        ----------
        names_from : str
            Column whose values become new column names
        values_from : str
            Column whose values fill the new columns
        id_cols : str or list, optional
            Columns that identify each row. If None, uses all other columns
        aggfunc : str or callable, default 'first'
            Aggregation function for duplicate entries
        fill_value : scalar, optional
            Value for missing combinations

        Examples
        --------
        >>> df.pivot_wider(names_from='variable', values_from='value', id_cols='id')
        """
        if id_cols is None:
            # Use all columns except names_from and values_from
            id_cols = [c for c in self._df.columns if c not in [names_from, values_from]]
            if len(id_cols) == 0:
                raise ValueError("No id columns found. Specify id_cols explicitly.")

        return self.dcast(
            index=id_cols,
            columns=names_from,
            values=values_from,
            aggfunc=aggfunc,
            fill_value=fill_value
        )

    def head(self, n: int = 6) -> 'RFrame':
        """Return first n rows (R default is 6)."""
        return RFrame(self._df.head(n))

    def tail(self, n: int = 6) -> 'RFrame':
        """Return last n rows (R default is 6)."""
        return RFrame(self._df.tail(n))

    def copy(self) -> 'RFrame':
        """Create a deep copy."""
        return RFrame(self._df.copy())

    def summary(self) -> pd.DataFrame:
        """
        Summary statistics (like R's summary()).

        Returns a transposed describe() for R-like output.
        """
        return self._df.describe().T

    def str(self) -> str:
        """
        Structure of the data (like R's str()).

        Returns string description of types and sample values.
        """
        lines = [f"RFrame: {self.nrow} obs. of {self.ncol} variables:"]
        for col in self._df.columns:
            dtype = self._df[col].dtype
            sample = self._df[col].head(5).tolist()
            sample_str = ', '.join(str(x) for x in sample)
            if len(sample_str) > 50:
                sample_str = sample_str[:47] + '...'
            lines.append(f" $ {col}: {dtype} - {sample_str}")
        return '\n'.join(lines)

    # =========================================================================
    # R-style Data Manipulation
    # =========================================================================

    def subset(self, condition=None, select=None, drop: bool = False) -> 'RFrame':
        """
        Subset data (like R's subset()).

        Parameters
        ----------
        condition : callable or array-like, optional
            Row filter condition
        select : str or list, optional
            Columns to select
        drop : bool, default False
            If True and select is single column, return Series

        Examples
        --------
        >>> df.subset(lambda d: d.x > 0, select=['x', 'y'])
        """
        result = self._df.copy()

        if condition is not None:
            if callable(condition):
                mask = condition(self)
            else:
                mask = condition
            result = result[mask]

        if select is not None:
            if isinstance(select, str):
                if drop:
                    return result[select]
                select = [select]
            result = result[select]

        return RFrame(result)

    def transform(self, **kwargs) -> 'RFrame':
        """
        Transform columns (like R's transform()).

        Parameters
        ----------
        **kwargs
            column_name=expression pairs

        Examples
        --------
        >>> df.transform(z=lambda d: d.x + d.y, w=lambda d: d.x * 2)
        """
        result = self._df.copy()
        for col_name, expr in kwargs.items():
            if callable(expr):
                result[col_name] = expr(self)
            else:
                result[col_name] = expr
        return RFrame(result)

    def within(self, **kwargs) -> 'RFrame':
        """
        Modify data within frame (like R's within()).

        Similar to transform but expressions can reference newly created columns.

        Examples
        --------
        >>> df.within(z=lambda d: d.x + d.y, w=lambda d: d.z * 2)  # w uses z
        """
        result = RFrame(self._df.copy())
        for col_name, expr in kwargs.items():
            if callable(expr):
                result._df[col_name] = expr(result)
            else:
                result._df[col_name] = expr
        return result

    def apply(self, func: Callable, axis: int = 0, **kwargs) -> Union['RFrame', pd.Series]:
        """
        Apply function along axis (like R's apply()).

        Parameters
        ----------
        func : callable
            Function to apply
        axis : int, default 0
            0 for columns, 1 for rows (R uses 1 for rows, 2 for columns)

        Examples
        --------
        >>> df.apply(np.mean, axis=0)  # Column means
        >>> df.apply(sum, axis=1)  # Row sums
        """
        result = self._df.apply(func, axis=axis, **kwargs)
        if isinstance(result, pd.DataFrame):
            return RFrame(result)
        return result

    def lapply(self, func: Callable) -> 'RFrame':
        """
        Apply function to each column (like R's lapply()).

        Parameters
        ----------
        func : callable
            Function to apply to each column

        Examples
        --------
        >>> df.lapply(lambda x: x.mean())
        """
        result = {col: func(self._df[col]) for col in self._df.columns}
        return RFrame(pd.DataFrame([result]))

    def sapply(self, func: Callable) -> pd.Series:
        """
        Apply function and simplify (like R's sapply()).

        Parameters
        ----------
        func : callable
            Function to apply

        Examples
        --------
        >>> df.sapply(lambda x: x.mean())
        """
        return pd.Series({col: func(self._df[col]) for col in self._df.columns})

    def mapply(self, func: Callable, *cols) -> pd.Series:
        """
        Apply function to multiple columns element-wise.

        Parameters
        ----------
        func : callable
            Function to apply
        *cols : str
            Column names

        Examples
        --------
        >>> df.mapply(lambda x, y: x + y, 'a', 'b')
        """
        arrays = [self._df[col] for col in cols]
        return pd.Series([func(*args) for args in zip(*arrays)])

    # =========================================================================
    # Aggregation
    # =========================================================================

    def aggregate(self, formula: str = None, by=None, func=None, **kwargs) -> 'RFrame':
        """
        Aggregate data (like R's aggregate()).

        Parameters
        ----------
        formula : str, optional
            R-style formula like "y ~ x" (y aggregated by x)
        by : str or list, optional
            Grouping columns
        func : callable, optional
            Aggregation function
        **kwargs
            column=func pairs

        Examples
        --------
        >>> df.aggregate(by='group', x='mean', y='sum')
        >>> df.aggregate('value ~ group', func=np.mean)
        """
        if formula:
            # Parse R formula "y ~ x"
            parts = formula.replace(' ', '').split('~')
            if len(parts) == 2:
                value_col = parts[0]
                group_col = parts[1]
                if func is None:
                    func = np.mean
                result = self._df.groupby(group_col)[value_col].agg(func).reset_index()
                return RFrame(result)

        if by is not None:
            if isinstance(by, str):
                by = [by]
            grouped = self._df.groupby(by)

            if kwargs:
                agg_dict = {}
                for col, fn in kwargs.items():
                    if isinstance(fn, str):
                        agg_dict[col] = fn
                    else:
                        agg_dict[col] = fn
                result = grouped.agg(agg_dict).reset_index()
            elif func:
                result = grouped.agg(func).reset_index()
            else:
                result = grouped.size().reset_index(name='n')

            return RFrame(result)

        return self

    # =========================================================================
    # I/O
    # =========================================================================

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return self._df.copy()

    def to_csv(self, path: str, **kwargs):
        """Write to CSV file."""
        self._df.to_csv(path, index=False, **kwargs)

    def to_dict(self, orient: str = 'list') -> dict:
        """Convert to dictionary."""
        return self._df.to_dict(orient=orient)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> 'RFrame':
        """Create from pandas DataFrame."""
        return cls(df)

    @classmethod
    def read_csv(cls, path: str, **kwargs) -> 'RFrame':
        """Read from CSV file."""
        return cls(pd.read_csv(path, **kwargs))

    # =========================================================================
    # Iteration
    # =========================================================================

    def __iter__(self):
        """Iterate over column names (R-like behavior)."""
        return iter(self._df.columns)

    def iterrows(self):
        """Iterate over rows as (index, RFrame) pairs."""
        for idx, row in self._df.iterrows():
            yield idx, RFrame(row.to_frame().T)

    def items(self):
        """Iterate over (column_name, Series) pairs."""
        return self._df.items()

    # =========================================================================
    # Comparison with pandas
    # =========================================================================

    def __eq__(self, other):
        """Element-wise equality."""
        if isinstance(other, RFrame):
            return self._df == other._df
        return self._df == other

    def equals(self, other: 'RFrame') -> bool:
        """Check if two RFrames are equal."""
        if isinstance(other, RFrame):
            return self._df.equals(other._df)
        return False


class ByClause:
    """
    Represents a 'by' grouping clause for data.table-style operations.
    """

    def __init__(self, cols):
        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = cols[0]
        self.cols = list(cols) if not isinstance(cols, str) else [cols]

    def __repr__(self):
        return f"by({', '.join(repr(c) for c in self.cols)})"


def by(*cols) -> ByClause:
    """
    Create a by clause for grouping.

    Examples
    --------
    >>> df[:, {'n': '.N'}, by('group')]
    """
    return ByClause(cols)


# Convenience function for creating RFrame
def data_frame(**kwargs) -> RFrame:
    """
    Create an RFrame from keyword arguments (like R's data.frame()).

    Examples
    --------
    >>> df = data_frame(x=[1, 2, 3], y=['a', 'b', 'c'])
    """
    return RFrame(kwargs)


def data_table(**kwargs) -> RFrame:
    """
    Create an RFrame from keyword arguments (like R's data.table()).

    Alias for data_frame() - the RFrame class supports both styles.

    Examples
    --------
    >>> dt = data_table(x=[1, 2, 3], y=['a', 'b', 'c'])
    """
    return RFrame(kwargs)
