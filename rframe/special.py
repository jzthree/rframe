"""
Special symbols for data.table-like operations.

Provides .N, .SD, .I, .GRP, and related special variables.
"""

from typing import Any, List, Optional
import numpy as np


class _SpecialSymbol:
    """Base class for special symbols."""

    def __init__(self, name: str):
        self.name = name
        self._value = None

    def __repr__(self):
        return f".{self.name}"

    def _set_value(self, value):
        self._value = value

    def _get_value(self):
        return self._value


class _N(_SpecialSymbol):
    """
    .N - Number of rows in the current group (or total rows if ungrouped).

    In data.table, .N represents the count of observations.

    Examples
    --------
    >>> dt[, .N]  # Total row count
    >>> dt[, .N, by='group']  # Count per group
    """

    def __init__(self):
        super().__init__('N')


class _SD(_SpecialSymbol):
    """
    .SD - Subset of Data for each group.

    Contains all columns except grouping columns for the current group.

    Examples
    --------
    >>> dt[, .SD, by='group']  # All non-grouping columns per group
    >>> dt[, lapply(.SD, mean), by='group']  # Mean of all columns per group
    """

    def __init__(self):
        super().__init__('SD')
        self._cols = None  # For .SDcols functionality

    def __getitem__(self, cols):
        """Support .SD[cols] to select specific columns."""
        new_sd = _SD()
        new_sd._cols = cols
        return new_sd


class _I(_SpecialSymbol):
    """
    .I - Row indices (0-indexed) in the current group.

    Examples
    --------
    >>> dt[, .I]  # Row indices
    >>> dt[, .I, by='group']  # Row indices per group
    """

    def __init__(self):
        super().__init__('I')


class _GRP(_SpecialSymbol):
    """
    .GRP - Current group number (0-indexed).

    Examples
    --------
    >>> dt[, .GRP, by='group']  # Group number for each row
    """

    def __init__(self):
        super().__init__('GRP')


class _BY(_SpecialSymbol):
    """
    .BY - Named list of current grouping values.

    Examples
    --------
    >>> dt[, print(.BY), by='group']  # Current group values
    """

    def __init__(self):
        super().__init__('BY')


class _NGRP(_SpecialSymbol):
    """
    .NGRP - Total number of groups.

    Examples
    --------
    >>> dt[, .NGRP, by='group']  # Total number of groups
    """

    def __init__(self):
        super().__init__('NGRP')


# Singleton instances
N = _N()
SD = _SD()
I = _I()
GRP = _GRP()
BY = _BY()
NGRP = _NGRP()


class AssignmentExpr:
    """
    Represents a := assignment expression.

    Used internally to handle data.table's := operator for
    assignment by reference.
    """

    def __init__(self, col_name: str, value: Any):
        self.col_name = col_name
        self.value = value
        self.is_delete = value is None

    def __repr__(self):
        if self.is_delete:
            return f"{self.col_name} := NULL"
        return f"{self.col_name} := <expr>"


class ColRef:
    """
    Column reference that supports := assignment.

    Allows syntax like: dt[:, {'new_col': col('x') + 1}]
    """

    def __init__(self, name: str):
        self.name = name
        self._ops = []

    def __repr__(self):
        return f"col({self.name!r})"

    # Arithmetic operations
    def __add__(self, other):
        return BinaryOp(self, '+', other)

    def __radd__(self, other):
        return BinaryOp(other, '+', self)

    def __sub__(self, other):
        return BinaryOp(self, '-', other)

    def __rsub__(self, other):
        return BinaryOp(other, '-', self)

    def __mul__(self, other):
        return BinaryOp(self, '*', other)

    def __rmul__(self, other):
        return BinaryOp(other, '*', self)

    def __truediv__(self, other):
        return BinaryOp(self, '/', other)

    def __rtruediv__(self, other):
        return BinaryOp(other, '/', self)

    def __floordiv__(self, other):
        return BinaryOp(self, '//', other)

    def __mod__(self, other):
        return BinaryOp(self, '%', other)

    def __pow__(self, other):
        return BinaryOp(self, '**', other)

    # Comparison operations
    def __eq__(self, other):
        return BinaryOp(self, '==', other)

    def __ne__(self, other):
        return BinaryOp(self, '!=', other)

    def __lt__(self, other):
        return BinaryOp(self, '<', other)

    def __le__(self, other):
        return BinaryOp(self, '<=', other)

    def __gt__(self, other):
        return BinaryOp(self, '>', other)

    def __ge__(self, other):
        return BinaryOp(self, '>=', other)

    # Logical operations
    def __and__(self, other):
        return BinaryOp(self, '&', other)

    def __or__(self, other):
        return BinaryOp(self, '|', other)

    def __invert__(self):
        return UnaryOp('~', self)


class BinaryOp:
    """Represents a binary operation between expressions."""

    def __init__(self, left, op: str, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"

    # Allow chaining operations
    def __add__(self, other):
        return BinaryOp(self, '+', other)

    def __radd__(self, other):
        return BinaryOp(other, '+', self)

    def __sub__(self, other):
        return BinaryOp(self, '-', other)

    def __rsub__(self, other):
        return BinaryOp(other, '-', self)

    def __mul__(self, other):
        return BinaryOp(self, '*', other)

    def __rmul__(self, other):
        return BinaryOp(other, '*', self)

    def __truediv__(self, other):
        return BinaryOp(self, '/', other)

    def __rtruediv__(self, other):
        return BinaryOp(other, '/', self)

    def __eq__(self, other):
        return BinaryOp(self, '==', other)

    def __ne__(self, other):
        return BinaryOp(self, '!=', other)

    def __lt__(self, other):
        return BinaryOp(self, '<', other)

    def __le__(self, other):
        return BinaryOp(self, '<=', other)

    def __gt__(self, other):
        return BinaryOp(self, '>', other)

    def __ge__(self, other):
        return BinaryOp(self, '>=', other)

    def __and__(self, other):
        return BinaryOp(self, '&', other)

    def __or__(self, other):
        return BinaryOp(self, '|', other)


class UnaryOp:
    """Represents a unary operation."""

    def __init__(self, op: str, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"({self.op}{self.operand})"


def col(name: str) -> ColRef:
    """
    Create a column reference for use in expressions.

    Parameters
    ----------
    name : str
        Column name

    Returns
    -------
    ColRef

    Examples
    --------
    >>> dt[:, col('x') + col('y')]
    >>> dt[:, {'z': col('x') * 2}]
    """
    return ColRef(name)


def assign(**kwargs) -> List[AssignmentExpr]:
    """
    Create assignment expressions for := operations.

    Parameters
    ----------
    **kwargs
        Column names and their new values

    Returns
    -------
    List[AssignmentExpr]

    Examples
    --------
    >>> dt[:, assign(new_col=col('x') + 1)]
    >>> dt[:, assign(a=1, b=col('x')*2)]
    """
    return [AssignmentExpr(k, v) for k, v in kwargs.items()]


def delete(*cols) -> List[AssignmentExpr]:
    """
    Create deletion expressions (equivalent to := NULL in R).

    Parameters
    ----------
    *cols : str
        Column names to delete

    Returns
    -------
    List[AssignmentExpr]

    Examples
    --------
    >>> dt[:, delete('col1', 'col2')]
    """
    return [AssignmentExpr(c, None) for c in cols]


class Exclude:
    """
    Represents column exclusion (like R's -c("col1", "col2")).

    Used internally by the exclude() function.
    """

    def __init__(self, cols: List[str]):
        self.cols = cols

    def __repr__(self):
        return f"exclude({', '.join(repr(c) for c in self.cols)})"


def exclude(*cols) -> Exclude:
    """
    Exclude columns from selection (like R's -c("col1", "col2")).

    Parameters
    ----------
    *cols : str
        Column names to exclude

    Returns
    -------
    Exclude

    Examples
    --------
    >>> df[:, exclude('x', 'y')]  # All columns except x and y
    >>> df[:, exclude('temp')]    # All columns except temp
    """
    # Flatten if a list is passed
    if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
        cols = cols[0]
    return Exclude(list(cols))
