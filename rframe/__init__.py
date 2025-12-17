"""
RFrame - An R-like DataFrame library for Python.

Provides data.frame and data.table-like syntax with pandas as backend.

Examples
--------
>>> from rframe import RFrame, data_frame, match, which, col

# Create a data frame
>>> df = data_frame(x=[1, 2, 3, 4, 5], y=['a', 'b', 'a', 'b', 'c'], z=[10, 20, 30, 40, 50])

# R-style bracket notation
>>> df[0, :]          # First row
>>> df[:, 'x']        # Column x
>>> df[df.x > 2, :]   # Filter rows

# data.table-style grouped operations
>>> df[:, {'mean_z': 'mean(z)'}, by('y')]

# R utility functions
>>> match([1, 2, 5], [5, 4, 3, 2, 1])  # Returns positions
>>> which(df.x > 2)                     # Returns indices where True

# Assignment by reference
>>> df[:, assign(new_col=col('x') * 2)]
"""

__version__ = '0.1.0'

# Main DataFrame class
from .frame import (
    RFrame,
    ByClause,
    by,
    data_frame,
    data_table,
)

# Special symbols for data.table operations
from .special import (
    N,      # .N - row count
    SD,     # .SD - subset of data
    I,      # .I - row indices
    GRP,    # .GRP - group number
    BY,     # .BY - current group values
    NGRP,   # .NGRP - total groups
    col,    # Column reference
    assign, # := assignment
    delete, # Delete columns
    ColRef,
    AssignmentExpr,
)

# R utility functions
from .rfuncs import (
    # Core matching/lookup
    match,
    which,
    which_max,
    which_min,
    isin,       # %in% operator

    # Sequence generation
    seq,
    seq_len,
    seq_along,
    rep,
    rev,

    # Set operations
    unique,
    duplicated,
    setdiff,
    intersect,
    union,

    # Array utilities
    head,
    tail,
    length,
    order,
    rank,

    # String operations
    paste,
    paste0,

    # Conditional
    ifelse,

    # NA handling
    is_na,
    na_omit,
    complete_cases,
    NA,

    # Binning
    cut,
    table,

    # Vector creation
    c,
)

# Convenient aliases
DataFrame = RFrame
DataTable = RFrame
DT = data_table
DF = data_frame

__all__ = [
    # Version
    '__version__',

    # Main classes
    'RFrame',
    'DataFrame',
    'DataTable',
    'ByClause',

    # Constructors
    'data_frame',
    'data_table',
    'by',
    'DT',
    'DF',

    # Special symbols
    'N',
    'SD',
    'I',
    'GRP',
    'BY',
    'NGRP',
    'col',
    'assign',
    'delete',
    'ColRef',
    'AssignmentExpr',

    # R functions - matching
    'match',
    'which',
    'which_max',
    'which_min',
    'isin',

    # R functions - sequences
    'seq',
    'seq_len',
    'seq_along',
    'rep',
    'rev',

    # R functions - sets
    'unique',
    'duplicated',
    'setdiff',
    'intersect',
    'union',

    # R functions - arrays
    'head',
    'tail',
    'length',
    'order',
    'rank',

    # R functions - strings
    'paste',
    'paste0',

    # R functions - conditional
    'ifelse',

    # R functions - NA
    'is_na',
    'na_omit',
    'complete_cases',
    'NA',

    # R functions - other
    'cut',
    'table',
    'c',
]
