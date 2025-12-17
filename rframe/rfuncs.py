"""
R utility functions implemented in Python.

Provides R-like functions such as match, which, seq, rep, etc.
"""

import numpy as np
from typing import Any, Sequence, Optional, Union, List

# Type alias for array-like inputs
ArrayLike = Union[Sequence, np.ndarray, 'pd.Series']


# Sentinel value for no match - max int64, will always cause IndexError if used
NOMATCH = np.iinfo(np.int64).max


def match(x: ArrayLike, table: ArrayLike, nomatch: int = None) -> np.ndarray:
    """
    R's match function: returns a vector of positions of first matches.

    Parameters
    ----------
    x : array-like
        Values to be matched
    table : array-like
        Values to be matched against
    nomatch : int, default NOMATCH (max int64)
        Value to return for non-matches. Default is max int64 which will
        raise IndexError if used as an index, making no-match cases explicit.
        Use None for np.nan.

    Returns
    -------
    np.ndarray
        Integer array of positions (0-indexed, unlike R's 1-indexed)

    Examples
    --------
    >>> match([1, 2, 3], [3, 2, 1])
    array([2, 1, 0])
    >>> match(['a', 'b', 'z'], ['a', 'b', 'c'])
    array([0, 1, 9223372036854775807])
    """
    x = np.asarray(x)
    table = np.asarray(table)

    # Use NOMATCH constant if nomatch not specified
    if nomatch is None:
        nomatch = NOMATCH

    # Create lookup dictionary for O(n) performance
    lookup = {v: i for i, v in enumerate(table)}

    result = np.array([lookup.get(v, nomatch) for v in x], dtype=np.int64)

    return result


def which(condition: ArrayLike) -> np.ndarray:
    """
    R's which function: returns indices where condition is True.

    Parameters
    ----------
    condition : array-like of bool
        Boolean array

    Returns
    -------
    np.ndarray
        Integer array of indices (0-indexed)

    Examples
    --------
    >>> which([True, False, True, False])
    array([0, 2])
    >>> which(np.array([1, 2, 3, 4]) > 2)
    array([2, 3])
    """
    return np.where(np.asarray(condition))[0]


def which_max(x: ArrayLike) -> int:
    """
    R's which.max: returns index of first maximum value.

    Examples
    --------
    >>> which_max([1, 3, 2, 3])
    1
    """
    return int(np.argmax(x))


def which_min(x: ArrayLike) -> int:
    """
    R's which.min: returns index of first minimum value.

    Examples
    --------
    >>> which_min([3, 1, 2, 1])
    1
    """
    return int(np.argmin(x))


def isin(x: ArrayLike, table: ArrayLike) -> np.ndarray:
    """
    R's %in% operator: test if elements of x are in table.

    Parameters
    ----------
    x : array-like
        Values to test
    table : array-like
        Values to test against

    Returns
    -------
    np.ndarray of bool

    Examples
    --------
    >>> isin([1, 2, 5], [1, 2, 3, 4])
    array([ True,  True, False])
    """
    return np.isin(x, table)


def seq(from_: float, to: float, by: float = 1.0, length_out: Optional[int] = None) -> np.ndarray:
    """
    R's seq function: generate regular sequences.

    Parameters
    ----------
    from_ : float
        Starting value
    to : float
        End value
    by : float, default 1.0
        Increment
    length_out : int, optional
        Desired length of sequence (overrides 'by')

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> seq(1, 5)
    array([1., 2., 3., 4., 5.])
    >>> seq(1, 5, by=0.5)
    array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])
    >>> seq(1, 10, length_out=5)
    array([ 1.  ,  3.25,  5.5 ,  7.75, 10.  ])
    """
    if length_out is not None:
        return np.linspace(from_, to, length_out)
    return np.arange(from_, to + by/2, by)  # +by/2 to include endpoint


def seq_len(length_out: int) -> np.ndarray:
    """
    R's seq_len: generate sequence 0 to length_out-1.

    Examples
    --------
    >>> seq_len(5)
    array([0, 1, 2, 3, 4])
    """
    return np.arange(length_out)


def seq_along(x: ArrayLike) -> np.ndarray:
    """
    R's seq_along: generate sequence along the length of x.

    Examples
    --------
    >>> seq_along(['a', 'b', 'c'])
    array([0, 1, 2])
    """
    return np.arange(len(x))


def rep(x: Any, times: int = 1, each: int = 1, length_out: Optional[int] = None) -> np.ndarray:
    """
    R's rep function: replicate elements of a vector.

    Parameters
    ----------
    x : any
        Value or array to replicate
    times : int, default 1
        Number of times to repeat the whole vector
    each : int, default 1
        Number of times to repeat each element
    length_out : int, optional
        Desired length (truncates or recycles)

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> rep([1, 2, 3], times=2)
    array([1, 2, 3, 1, 2, 3])
    >>> rep([1, 2, 3], each=2)
    array([1, 1, 2, 2, 3, 3])
    >>> rep(1, times=5)
    array([1, 1, 1, 1, 1])
    """
    x = np.atleast_1d(x)
    result = np.tile(np.repeat(x, each), times)

    if length_out is not None:
        if length_out <= len(result):
            result = result[:length_out]
        else:
            # Recycle
            result = np.resize(result, length_out)

    return result


def rev(x: ArrayLike) -> np.ndarray:
    """
    R's rev: reverse a vector.

    Examples
    --------
    >>> rev([1, 2, 3])
    array([3, 2, 1])
    """
    return np.asarray(x)[::-1]


def unique(x: ArrayLike, return_index: bool = False) -> np.ndarray:
    """
    R's unique: return unique values preserving order of first occurrence.

    Parameters
    ----------
    x : array-like
        Input array
    return_index : bool, default False
        If True, also return indices of first occurrences

    Returns
    -------
    np.ndarray or tuple

    Examples
    --------
    >>> unique([3, 1, 2, 1, 3])
    array([3, 1, 2])
    """
    x = np.asarray(x)
    _, idx = np.unique(x, return_index=True)
    idx = np.sort(idx)  # Preserve order of first occurrence

    if return_index:
        return x[idx], idx
    return x[idx]


def duplicated(x: ArrayLike, from_last: bool = False) -> np.ndarray:
    """
    R's duplicated: determine duplicate elements.

    Parameters
    ----------
    x : array-like
        Input array
    from_last : bool, default False
        If True, consider duplicates from the end

    Returns
    -------
    np.ndarray of bool
        True for duplicates

    Examples
    --------
    >>> duplicated([1, 2, 1, 3, 2])
    array([False, False,  True, False,  True])
    """
    x = np.asarray(x)
    seen = {}
    result = np.zeros(len(x), dtype=bool)

    indices = range(len(x) - 1, -1, -1) if from_last else range(len(x))

    for i in indices:
        key = x[i] if np.isscalar(x[i]) else tuple(x[i]) if hasattr(x[i], '__iter__') else x[i]
        try:
            hash(key)
        except TypeError:
            key = str(x[i])

        if key in seen:
            result[i] = True
        else:
            seen[key] = True

    return result


def head(x: ArrayLike, n: int = 6) -> np.ndarray:
    """
    R's head: return first n elements.

    Examples
    --------
    >>> head([1, 2, 3, 4, 5, 6, 7, 8], 3)
    array([1, 2, 3])
    """
    return np.asarray(x)[:n]


def tail(x: ArrayLike, n: int = 6) -> np.ndarray:
    """
    R's tail: return last n elements.

    Examples
    --------
    >>> tail([1, 2, 3, 4, 5, 6, 7, 8], 3)
    array([6, 7, 8])
    """
    return np.asarray(x)[-n:]


def length(x: ArrayLike) -> int:
    """
    R's length function.

    Examples
    --------
    >>> length([1, 2, 3])
    3
    """
    return len(x)


def paste(*args, sep: str = ' ', collapse: Optional[str] = None) -> Union[np.ndarray, str]:
    """
    R's paste function: concatenate strings.

    Parameters
    ----------
    *args : array-like
        Vectors to concatenate
    sep : str, default ' '
        Separator between elements
    collapse : str, optional
        If provided, collapse result into single string

    Returns
    -------
    np.ndarray or str

    Examples
    --------
    >>> paste(['a', 'b'], [1, 2], sep='-')
    array(['a-1', 'b-2'], dtype='<U3')
    >>> paste(['a', 'b', 'c'], collapse=',')
    'a,b,c'
    """
    arrays = [np.atleast_1d(a).astype(str) for a in args]
    max_len = max(len(a) for a in arrays)

    # Recycle arrays to same length
    arrays = [np.resize(a, max_len) for a in arrays]

    # Concatenate with separator
    result = arrays[0]
    for arr in arrays[1:]:
        result = np.char.add(np.char.add(result, sep), arr)

    if collapse is not None:
        return collapse.join(result)
    return result


def paste0(*args, collapse: Optional[str] = None) -> Union[np.ndarray, str]:
    """
    R's paste0: paste with no separator.

    Examples
    --------
    >>> paste0(['a', 'b'], [1, 2])
    array(['a1', 'b2'], dtype='<U2')
    """
    return paste(*args, sep='', collapse=collapse)


def ifelse(condition: ArrayLike, yes: Any, no: Any) -> np.ndarray:
    """
    R's ifelse: vectorized conditional.

    Parameters
    ----------
    condition : array-like of bool
        Condition to test
    yes : any
        Value(s) where condition is True
    no : any
        Value(s) where condition is False

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> ifelse([True, False, True], 'yes', 'no')
    array(['yes', 'no', 'yes'], dtype='<U3')
    >>> ifelse(np.array([1,2,3]) > 1, 'big', 'small')
    array(['small', 'big', 'big'], dtype='<U5')
    """
    condition = np.asarray(condition)
    yes = np.atleast_1d(yes)
    no = np.atleast_1d(no)

    # Broadcast to condition length
    if len(yes) == 1:
        yes = np.repeat(yes, len(condition))
    if len(no) == 1:
        no = np.repeat(no, len(condition))

    return np.where(condition, yes, no)


def is_na(x: ArrayLike) -> np.ndarray:
    """
    R's is.na: test for NA/NaN values.

    Examples
    --------
    >>> is_na([1, np.nan, 3, None])
    array([False,  True, False,  True])
    """
    import pandas as pd
    return pd.isna(x)


def na_omit(x: ArrayLike) -> np.ndarray:
    """
    R's na.omit: remove NA values.

    Examples
    --------
    >>> na_omit([1, np.nan, 3, None, 5])
    array([1., 3., 5.])
    """
    import pandas as pd
    x = np.asarray(x)
    mask = ~pd.isna(x)
    return x[mask]


def complete_cases(*args) -> np.ndarray:
    """
    R's complete.cases: find rows with no NA values.

    Parameters
    ----------
    *args : array-like
        Vectors to check

    Returns
    -------
    np.ndarray of bool

    Examples
    --------
    >>> complete_cases([1, np.nan, 3], [4, 5, 6])
    array([ True, False,  True])
    """
    import pandas as pd
    arrays = [np.atleast_1d(a) for a in args]
    result = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        result &= ~pd.isna(arr)
    return result


def order(*args, decreasing: bool = False, na_last: bool = True) -> np.ndarray:
    """
    R's order: return indices that would sort the input.

    Parameters
    ----------
    *args : array-like
        Vectors to sort by (in order of priority)
    decreasing : bool, default False
        Sort in decreasing order
    na_last : bool, default True
        Put NA values last

    Returns
    -------
    np.ndarray of int
        Indices that would sort the input

    Examples
    --------
    >>> order([3, 1, 2])
    array([1, 2, 0])
    >>> order([3, 1, 2], decreasing=True)
    array([0, 2, 1])
    """
    import pandas as pd

    if len(args) == 1:
        x = np.asarray(args[0])
        # Handle NA values
        na_pos = 'last' if na_last else 'first'
        s = pd.Series(x)
        return s.sort_values(ascending=not decreasing, na_position=na_pos).index.values
    else:
        # Multiple sort keys
        df = pd.DataFrame({f'col{i}': np.asarray(a) for i, a in enumerate(args)})
        ascending = [not decreasing] * len(args)
        na_pos = 'last' if na_last else 'first'
        return df.sort_values(by=list(df.columns), ascending=ascending, na_position=na_pos).index.values


def rank(x: ArrayLike, ties_method: str = 'average', na_last: bool = True) -> np.ndarray:
    """
    R's rank function: compute ranks of values.

    Parameters
    ----------
    x : array-like
        Values to rank
    ties_method : str, default 'average'
        How to handle ties: 'average', 'first', 'min', 'max', 'dense'
    na_last : bool, default True
        Put NA values last (assign highest rank)

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> rank([3, 1, 2])
    array([3., 1., 2.])
    >>> rank([3, 1, 2, 1])
    array([4. , 1.5, 3. , 1.5])
    """
    import pandas as pd
    s = pd.Series(x)
    return s.rank(method=ties_method, na_option='keep' if not na_last else 'bottom').values


def cut(x: ArrayLike, breaks: Union[int, ArrayLike], labels: Optional[ArrayLike] = None,
        right: bool = True, include_lowest: bool = False) -> np.ndarray:
    """
    R's cut: divide into intervals.

    Parameters
    ----------
    x : array-like
        Values to bin
    breaks : int or array-like
        Number of bins or break points
    labels : array-like, optional
        Labels for bins
    right : bool, default True
        Intervals closed on right
    include_lowest : bool, default False
        Include lowest value in first bin

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> cut([1, 5, 10, 15, 20], breaks=[0, 5, 10, 20])
    array(['(0, 5]', '(0, 5]', '(5, 10]', '(10, 20]', '(10, 20]'], dtype=object)
    """
    import pandas as pd
    return pd.cut(x, breaks, labels=labels, right=right, include_lowest=include_lowest).values


def table(*args) -> 'pd.Series':
    """
    R's table: build contingency table (frequency counts).

    Parameters
    ----------
    *args : array-like
        Vectors to tabulate

    Returns
    -------
    pd.Series or pd.DataFrame

    Examples
    --------
    >>> table(['a', 'b', 'a', 'c', 'b', 'a'])
    a    3
    b    2
    c    1
    dtype: int64
    """
    import pandas as pd

    if len(args) == 1:
        return pd.Series(args[0]).value_counts().sort_index()
    else:
        df = pd.DataFrame({f'V{i}': a for i, a in enumerate(args)})
        return df.groupby(list(df.columns)).size()


def setdiff(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    R's setdiff: set difference (elements in x but not in y).

    Examples
    --------
    >>> setdiff([1, 2, 3, 4], [2, 4])
    array([1, 3])
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return x[~np.isin(x, y)]


def intersect(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    R's intersect: set intersection.

    Examples
    --------
    >>> intersect([1, 2, 3], [2, 3, 4])
    array([2, 3])
    """
    return np.intersect1d(x, y)


def union(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    R's union: set union.

    Examples
    --------
    >>> union([1, 2, 3], [2, 3, 4])
    array([1, 2, 3, 4])
    """
    return np.union1d(x, y)


# NA constant (similar to R's NA)
NA = np.nan


# Convenience function for creating named vectors (like R's c())
def c(*args, **kwargs) -> Union[np.ndarray, dict]:
    """
    R's c() function: combine values into a vector.

    If keyword arguments are provided, returns a dict (named vector).

    Examples
    --------
    >>> c(1, 2, 3)
    array([1, 2, 3])
    >>> c(a=1, b=2, c=3)
    {'a': 1, 'b': 2, 'c': 3}
    """
    if kwargs and not args:
        return kwargs
    if args and not kwargs:
        # Flatten nested arrays
        result = []
        for a in args:
            if hasattr(a, '__iter__') and not isinstance(a, (str, dict)):
                result.extend(a)
            else:
                result.append(a)
        return np.array(result)
    # Both args and kwargs - not typical R behavior, just combine
    result = list(args)
    result.extend(kwargs.values())
    return np.array(result)
