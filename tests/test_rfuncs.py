"""Tests for R utility functions."""

import numpy as np
import pytest
from rframe import (
    match, NOMATCH, which, which_max, which_min, isin,
    seq, seq_len, seq_along, rep, rev,
    unique, duplicated, setdiff, intersect, union,
    head, tail, length, order, rank,
    paste, paste0, ifelse,
    is_na, na_omit, complete_cases, NA,
    cut, table, c
)


class TestMatch:
    """Tests for match() function."""

    def test_basic_match(self):
        result = match([1, 2, 3], [3, 2, 1])
        np.testing.assert_array_equal(result, [2, 1, 0])

    def test_match_with_missing(self):
        result = match([1, 2, 5], [1, 2, 3, 4])
        np.testing.assert_array_equal(result, [0, 1, NOMATCH])

    def test_match_strings(self):
        result = match(['a', 'b', 'z'], ['a', 'b', 'c'])
        np.testing.assert_array_equal(result, [0, 1, NOMATCH])

    def test_match_custom_nomatch(self):
        result = match([1, 5], [1, 2, 3], nomatch=999)
        np.testing.assert_array_equal(result, [0, 999])

    def test_nomatch_causes_index_error(self):
        """NOMATCH value should raise IndexError if used as index."""
        table = [10, 20, 30]
        result = match([10, 99], table)  # 99 not in table
        assert result[0] == 0  # 10 found at index 0
        assert result[1] == NOMATCH
        with pytest.raises(IndexError):
            _ = table[result[1]]  # Should raise IndexError


class TestWhich:
    """Tests for which() function."""

    def test_basic_which(self):
        result = which([True, False, True, False])
        np.testing.assert_array_equal(result, [0, 2])

    def test_which_from_comparison(self):
        result = which(np.array([1, 2, 3, 4]) > 2)
        np.testing.assert_array_equal(result, [2, 3])

    def test_which_empty(self):
        result = which([False, False, False])
        assert len(result) == 0


class TestWhichMaxMin:
    """Tests for which_max() and which_min()."""

    def test_which_max(self):
        assert which_max([1, 3, 2, 3]) == 1

    def test_which_min(self):
        assert which_min([3, 1, 2, 1]) == 1


class TestIsin:
    """Tests for isin() function (%in% operator)."""

    def test_basic_isin(self):
        result = isin([1, 2, 5], [1, 2, 3, 4])
        np.testing.assert_array_equal(result, [True, True, False])


class TestSeq:
    """Tests for seq functions."""

    def test_seq_basic(self):
        result = seq(1, 5)
        np.testing.assert_array_almost_equal(result, [1, 2, 3, 4, 5])

    def test_seq_with_by(self):
        result = seq(0, 1, by=0.5)
        np.testing.assert_array_almost_equal(result, [0, 0.5, 1.0])

    def test_seq_length_out(self):
        result = seq(1, 10, length_out=5)
        assert len(result) == 5
        assert result[0] == 1
        assert result[-1] == 10

    def test_seq_len(self):
        result = seq_len(5)
        np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])

    def test_seq_along(self):
        result = seq_along(['a', 'b', 'c'])
        np.testing.assert_array_equal(result, [0, 1, 2])


class TestRep:
    """Tests for rep() function."""

    def test_rep_times(self):
        result = rep([1, 2, 3], times=2)
        np.testing.assert_array_equal(result, [1, 2, 3, 1, 2, 3])

    def test_rep_each(self):
        result = rep([1, 2, 3], each=2)
        np.testing.assert_array_equal(result, [1, 1, 2, 2, 3, 3])

    def test_rep_scalar(self):
        result = rep(1, times=5)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 1])

    def test_rep_length_out(self):
        result = rep([1, 2], length_out=5)
        np.testing.assert_array_equal(result, [1, 2, 1, 2, 1])


class TestRev:
    """Tests for rev() function."""

    def test_rev(self):
        result = rev([1, 2, 3])
        np.testing.assert_array_equal(result, [3, 2, 1])


class TestUnique:
    """Tests for unique() function."""

    def test_unique_preserves_order(self):
        result = unique([3, 1, 2, 1, 3])
        np.testing.assert_array_equal(result, [3, 1, 2])


class TestDuplicated:
    """Tests for duplicated() function."""

    def test_duplicated(self):
        result = duplicated([1, 2, 1, 3, 2])
        np.testing.assert_array_equal(result, [False, False, True, False, True])


class TestSetOperations:
    """Tests for set operations."""

    def test_setdiff(self):
        result = setdiff([1, 2, 3, 4], [2, 4])
        np.testing.assert_array_equal(result, [1, 3])

    def test_intersect(self):
        result = intersect([1, 2, 3], [2, 3, 4])
        np.testing.assert_array_equal(result, [2, 3])

    def test_union(self):
        result = union([1, 2, 3], [2, 3, 4])
        np.testing.assert_array_equal(result, [1, 2, 3, 4])


class TestHeadTail:
    """Tests for head() and tail()."""

    def test_head(self):
        result = head([1, 2, 3, 4, 5, 6, 7, 8], 3)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_tail(self):
        result = tail([1, 2, 3, 4, 5, 6, 7, 8], 3)
        np.testing.assert_array_equal(result, [6, 7, 8])


class TestOrder:
    """Tests for order() function."""

    def test_order_ascending(self):
        result = order([3, 1, 2])
        np.testing.assert_array_equal(result, [1, 2, 0])

    def test_order_descending(self):
        result = order([3, 1, 2], decreasing=True)
        np.testing.assert_array_equal(result, [0, 2, 1])


class TestPaste:
    """Tests for paste functions."""

    def test_paste(self):
        result = paste(['a', 'b'], [1, 2], sep='-')
        np.testing.assert_array_equal(result, ['a-1', 'b-2'])

    def test_paste0(self):
        result = paste0(['a', 'b'], [1, 2])
        np.testing.assert_array_equal(result, ['a1', 'b2'])

    def test_paste_collapse(self):
        result = paste(['a', 'b', 'c'], collapse=',')
        assert result == 'a,b,c'


class TestIfelse:
    """Tests for ifelse() function."""

    def test_ifelse(self):
        result = ifelse([True, False, True], 'yes', 'no')
        np.testing.assert_array_equal(result, ['yes', 'no', 'yes'])


class TestNAHandling:
    """Tests for NA handling functions."""

    def test_is_na(self):
        result = is_na([1, np.nan, 3, None])
        np.testing.assert_array_equal(result, [False, True, False, True])

    def test_na_omit(self):
        result = na_omit([1, np.nan, 3, None, 5])
        np.testing.assert_array_equal(result, [1, 3, 5])

    def test_complete_cases(self):
        result = complete_cases([1, np.nan, 3], [4, 5, 6])
        np.testing.assert_array_equal(result, [True, False, True])


class TestC:
    """Tests for c() function."""

    def test_c_simple(self):
        result = c(1, 2, 3)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_c_flatten(self):
        result = c([1, 2], [3, 4], 5)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])

    def test_c_named(self):
        result = c(a=1, b=2, c=3)
        assert result == {'a': 1, 'b': 2, 'c': 3}


class TestTable:
    """Tests for table() function."""

    def test_table_simple(self):
        result = table(['a', 'b', 'a', 'c', 'b', 'a'])
        assert result['a'] == 3
        assert result['b'] == 2
        assert result['c'] == 1

    def test_table_numeric(self):
        result = table([1, 2, 1, 1, 2, 3])
        assert result[1] == 3
        assert result[2] == 2
        assert result[3] == 1


class TestRank:
    """Tests for rank() function."""

    def test_rank_simple(self):
        result = rank([30, 10, 20])
        np.testing.assert_array_equal(result, [3., 1., 2.])

    def test_rank_ties_average(self):
        result = rank([30, 10, 10, 20])
        np.testing.assert_array_equal(result, [4., 1.5, 1.5, 3.])

    def test_rank_ties_first(self):
        result = rank([30, 10, 10, 20], ties_method='first')
        np.testing.assert_array_equal(result, [4., 1., 2., 3.])


class TestCut:
    """Tests for cut() function."""

    def test_cut_basic(self):
        result = cut([1, 5, 10, 15, 20], breaks=[0, 5, 10, 20])
        assert str(result[0]) == '(0, 5]'
        assert str(result[2]) == '(5, 10]'
        assert str(result[4]) == '(10, 20]'

    def test_cut_with_labels(self):
        result = cut([1, 5, 15], breaks=[0, 5, 10, 20], labels=['low', 'med', 'high'])
        np.testing.assert_array_equal(result, ['low', 'low', 'high'])


class TestLength:
    """Tests for length() function."""

    def test_length(self):
        assert length([1, 2, 3]) == 3
        assert length([]) == 0
        assert length('abc') == 3
