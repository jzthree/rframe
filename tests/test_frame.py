"""Tests for RFrame class."""

import numpy as np
import pandas as pd
import pytest
from rframe import (
    RFrame, data_frame, data_table, by,
    col, assign, delete, N, SD
)


class TestRFrameCreation:
    """Tests for RFrame creation."""

    def test_from_dict(self):
        df = RFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        assert df.nrow == 3
        assert df.ncol == 2

    def test_data_frame_function(self):
        df = data_frame(x=[1, 2, 3], y=['a', 'b', 'c'])
        assert df.nrow == 3
        assert df.colnames == ['x', 'y']

    def test_data_table_function(self):
        dt = data_table(id=[1, 2], value=[10, 20])
        assert dt.nrow == 2

    def test_from_pandas(self):
        pdf = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        df = RFrame(pdf)
        assert df.nrow == 2


class TestProperties:
    """Tests for R-style properties."""

    def test_nrow_ncol(self):
        df = data_frame(x=[1, 2, 3, 4], y=[5, 6, 7, 8])
        assert df.nrow == 4
        assert df.ncol == 2

    def test_colnames(self):
        df = data_frame(a=[1], b=[2], c=[3])
        assert df.colnames == ['a', 'b', 'c']

    def test_names(self):
        df = data_frame(x=[1], y=[2])
        assert df.names == ['x', 'y']

    def test_shape(self):
        df = data_frame(x=[1, 2, 3], y=[4, 5, 6])
        assert df.shape == (3, 2)


class TestColumnAccess:
    """Tests for $ style column access."""

    def test_get_column(self):
        df = data_frame(x=[1, 2, 3], y=[4, 5, 6])
        result = df.x
        np.testing.assert_array_equal(result.values, [1, 2, 3])

    def test_set_column(self):
        df = data_frame(x=[1, 2, 3])
        df.y = [4, 5, 6]
        assert 'y' in df.colnames
        np.testing.assert_array_equal(df.y.values, [4, 5, 6])

    def test_delete_column(self):
        df = data_frame(x=[1, 2], y=[3, 4])
        del df.y
        assert df.colnames == ['x']


class TestBracketNotation:
    """Tests for [i, j] bracket notation."""

    def test_single_row(self):
        df = data_frame(x=[1, 2, 3], y=[4, 5, 6])
        result = df[0, :]
        assert result.nrow == 1
        assert result.x.iloc[0] == 1

    def test_row_slice(self):
        df = data_frame(x=[1, 2, 3, 4, 5])
        result = df[0:3, :]
        assert result.nrow == 3

    def test_single_column_by_name(self):
        df = data_frame(x=[1, 2, 3], y=[4, 5, 6])
        result = df[:, 'x']
        np.testing.assert_array_equal(result.values, [1, 2, 3])

    def test_multiple_columns(self):
        df = data_frame(x=[1, 2], y=[3, 4], z=[5, 6])
        result = df[:, ['x', 'z']]
        assert result.colnames == ['x', 'z']

    def test_row_and_column(self):
        df = data_frame(x=[1, 2, 3], y=[4, 5, 6])
        result = df[0:2, ['x']]
        assert result.nrow == 2
        assert result.ncol == 1


class TestFiltering:
    """Tests for row filtering."""

    def test_filter_by_condition(self):
        df = data_frame(x=[1, 2, 3, 4, 5])
        result = df[df.x > 2, :]
        assert result.nrow == 3

    def test_filter_by_equality(self):
        df = data_frame(x=[1, 2, 3], y=['a', 'b', 'a'])
        result = df[df.y == 'a', :]
        assert result.nrow == 2

    def test_filter_combined_conditions(self):
        df = data_frame(x=[1, 2, 3, 4], y=['a', 'b', 'a', 'b'])
        result = df[(df.x > 1) & (df.y == 'b'), :]
        assert result.nrow == 2


class TestGroupedOperations:
    """Tests for data.table-style grouped operations."""

    def test_count_per_group(self):
        df = data_frame(x=[1, 2, 3, 4], group=['a', 'a', 'b', 'b'])
        result = df[:, N, by('group')]
        assert result.nrow == 2
        assert 'N' in result.colnames

    def test_grouped_aggregation(self):
        df = data_frame(x=[1, 2, 3, 4], group=['a', 'a', 'b', 'b'])
        result = df[:, {'sum_x': 'sum(x)'}, by('group')]
        assert result.nrow == 2

    def test_n_function_no_args(self):
        """Test n() function with no arguments returns count."""
        df = data_frame(group=['A', 'A', 'B', 'B', 'B'], value=[10, 20, 30, 40, 50])
        result = df[df.value > 15, {'n': 'n()'}, by('group')]
        assert result.nrow == 2
        # A has 1 value > 15 (20), B has 3 values > 15 (30, 40, 50)
        assert result[result.group == 'A', :].n.iloc[0] == 1
        assert result[result.group == 'B', :].n.iloc[0] == 3

    def test_count_function_no_args(self):
        """Test count() function with no arguments returns count."""
        df = data_frame(group=['A', 'A', 'B'], value=[1, 2, 3])
        result = df[:, {'cnt': 'count()'}, by('group')]
        assert result[result.group == 'A', :].cnt.iloc[0] == 2
        assert result[result.group == 'B', :].cnt.iloc[0] == 1


class TestColumnExpressions:
    """Tests for column expressions."""

    def test_dict_expression(self):
        df = data_frame(x=[1, 2, 3])
        result = df[:, {'x_squared': 'x**2'}]
        np.testing.assert_array_equal(result.x_squared.values, [1, 4, 9])

    def test_col_ref_expression(self):
        df = data_frame(x=[1, 2, 3], y=[10, 20, 30])
        result = df[:, {'z': col('x') + col('y')}]
        np.testing.assert_array_equal(result.z.values, [11, 22, 33])


class TestAssignment:
    """Tests for := assignment."""

    def test_assign_new_column(self):
        df = data_frame(x=[1, 2, 3])
        df[:, assign(y=col('x') * 2)]
        np.testing.assert_array_equal(df.y.values, [2, 4, 6])

    def test_delete_column(self):
        df = data_frame(x=[1, 2], y=[3, 4])
        df[:, delete('y')]
        assert 'y' not in df.colnames


class TestMethods:
    """Tests for RFrame methods."""

    def test_head(self):
        df = data_frame(x=list(range(10)))
        result = df.head(3)
        assert result.nrow == 3

    def test_tail(self):
        df = data_frame(x=list(range(10)))
        result = df.tail(3)
        assert result.nrow == 3

    def test_order(self):
        df = data_frame(x=[3, 1, 2])
        result = df.order('x')
        np.testing.assert_array_equal(result.x.values, [1, 2, 3])

    def test_order_descending(self):
        df = data_frame(x=[3, 1, 2])
        result = df.order('-x')
        np.testing.assert_array_equal(result.x.values, [3, 2, 1])

    def test_unique(self):
        df = data_frame(x=[1, 2, 1, 3, 2])
        result = df.unique('x')
        assert result.nrow == 3

    def test_copy(self):
        df = data_frame(x=[1, 2, 3])
        df2 = df.copy()
        df2.x = [4, 5, 6]
        assert df.x.iloc[0] == 1  # Original unchanged


class TestMerge:
    """Tests for merge operations."""

    def test_inner_merge(self):
        df1 = data_frame(id=[1, 2, 3], x=[10, 20, 30])
        df2 = data_frame(id=[2, 3, 4], y=[200, 300, 400])
        result = df1.merge(df2, on='id')
        assert result.nrow == 2

    def test_left_merge(self):
        df1 = data_frame(id=[1, 2, 3], x=[10, 20, 30])
        df2 = data_frame(id=[2, 3, 4], y=[200, 300, 400])
        result = df1.merge(df2, on='id', how='left')
        assert result.nrow == 3


class TestRbindCbind:
    """Tests for rbind and cbind."""

    def test_rbind(self):
        df1 = data_frame(x=[1, 2], y=[3, 4])
        df2 = data_frame(x=[5, 6], y=[7, 8])
        result = df1.rbind(df2)
        assert result.nrow == 4

    def test_cbind(self):
        df1 = data_frame(x=[1, 2])
        df2 = data_frame(y=[3, 4])
        result = df1.cbind(df2)
        assert result.ncol == 2
        assert result.colnames == ['x', 'y']


class TestTransformWithin:
    """Tests for transform and within."""

    def test_transform(self):
        df = data_frame(x=[1, 2, 3])
        result = df.transform(y=lambda d: d.x * 2)
        assert 'y' in result.colnames
        np.testing.assert_array_equal(result.y.values, [2, 4, 6])

    def test_within(self):
        df = data_frame(x=[1, 2, 3])
        result = df.within(y=lambda d: d.x * 2, z=lambda d: d.y + 10)
        assert 'z' in result.colnames
        np.testing.assert_array_equal(result.z.values, [12, 14, 16])


class TestChaining:
    """Tests for method chaining."""

    def test_chain(self):
        df = data_frame(x=[5, 2, 8, 1, 9])
        result = df.chain(
            lambda d: d[d.x > 2, :],
            lambda d: d.order('-x'),
            lambda d: d.head(2)
        )
        assert result.nrow == 2
        assert result.x.iloc[0] == 9


class TestIO:
    """Tests for I/O operations."""

    def test_to_pandas(self):
        df = data_frame(x=[1, 2, 3])
        pdf = df.to_pandas()
        assert isinstance(pdf, pd.DataFrame)

    def test_to_dict(self):
        df = data_frame(x=[1, 2], y=[3, 4])
        d = df.to_dict()
        assert d == {'x': [1, 2], 'y': [3, 4]}


class TestStr:
    """Tests for str() method."""

    def test_str_output(self):
        df = data_frame(x=[1, 2, 3], y=['a', 'b', 'c'])
        result = df.str()
        assert 'RFrame' in result
        assert '3 obs' in result
        assert '2 variables' in result
        assert '$ x' in result
        assert '$ y' in result


class TestSummary:
    """Tests for summary() method."""

    def test_summary(self):
        df = data_frame(x=[1, 2, 3, 4, 5], y=[10, 20, 30, 40, 50])
        result = df.summary()
        assert 'x' in result.index
        assert 'mean' in result.columns
