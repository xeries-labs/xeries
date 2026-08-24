"""Unit tests for HierarchyDefinition."""

from __future__ import annotations

import pandas as pd
import pytest

from xeries.hierarchy.definition import HierarchyDefinition


class TestHierarchyDefinitionInit:
    """Tests for HierarchyDefinition initialization."""

    def test_column_based_init(self) -> None:
        """Test initialization with column-based strategy."""
        hierarchy = HierarchyDefinition(
            levels=["state", "store", "product"],
            columns=["state_id", "store_id", "item_id"],
        )

        assert hierarchy.levels == ["state", "store", "product"]
        assert hierarchy.columns == ["state_id", "store_id", "item_id"]
        assert hierarchy.parse_pattern is None
        assert hierarchy.explicit_mapping is None

    def test_parse_pattern_init(self) -> None:
        """Test initialization with parse pattern strategy."""
        hierarchy = HierarchyDefinition(
            levels=["state", "store"],
            parse_pattern=r"(?P<state>\w+)_(?P<store>\w+)",
        )

        assert hierarchy.levels == ["state", "store"]
        assert hierarchy.parse_pattern is not None
        assert hierarchy._compiled_pattern is not None

    def test_explicit_mapping_init(self) -> None:
        """Test initialization with explicit mapping strategy."""
        hierarchy = HierarchyDefinition(
            levels=["region", "channel"],
            explicit_mapping={
                "series_A": {"region": "North", "channel": "Online"},
                "series_B": {"region": "South", "channel": "Retail"},
            },
        )

        assert hierarchy.levels == ["region", "channel"]
        assert hierarchy.explicit_mapping is not None

    def test_no_strategy_raises(self) -> None:
        """Test that missing strategy raises ValueError."""
        with pytest.raises(ValueError, match="Must provide one of"):
            HierarchyDefinition(levels=["state", "store"])

    def test_multiple_strategies_raises(self) -> None:
        """Test that multiple strategies raise ValueError."""
        with pytest.raises(ValueError, match="Only one of"):
            HierarchyDefinition(
                levels=["state", "store"],
                columns=["state_id", "store_id"],
                parse_pattern=r"(?P<state>\w+)_(?P<store>\w+)",
            )

    def test_column_count_mismatch_raises(self) -> None:
        """Test that column/level count mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Number of columns"):
            HierarchyDefinition(
                levels=["state", "store", "product"],
                columns=["state_id", "store_id"],
            )

    def test_parse_pattern_missing_groups_raises(self) -> None:
        """Test that missing named groups in pattern raises ValueError."""
        with pytest.raises(ValueError, match="must have named groups"):
            HierarchyDefinition(
                levels=["state", "store"],
                parse_pattern=r"(?P<state>\w+)_(\w+)",
            )

    def test_explicit_mapping_missing_levels_raises(self) -> None:
        """Test that incomplete explicit mapping raises ValueError."""
        with pytest.raises(ValueError, match="is missing levels"):
            HierarchyDefinition(
                levels=["region", "channel"],
                explicit_mapping={
                    "series_A": {"region": "North"},
                },
            )


class TestHierarchyDefinitionColumnBased:
    """Tests for column-based hierarchy operations."""

    @pytest.fixture
    def hierarchical_data(self) -> pd.DataFrame:
        """Create sample hierarchical data."""
        data = {
            "state_id": ["TX", "TX", "TX", "WI", "WI", "WI"],
            "store_id": ["S1", "S1", "S2", "S1", "S2", "S2"],
            "item_id": ["P1", "P2", "P1", "P1", "P1", "P2"],
            "sales": [100, 150, 200, 120, 180, 90],
            "price": [10, 15, 10, 12, 10, 15],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def column_hierarchy(self) -> HierarchyDefinition:
        """Create column-based hierarchy."""
        return HierarchyDefinition(
            levels=["state", "store", "product"],
            columns=["state_id", "store_id", "item_id"],
        )

    def test_get_cohorts_state_level(
        self, hierarchical_data: pd.DataFrame, column_hierarchy: HierarchyDefinition
    ) -> None:
        """Test getting cohorts at state level."""
        cohorts = column_hierarchy.get_cohorts(hierarchical_data, "state")

        assert "TX" in cohorts
        assert "WI" in cohorts
        assert len(cohorts["TX"]) == 3
        assert len(cohorts["WI"]) == 3

    def test_get_cohorts_store_level(
        self, hierarchical_data: pd.DataFrame, column_hierarchy: HierarchyDefinition
    ) -> None:
        """Test getting cohorts at store level (includes state)."""
        cohorts = column_hierarchy.get_cohorts(hierarchical_data, "store")

        assert "TX_S1" in cohorts
        assert "TX_S2" in cohorts
        assert "WI_S1" in cohorts
        assert "WI_S2" in cohorts
        assert len(cohorts["TX_S1"]) == 2

    def test_get_cohorts_product_level(
        self, hierarchical_data: pd.DataFrame, column_hierarchy: HierarchyDefinition
    ) -> None:
        """Test getting cohorts at product level (includes state and store)."""
        cohorts = column_hierarchy.get_cohorts(hierarchical_data, "product")

        assert "TX_S1_P1" in cohorts
        assert "TX_S1_P2" in cohorts
        assert len(cohorts["TX_S1_P1"]) == 1

    def test_get_cohorts_invalid_level_raises(
        self, hierarchical_data: pd.DataFrame, column_hierarchy: HierarchyDefinition
    ) -> None:
        """Test that invalid level raises KeyError."""
        with pytest.raises(KeyError, match="not in hierarchy"):
            column_hierarchy.get_cohorts(hierarchical_data, "invalid_level")

    def test_validate_data_success(
        self, hierarchical_data: pd.DataFrame, column_hierarchy: HierarchyDefinition
    ) -> None:
        """Test successful data validation."""
        assert column_hierarchy.validate_data(hierarchical_data) is True

    def test_validate_data_missing_column(self, column_hierarchy: HierarchyDefinition) -> None:
        """Test data validation with missing column."""
        data = pd.DataFrame({"state_id": ["TX"], "other": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            column_hierarchy.validate_data(data)

    def test_get_all_levels_for_data(
        self, hierarchical_data: pd.DataFrame, column_hierarchy: HierarchyDefinition
    ) -> None:
        """Test getting cohorts for all levels."""
        all_levels = column_hierarchy.get_all_levels_for_data(hierarchical_data)

        assert "global" in all_levels
        assert "state" in all_levels
        assert "store" in all_levels
        assert "product" in all_levels
        assert len(all_levels["global"]["all"]) == 6


class TestHierarchyDefinitionParseBased:
    """Tests for parse pattern-based hierarchy operations."""

    @pytest.fixture
    def parse_hierarchy(self) -> HierarchyDefinition:
        """Create parse pattern hierarchy."""
        return HierarchyDefinition(
            levels=["state", "store"],
            parse_pattern=r"(?P<state>\w+)_(?P<store>\w+)",
            series_col="series_id",
        )

    @pytest.fixture
    def series_data(self) -> pd.DataFrame:
        """Create data with series_id column."""
        return pd.DataFrame(
            {
                "series_id": ["TX_S1", "TX_S1", "TX_S2", "WI_S1", "WI_S2"],
                "value": [1, 2, 3, 4, 5],
            }
        )

    def test_get_level_value(self, parse_hierarchy: HierarchyDefinition) -> None:
        """Test extracting level value from series_id."""
        assert parse_hierarchy.get_level_value("TX_S1", "state") == "TX"
        assert parse_hierarchy.get_level_value("TX_S1", "store") == "S1"

    def test_get_ancestors(self, parse_hierarchy: HierarchyDefinition) -> None:
        """Test getting full hierarchy path."""
        ancestors = parse_hierarchy.get_ancestors("WI_S2")
        assert ancestors == {"state": "WI", "store": "S2"}

    def test_get_cohorts_parse_based(
        self, series_data: pd.DataFrame, parse_hierarchy: HierarchyDefinition
    ) -> None:
        """Test getting cohorts with parse pattern."""
        cohorts = parse_hierarchy.get_cohorts(series_data, "state")

        assert "TX" in cohorts
        assert "WI" in cohorts
        assert len(cohorts["TX"]) == 3
        assert len(cohorts["WI"]) == 2

    def test_invalid_series_id_raises(self, parse_hierarchy: HierarchyDefinition) -> None:
        """Test that invalid series_id raises ValueError."""
        with pytest.raises(ValueError, match="does not match pattern"):
            parse_hierarchy.get_level_value("INVALID", "state")

    def test_add_hierarchy_columns(
        self, series_data: pd.DataFrame, parse_hierarchy: HierarchyDefinition
    ) -> None:
        """Test adding hierarchy columns to DataFrame."""
        result = parse_hierarchy.add_hierarchy_columns(series_data)

        assert "state" in result.columns
        assert "store" in result.columns
        assert result["state"].iloc[0] == "TX"
        assert result["store"].iloc[0] == "S1"


class TestHierarchyDefinitionExplicitMapping:
    """Tests for explicit mapping-based hierarchy operations."""

    @pytest.fixture
    def explicit_hierarchy(self) -> HierarchyDefinition:
        """Create explicit mapping hierarchy."""
        return HierarchyDefinition(
            levels=["region", "channel"],
            explicit_mapping={
                "A": {"region": "North", "channel": "Online"},
                "B": {"region": "North", "channel": "Retail"},
                "C": {"region": "South", "channel": "Online"},
            },
            series_col="series_id",
        )

    @pytest.fixture
    def explicit_data(self) -> pd.DataFrame:
        """Create data for explicit mapping."""
        return pd.DataFrame(
            {
                "series_id": ["A", "A", "B", "C", "C"],
                "value": [1, 2, 3, 4, 5],
            }
        )

    def test_get_level_value_explicit(self, explicit_hierarchy: HierarchyDefinition) -> None:
        """Test extracting level value from explicit mapping."""
        assert explicit_hierarchy.get_level_value("A", "region") == "North"
        assert explicit_hierarchy.get_level_value("C", "channel") == "Online"

    def test_get_cohorts_explicit(
        self, explicit_data: pd.DataFrame, explicit_hierarchy: HierarchyDefinition
    ) -> None:
        """Test getting cohorts with explicit mapping."""
        cohorts = explicit_hierarchy.get_cohorts(explicit_data, "region")

        assert "North" in cohorts
        assert "South" in cohorts
        assert len(cohorts["North"]) == 3
        assert len(cohorts["South"]) == 2

    def test_missing_series_in_mapping_raises(
        self, explicit_hierarchy: HierarchyDefinition
    ) -> None:
        """Test that missing series in mapping raises KeyError."""
        data = pd.DataFrame({"series_id": ["A", "X"], "value": [1, 2]})
        with pytest.raises(KeyError, match="not found in explicit_mapping"):
            explicit_hierarchy.get_cohorts(data, "region")


class TestHierarchyDefinitionRepr:
    """Tests for HierarchyDefinition string representation."""

    def test_repr_column_strategy(self) -> None:
        """Test repr with column strategy."""
        hierarchy = HierarchyDefinition(levels=["a", "b"], columns=["col_a", "col_b"])
        assert "columns" in repr(hierarchy)
        assert "['a', 'b']" in repr(hierarchy)

    def test_repr_parse_strategy(self) -> None:
        """Test repr with parse strategy."""
        hierarchy = HierarchyDefinition(levels=["a", "b"], parse_pattern=r"(?P<a>\w+)_(?P<b>\w+)")
        assert "parse" in repr(hierarchy)

    def test_repr_explicit_strategy(self) -> None:
        """Test repr with explicit strategy."""
        hierarchy = HierarchyDefinition(levels=["a"], explicit_mapping={"x": {"a": "1"}})
        assert "explicit" in repr(hierarchy)
