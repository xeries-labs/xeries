"""Hierarchy definition for multi-series data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class HierarchyDefinition:
    """Defines a hierarchical structure for multi-series time series data.

    Supports three strategies for defining hierarchy:
    1. Column-based: Map hierarchy levels to DataFrame columns
    2. Parse pattern: Regex extraction from series identifiers
    3. Explicit mapping: Direct series_id to hierarchy values dictionary

    The hierarchy is defined top-to-bottom (e.g., ['region', 'store', 'product']
    where 'region' is the highest/coarsest level).

    Example (column-based):
        >>> hierarchy = HierarchyDefinition(
        ...     levels=["state", "store", "product"],
        ...     columns=["state_id", "store_id", "item_id"]
        ... )

    Example (parse pattern):
        >>> hierarchy = HierarchyDefinition(
        ...     levels=["state", "store", "product"],
        ...     parse_pattern=r"(?P<state>\\w+)_(?P<store>\\w+)_(?P<product>\\w+)"
        ... )

    Example (explicit mapping):
        >>> hierarchy = HierarchyDefinition(
        ...     levels=["region", "channel"],
        ...     explicit_mapping={
        ...         "series_A": {"region": "North", "channel": "Online"},
        ...         "series_B": {"region": "North", "channel": "Retail"},
        ...     }
        ... )

    Attributes:
        levels: Ordered list of hierarchy level names (top to bottom).
        columns: Column names in DataFrame matching each level (column-based strategy).
        parse_pattern: Regex pattern with named groups to parse series_id.
        explicit_mapping: Direct mapping of series_id to hierarchy values.
        series_col: Column name containing series identifiers (for parsing).
    """

    levels: list[str]
    columns: list[str] | None = None
    parse_pattern: str | None = None
    explicit_mapping: dict[str, dict[str, Any]] | None = None
    series_col: str = "series_id"

    _compiled_pattern: re.Pattern[str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and compile regex if provided."""
        strategies = [
            self.columns is not None,
            self.parse_pattern is not None,
            self.explicit_mapping is not None,
        ]
        if sum(strategies) == 0:
            raise ValueError("Must provide one of: columns, parse_pattern, or explicit_mapping")
        if sum(strategies) > 1:
            raise ValueError(
                "Only one of columns, parse_pattern, or explicit_mapping can be provided"
            )

        if self.columns is not None and len(self.columns) != len(self.levels):
            raise ValueError(
                f"Number of columns ({len(self.columns)}) must match "
                f"number of levels ({len(self.levels)})"
            )

        if self.parse_pattern is not None:
            self._compiled_pattern = re.compile(self.parse_pattern)
            groups = self._compiled_pattern.groupindex.keys()
            missing = set(self.levels) - set(groups)
            if missing:
                raise ValueError(
                    f"Parse pattern must have named groups for all levels. Missing: {missing}"
                )

        if self.explicit_mapping is not None:
            for series_id, mapping in self.explicit_mapping.items():
                missing = set(self.levels) - set(mapping.keys())
                if missing:
                    raise ValueError(
                        f"Explicit mapping for '{series_id}' is missing levels: {missing}"
                    )

    def get_cohorts(self, data: pd.DataFrame, level: str) -> dict[str, pd.Index]:
        """Get cohort assignments for a given hierarchy level.

        Groups rows by the specified level and all levels above it in the hierarchy.

        Args:
            data: Input DataFrame with hierarchy information.
            level: The hierarchy level to group by.

        Returns:
            Dictionary mapping cohort name to row indices belonging to that cohort.

        Raises:
            KeyError: If level is not in the hierarchy definition.
        """
        if level not in self.levels:
            raise KeyError(f"Level '{level}' not in hierarchy. Available: {self.levels}")

        level_idx = self.levels.index(level)
        grouping_levels = self.levels[: level_idx + 1]

        if self.columns is not None:
            return self._get_cohorts_column_based(data, grouping_levels)
        elif self.parse_pattern is not None:
            return self._get_cohorts_parse_based(data, grouping_levels)
        else:
            return self._get_cohorts_explicit(data, grouping_levels)

    def _get_cohorts_column_based(
        self, data: pd.DataFrame, grouping_levels: list[str]
    ) -> dict[str, pd.Index]:
        """Get cohorts using column-based strategy."""
        level_to_col = dict(zip(self.levels, self.columns, strict=True))  # type: ignore[arg-type]
        group_cols = [level_to_col[lv] for lv in grouping_levels]

        cohorts: dict[str, pd.Index] = {}
        for name, group in data.groupby(group_cols, observed=True):
            cohort_name = "_".join(str(v) for v in name) if isinstance(name, tuple) else str(name)
            cohorts[cohort_name] = group.index

        return cohorts

    def _get_cohorts_parse_based(
        self, data: pd.DataFrame, grouping_levels: list[str]
    ) -> dict[str, pd.Index]:
        """Get cohorts by parsing series identifiers."""
        series_ids = self._get_series_ids(data)
        parsed = series_ids.apply(self._parse_series_id)

        parsed_df = pd.DataFrame(parsed.tolist(), index=data.index)

        cohorts: dict[str, pd.Index] = {}
        for name, group in parsed_df.groupby(grouping_levels, observed=True):
            cohort_name = "_".join(str(v) for v in name) if isinstance(name, tuple) else str(name)
            cohorts[cohort_name] = group.index

        return cohorts

    def _get_cohorts_explicit(
        self, data: pd.DataFrame, grouping_levels: list[str]
    ) -> dict[str, pd.Index]:
        """Get cohorts using explicit mapping."""
        series_ids = self._get_series_ids(data)

        level_values = []
        for sid in series_ids:
            if sid not in self.explicit_mapping:  # type: ignore[operator]
                raise KeyError(f"Series '{sid}' not found in explicit_mapping")
            mapping = self.explicit_mapping[sid]  # type: ignore[index]
            level_values.append({lv: mapping[lv] for lv in grouping_levels})

        parsed_df = pd.DataFrame(level_values, index=data.index)

        cohorts: dict[str, pd.Index] = {}
        for name, group in parsed_df.groupby(grouping_levels, observed=True):
            cohort_name = "_".join(str(v) for v in name) if isinstance(name, tuple) else str(name)
            cohorts[cohort_name] = group.index

        return cohorts

    def get_level_value(self, series_id: str, level: str) -> str:
        """Extract hierarchy level value from a series identifier.

        Args:
            series_id: The series identifier string.
            level: The hierarchy level to extract.

        Returns:
            The value at the specified hierarchy level.

        Raises:
            KeyError: If level is not in the hierarchy.
            ValueError: If series_id cannot be parsed (parse strategy).
        """
        if level not in self.levels:
            raise KeyError(f"Level '{level}' not in hierarchy. Available: {self.levels}")

        if self.parse_pattern is not None:
            parsed = self._parse_series_id(series_id)
            return str(parsed[level])
        elif self.explicit_mapping is not None:
            if series_id not in self.explicit_mapping:
                raise KeyError(f"Series '{series_id}' not in explicit_mapping")
            return str(self.explicit_mapping[series_id][level])
        else:
            raise ValueError("get_level_value requires parse_pattern or explicit_mapping strategy")

    def get_ancestors(self, series_id: str) -> dict[str, str]:
        """Get all ancestor values for a series (full hierarchy path).

        Args:
            series_id: The series identifier string.

        Returns:
            Dictionary mapping level names to values for this series.
        """
        if self.parse_pattern is not None:
            return {lv: str(v) for lv, v in self._parse_series_id(series_id).items()}
        elif self.explicit_mapping is not None:
            if series_id not in self.explicit_mapping:
                raise KeyError(f"Series '{series_id}' not in explicit_mapping")
            return {lv: str(self.explicit_mapping[series_id][lv]) for lv in self.levels}
        else:
            raise ValueError("get_ancestors requires parse_pattern or explicit_mapping strategy")

    def _parse_series_id(self, series_id: str) -> dict[str, Any]:
        """Parse a series_id using the compiled regex pattern."""
        if self._compiled_pattern is None:
            raise ValueError("No parse pattern configured")

        match = self._compiled_pattern.match(str(series_id))
        if match is None:
            raise ValueError(
                f"Series ID '{series_id}' does not match pattern '{self.parse_pattern}'"
            )
        return match.groupdict()

    def _get_series_ids(self, data: pd.DataFrame) -> pd.Series:
        """Extract series identifiers from DataFrame."""
        if isinstance(data.index, pd.MultiIndex) and self.series_col in data.index.names:
            return data.index.get_level_values(self.series_col).to_series(index=data.index)

        if self.series_col in data.columns:
            return data[self.series_col]

        if self.series_col == "series_id" and "_level_skforecast" in data.columns:
            return data["_level_skforecast"].astype(str)

        raise KeyError(f"Series column '{self.series_col}' not found in DataFrame columns or index")

    def get_all_levels_for_data(self, data: pd.DataFrame) -> dict[str, dict[str, pd.Index]]:
        """Get cohorts for all hierarchy levels.

        Args:
            data: Input DataFrame.

        Returns:
            Nested dict: level -> cohort_name -> row indices.
        """
        result = {"global": {"all": data.index}}

        for level in self.levels:
            result[level] = self.get_cohorts(data, level)

        return result

    def add_hierarchy_columns(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Add hierarchy level columns to the DataFrame.

        Useful for data preparation when using parse or explicit mapping strategies.

        Args:
            data: Input DataFrame.
            inplace: If True, modify DataFrame in place.

        Returns:
            DataFrame with added hierarchy columns.
        """
        if not inplace:
            data = data.copy()

        if self.columns is not None:
            return data

        if self.parse_pattern is not None:
            series_ids = self._get_series_ids(data)
            parsed = series_ids.apply(self._parse_series_id)
            parsed_df = pd.DataFrame(parsed.tolist(), index=data.index)

            for level in self.levels:
                data[level] = parsed_df[level]

        elif self.explicit_mapping is not None:
            series_ids = self._get_series_ids(data)

            for level in self.levels:
                data[level] = series_ids.map(
                    lambda sid, lv=level: self.explicit_mapping[sid][lv]  # type: ignore[index]
                )

        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Check if the DataFrame has required columns/structure for this hierarchy.

        Args:
            data: DataFrame to validate.

        Returns:
            True if valid, raises ValueError otherwise.
        """
        if self.columns is not None:
            missing = [c for c in self.columns if c not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        elif self.parse_pattern is not None or self.explicit_mapping is not None:
            try:
                self._get_series_ids(data)
            except KeyError as e:
                raise ValueError(str(e)) from e

        return True

    def get_unique_values_at_level(self, data: pd.DataFrame, level: str) -> list[Any]:
        """Get unique values at a specific hierarchy level.

        Args:
            data: Input DataFrame.
            level: Hierarchy level name.

        Returns:
            List of unique values at that level.
        """
        if level not in self.levels:
            raise KeyError(f"Level '{level}' not in hierarchy.")

        if self.columns is not None:
            level_idx = self.levels.index(level)
            col = self.columns[level_idx]
            return list(data[col].unique())

        cohorts = self.get_cohorts(data, level)
        if level == self.levels[0]:
            return list(cohorts.keys())

        values = set()
        for cohort_name in cohorts:
            parts = cohort_name.split("_")
            level_idx = self.levels.index(level)
            if level_idx < len(parts):
                values.add(parts[level_idx])
        return list(values)

    def __repr__(self) -> str:
        """Return string representation."""
        strategy = "columns" if self.columns else "parse" if self.parse_pattern else "explicit"
        return f"HierarchyDefinition(levels={self.levels}, strategy='{strategy}')"
