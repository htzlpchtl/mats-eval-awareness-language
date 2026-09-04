import pytest

from src.audit_length import distribution


def test_distribution_reports_fixed_population_statistics():
    result = distribution([1, 2, 3, 4])
    assert result["n"] == 4
    assert result["mean"] == 2.5
    assert result["median"] == 2.5
    assert result["min"] == 1
    assert result["max"] == 4
    assert result["std_population"] == pytest.approx(1.11803398875)


def test_distribution_rejects_empty_input():
    with pytest.raises(ValueError, match="empty distribution"):
        distribution([])
