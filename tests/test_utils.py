"""
Collection of tests for utils.
"""
import os

import numpy as np
import pytest

from quvac.utils import (
    find_classes_in_package,
    format_memory,
    format_time,
    load_wisdom,
    read_yaml,
    round_to_n,
    save_wisdom,
    write_yaml,
)

TEST_YAML = {
    "param_1": 1,
    "param_10": 10,
}


@pytest.fixture(scope="session")
def get_tmp_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("data")
    return tmp_path


def test_yaml(tmp_path):
    yaml_path = tmp_path / "test.yml"
    write_yaml(yaml_path, TEST_YAML)
    yaml_from_file = read_yaml(yaml_path)
    assert yaml_from_file == TEST_YAML


def test_wisdom(tmp_path):
    ini_path = tmp_path / "ini.yml"
    save_wisdom(ini_path)

    wisdom_path = os.path.dirname(ini_path)
    wisdom_file = os.path.join(wisdom_path, "fftw-wisdom")
    wisdom = load_wisdom(wisdom_file)
    print(wisdom)
    assert len(wisdom) > 0, "Wisdom file should be not empty."


@pytest.mark.parametrize("t, expected", [
    (48, "   48.00 s"),
    (61, "  1 min 1.00 s"),
    (3661, " 1 h 1 min 1.00 s"),
    ])
def test_format_time(t, expected):
    t_str = format_time(t)
    assert t_str == expected


@pytest.mark.parametrize("mem, expected", [
    (128, "128.00 KB"),
    (1025, "1.00 MB"),
    (1048577, "1.00 GB"),
])
def test_format_memory(mem, expected):
    mem_str = format_memory(mem)
    assert mem_str == expected


def test_find_classes():
    cls_names = find_classes_in_package("quvac")
    assert len(cls_names) > 0, "Quvac package has classes."


@pytest.mark.parametrize("num, number_of_digits, expected", [
    (12345, 1, 12000),
    (12345, 3, 12340),
    (0.000345, 1, 0.00034),
])
def test_round(num, number_of_digits, expected):
    num_rounded = round_to_n(num, number_of_digits)
    assert np.isclose(num_rounded, expected)

