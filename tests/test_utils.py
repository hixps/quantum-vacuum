"""
Collection of tests for utils.
"""
import numpy as np
import pytest
import yaml

from quvac.utils import read_yaml, write_yaml, save_wisdom, load_wisdom


@pytest.fixture(scope="session")
def get_tmp_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("data")
    return tmp_path

TEST_YAML = {
    "param_1": 1,
    "param_10": 10,
}


def test_yaml(tmp_path):
    yaml_path = tmp_path / "test.yml"
    write_yaml(yaml_path, TEST_YAML)
    yaml_from_file = read_yaml(yaml_path)
    assert yaml_from_file == TEST_YAML 


