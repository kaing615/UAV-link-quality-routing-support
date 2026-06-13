"""Test data quality validation functions."""

import torch

from src.validation.data_quality import (
    check_class_balance,
    check_tensor_quality,
)


def test_nan_detection():
    tensor = torch.tensor([[1.0, float("nan")]])
    issues = check_tensor_quality(tensor, "test", {})
    assert any(i["check"] == "nan_detection" for i in issues)


def test_inf_detection():
    tensor = torch.tensor([[1.0, float("inf")]])
    issues = check_tensor_quality(tensor, "test", {})
    assert any(i["check"] == "inf_detection" for i in issues)


def test_balanced_class():
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    issues = check_class_balance(labels, "test_run")
    assert len(issues) == 0


def test_imbalanced_class():
    labels = torch.ones(100)
    issues = check_class_balance(labels, "test_run")
    assert any(i["check"] == "class_imbalance" for i in issues)
