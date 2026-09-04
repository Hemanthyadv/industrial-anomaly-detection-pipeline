"""Tests for MLflow routing and configuration helpers."""
from src.ml.mlflow_setup import ABRouter


class MockModel:
    def __init__(self, name):
        self.name = name


def test_ab_router():
    model_a = MockModel("A")
    model_b = MockModel("B")
    router = ABRouter(model_a, model_b, weight_a=1.0, weight_b=0.0)
    m, label = router.select()
    assert m.name == "A"
    assert label == "model_a"

    router_b = ABRouter(model_a, model_b, weight_a=0.0, weight_b=1.0)
    m2, label2 = router_b.select()
    assert m2.name == "B"
    assert label2 == "model_b"
