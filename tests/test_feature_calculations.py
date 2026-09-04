"""Tests for feature store registry."""
import os
import tempfile

from src.spark.feature_store import FeatureRegistry


def test_register_and_retrieve():
    """Register a feature and retrieve it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "registry.json")
        reg = FeatureRegistry(path)
        reg.register(
            name="test_feature",
            description="A test feature",
            data_type="double",
            version="1.0",
        )
        feat = reg.get_feature("test_feature")
        assert feat is not None
        assert feat["description"] == "A test feature"
        assert feat["version"] == "1.0"
        assert feat["data_type"] == "double"


def test_list_features():
    """List all registered features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "registry.json")
        reg = FeatureRegistry(path)
        reg.register(name="feat_a", description="A", version="1.0")
        reg.register(name="feat_b", description="B", version="2.0")
        features = reg.list_features()
        assert "feat_a" in features
        assert "feat_b" in features
        assert len(features) == 2


def test_get_latest_version():
    """Get version of a registered feature."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "registry.json")
        reg = FeatureRegistry(path)
        reg.register(name="my_feat", description="test", version="3.1")
        assert reg.get_latest_version("my_feat") == "3.1"
        assert reg.get_latest_version("nonexistent") is None


def test_get_nonexistent_feature():
    """Getting a non-existent feature returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "registry.json")
        reg = FeatureRegistry(path)
        assert reg.get_feature("nonexistent") is None
