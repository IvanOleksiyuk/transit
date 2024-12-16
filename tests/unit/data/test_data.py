# Standard block for correct import
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

from pathlib import Path

current_dir = Path.cwd()

import unittest
import hydra
import numpy as np
from transit.src.data.data import InMemoryDataFrameDict

class TestProcessors(unittest.TestCase):
    def test_ProcessorApplyCuts(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorApplyCuts.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([0.7504512, 0.94056472])
        )

    def test_ProcessorSplitDataFrameVars(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorSplitDataFrameVars.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(dataset[0]["data"], np.array([0.30471708]))
        np.testing.assert_allclose(dataset[0]["aux"], np.array([-1.03998411]))

    def test_ProcessorNormalise(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorNormalise.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([0.14314841, -0.86843804])
        )

    def test_ProcessorIntervals(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorIntervals.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([-1.95103519, -1.30217951])
        )

    def test_ProcessorShuffle(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorShuffle.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([-1.95103519, -1.30217951])
        )

    def test_ProcessorLoadInsertDatasets(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorLoadInsertDatasets.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        assert "df" in dataset[0]

    def test_ProcessorLHCOcurtains(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorLHCOcurtains.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        assert "tau32_j2" in dataset.data["df"]

    def test_ProcessorSignalContamination(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorSignalContamination_0cont.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        assert sum(dataset.data["df"]["is_signal"]) == 0

    def test_ProcessorRemoveFrames(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorRemoveFrames.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        assert len(dataset.data) == 0

    def test_ProcessorCATHODE(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorCATHODE.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"],
            np.array([
                0.22069168,
                1.1354702,
                -0.33870256,
                -0.22138572,
                -0.9122287,
                -1.4461726,
            ]),
        )

    def test_ProcessorMergeFrames(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorMergeFrames.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        assert "data" in dataset.data
        assert "aux" not in dataset.data


class TestDatasets(unittest.TestCase):
    """Class for testing InMemoryDataFrameDictBase subclasses."""

    def test_InMemoryDataFrameDict(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_InMemoryDataFrameDict.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([0.30471708, -1.03998411])
        )
        assert len(dataset) == 10
        assert dataset.get_dim()["data"] == (2,)

    def test_InMemoryDataMerge(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_InMemoryDataMerge.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([0.30471708, -1.03998411])
        )
        np.testing.assert_allclose(
            dataset[10]["data"], np.array([0.30471708, -1.03998411])
        )

    def test_InMemoryDataMergeClasses(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_InMemoryDataMergeClasses.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([0.30471708, -1.03998411])
        )
        np.testing.assert_allclose(
            dataset[10]["data"], np.array([0.30471708, -1.03998411])
        )
        np.testing.assert_allclose(dataset[10]["label"], np.array([1]))


class TestDatamodule(unittest.TestCase):
    def test_SimpleDataModule(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_SimpleDataModule.yaml")
        datamodule = hydra.utils.instantiate(cfg.data)
        assert datamodule.get_dim()["data"] == (2,)


if __name__ == "__main__":
    testclass = TestProcessors()
    testclass.test_ProcessorApplyCuts()
    testclass.test_ProcessorSplitDataFrameVars()
    testclass.test_ProcessorNormalise()
    testclass.test_ProcessorIntervals()
    testclass.test_ProcessorShuffle()
    testclass.test_ProcessorLoadInsertDatasets()
    testclass.test_ProcessorLHCOcurtains()
    testclass.test_ProcessorSignalContamination()
    testclass.test_ProcessorRemoveFrames()
    testclass.test_ProcessorCATHODE()
    testclass.test_ProcessorMergeFrames()
    testclass = TestDatasets()
    testclass.test_InMemoryDataFrameDict()
    testclass.test_InMemoryDataMerge()
    testclass.test_InMemoryDataMergeClasses()
    testclass = TestDatamodule()
    testclass.test_SimpleDataModule()
    print("Success! All tests passed!")
