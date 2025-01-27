from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_dl3dv import DatasetDL3DV, DatasetDL3DVCfg
from .dataset_cuboid import DatasetCuboid, DatasetCuboidCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "dl3dv": DatasetDL3DV,
    "cuboid": DatasetCuboid,
}


DatasetCfg = DatasetRE10kCfg | DatasetDL3DVCfg | DatasetCuboidCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    """
    根据输入的配置文件cfg 阶段stage等其它参数 获取view_sampler
    再根据cfg.name从DATASETS字典中获取对应的数据集类Datasetxxx(cfg, stage, view_sampler)
    最后返回构造出的Datasetxxx对象
    """
    # debug
    # print(f"cfg.view_sampler: {cfg.view_sampler}")
    
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)
