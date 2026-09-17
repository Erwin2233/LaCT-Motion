"""T2M 数据集模块"""

# 延迟导入以避免循环依赖
__all__ = ["Text2MotionDatasetRL"]

def __getattr__(name):
    if name == "Text2MotionDatasetRL":
        from .dataset_TM_eval_RL import Text2MotionDatasetRL
        return Text2MotionDatasetRL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
