# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import TYPE_CHECKING

from ..import_utils import _LazyModule


_import_structure = {
    "accuracy_rewards": ["accuracy_reward", "reasoning_accuracy_reward"],
    "format_rewards": ["think_format_reward"],
    "other_rewards": ["get_soft_overlong_punishment"],
    "t2m_rewards": [
        # 基础奖励函数
        "t2m_format_reward",
        "t2m_format_soft_reward",
        "t2m_motion_f1_reward",
        "t2m_motion_lcs_reward",
        # Embedding 奖励函数 (需要 VQ-VAE + Evaluator)
        "t2m_motion_embedding_reward",
        "t2m_semantic_reward",
        # 物理奖励函数
        "t2m_phys_reward",
        "t2m_phys_joint_reward",
        "t2m_phys_vel_reward",
        # 组合奖励函数
        "t2m_combined_reward",
        "t2m_flexible_reward",
        # 预定义奖励
        "t2m_basic_reward",
        "t2m_strict_reward",
        "t2m_physics_reward",
        "t2m_semantic_preset_reward",
        "t2m_full_reward",
        "t2m_lcs_reward",
        # 工厂函数
        "create_t2m_reward_func",
        # 常量
        "AVAILABLE_REWARDS",
        "REWARD_PRESETS",
        # M2T 奖励函数
        "m2t_format_reward",
        "m2t_format_soft_reward",
        "m2t_semantic_reward",
        # 联合训练奖励函数
        "task_aware_reward",
        "task_aware_preset_reward",
        # UniMo 风格奖励函数
        "unified_format_reward_func",
        "unified_similarity_reward_func",
    ],
}


if TYPE_CHECKING:
    from .accuracy_rewards import accuracy_reward, reasoning_accuracy_reward
    from .format_rewards import think_format_reward
    from .other_rewards import get_soft_overlong_punishment
    from .t2m_rewards import (
        AVAILABLE_REWARDS,
        REWARD_PRESETS,
        create_t2m_reward_func,
        t2m_basic_reward,
        t2m_combined_reward,
        t2m_flexible_reward,
        t2m_format_reward,
        t2m_format_soft_reward,
        t2m_full_reward,
        t2m_lcs_reward,
        t2m_motion_embedding_reward,
        t2m_motion_f1_reward,
        t2m_motion_lcs_reward,
        t2m_physics_reward,
        t2m_phys_joint_reward,
        t2m_phys_reward,
        t2m_phys_vel_reward,
        t2m_semantic_preset_reward,
        t2m_semantic_reward,
        t2m_strict_reward,
        # M2T 奖励函数
        m2t_format_reward,
        m2t_format_soft_reward,
        m2t_semantic_reward,
        # 联合训练奖励函数
        task_aware_reward,
        task_aware_preset_reward,
        # UniMo 风格奖励函数
        unified_format_reward_func,
        unified_similarity_reward_func,
    )


else:
    sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
