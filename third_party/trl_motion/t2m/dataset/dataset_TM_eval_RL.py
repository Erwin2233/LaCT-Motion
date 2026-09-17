"""
dataset_TM_eval_RL.py
RL 训练专用：Text2Motion 数据集（按 motion 粒度随机 caption）
- 采样方式：以 motion 为粒度，每次在 rec['text'] 中 random.choice 1 个 caption
- 保留 RL 训练所需返回字段（含 prompt）
- 去除 flat_index 静态展开，降低内存占用
"""

import os
from _paths import project_path
import random
import codecs as cs
from os.path import join as pjoin
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from third_party.trl_motion.t2m.utils import paramUtil


def collate_fn_rl(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """让上层 Trainer 的 data_collator 透传即可"""
    batch.sort(key=lambda ex: ex.get("gt_len", 0), reverse=True)
    return batch


class Text2MotionDatasetRL(data.Dataset):
    """
    面向 GRPO/RLHF 的 Text2Motion 数据集（随机 caption 版）
    """

    def __init__(
        self,
        dataset_name: str,                 # 't2m' or 'kit'
        split: str,                        # 'train'/'val'/'test'
        w_vectorizer,
        feat_bias: int = 5,
        max_text_len: int = 20,
        unit_length: int = 4,
        max_motion_len_override: Optional[int] = None,
        verbose_missing_cot: bool = False,
        data_root: Optional[str] = None,
        meta_dir: Optional[str] = None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        self.verbose_missing_cot = verbose_missing_cot

        # === 1. 路径 & 基本参数 ===
        if dataset_name == 't2m':
            self.data_root = data_root or project_path('dataset', 'HumanML3D')
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = meta_dir or project_path('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta')
            min_motion_len = 40
        elif dataset_name == 'kit':
            self.data_root = data_root or project_path('dataset', 'KIT-ML')
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = meta_dir or project_path('checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta')
            min_motion_len = 24
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        if max_motion_len_override is not None:
            self.max_motion_length = int(max_motion_len_override)

        # 归一化统计
        self.mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(self.meta_dir, 'std.npy'))

        # split 文件
        split_file = pjoin(self.data_root, f'{split}.txt')
        with cs.open(split_file, 'r') as f:
            id_list: List[str] = [line.strip() for line in f]

        # === 2. 构造 data_dict：motion → List[text] ===
        data_dict: Dict[str, Dict[str, Any]] = {}
        new_name_list: List[str] = []
        length_list: List[int] = []

        for name in tqdm(id_list, desc=f"build {dataset_name}-{split}"):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion) < min_motion_len) or (len(motion) >= 200):
                    continue

                text_data: List[Dict[str, Any]] = []

                with cs.open(pjoin(self.text_dir, name + '.txt'), 'r', encoding='utf-8') as ftxt:
                    for i, line in enumerate(ftxt):
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = 0.0 if np.isnan(float(line_split[2])) else float(line_split[2])
                        to_tag = 0.0 if np.isnan(float(line_split[3])) else float(line_split[3])

                        # 读取可选 CoT
                        cot_file = pjoin(self.data_root, "final", f"{name}_{i}.txt")
                        cot = ""
                        if os.path.exists(cot_file):
                            with cs.open(cot_file, "r", encoding="utf-8") as cf:
                                cot = cf.read().strip().replace("<thinking>", "").replace("</thinking>", "")
                        elif self.verbose_missing_cot:
                            print(f"[CoT missing] {cot_file}")

                        text_data.append({'caption': caption, 'tokens': tokens, 'cot': cot})

                        # 子片段（保留原逻辑，子片段的 text 只包含该 caption）
                        if f_tag != 0.0 or to_tag != 0.0:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion) < min_motion_len) or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text': [{'caption': caption, 'tokens': tokens, 'cot': cot}],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except Exception as e:
                                print("[slice-error]", line_split, f_tag, to_tag, name, e)

                # 整段（整段保留所有 captions，供随机抽取）
                data_dict[name] = {'motion': motion, 'length': len(motion), 'text': text_data}
                new_name_list.append(name)
                length_list.append(len(motion))

            except Exception:
                # 跳过损坏样本
                pass

        # 按长度排序（方便后续按长度重置等）
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = list(name_list)

        # 不再静态展开 flat_index（节省内存）
        self._epoch = 0

    # ------------------------------------------------------------------
    def shuffle(self):
        """外部可在 epoch 结束后调用：对 motion 粒度的 name_list 打乱"""
        random.shuffle(self.name_list)

    def reset_max_len(self, length: int):
        # 将"最大 motion 长度"真正重定向到 self.max_motion_length
        assert 1 <= length <= self.max_motion_length
        self.max_motion_length = int(length)
        print(f"[DatasetRL] Max motion length hard-cap reset to {self.max_motion_length}")

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        # 按 motion 数量计数
        return len(self.name_list)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        # ---- 以 motion 为粒度取样 ----
        name = self.name_list[item]
        rec = self.data_dict[name]
        motion = rec['motion']
        m_length = rec['length']

        # ---- 在该 motion 的所有描述里随机选 1 个 caption ----
        text_dict = random.choice(rec['text'])
        caption = text_dict['caption']
        tokens = text_dict['tokens']
        cot = text_dict.get('cot', '')

        # ---------- token 到 embedding ----------
        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots, word_embeddings = [], []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # ---------- motion 采样 ----------
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        start = random.randint(0, max(0, len(motion) - m_length))
        motion = motion[start: start + m_length]

        # ---------- 归一化 & padding ----------
        motion = (motion - self.mean) / self.std
        gt_len = int(m_length)
        if gt_len < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - gt_len, motion.shape[1]), dtype=motion.dtype)
            motion = np.concatenate([motion, pad], axis=0)

        # ---------- RL 用的 prompt ----------
        t2m_messages = [
            {"role": "system",
             "content": ("You are an assistant who helps users generate 3D human motion representations. "
                         "The users will describe a motion, your job is to break it down into a short "
                         "sequence of atomic physical actions. Show your reasoning inside <think> </think> "
                         "and output motion in <answer> </answer> tags. Response Format: "
                         "<think>...</think><answer>...</answer>")},
            {"role": "user",
             "content": f"### Instruction:\nGenerate your reasoning and motion matching the following description: {caption}"},
        ]

        return {
            "prompt": t2m_messages,
            "qa_prompt": None,
            "motion_codes": [],

            "word_embeddings": torch.tensor(word_embeddings, dtype=torch.float32),
            "pos_one_hots": torch.tensor(pos_one_hots, dtype=torch.float32),
            "sent_len": torch.tensor(sent_len, dtype=torch.long),

            "gt_pose": torch.tensor(motion, dtype=torch.float32),
            "gt_len": torch.tensor(gt_len, dtype=torch.long),

            "caption": caption,
            "name": name,
            "cot": cot,
        }

    # ------------------------------------------------------------------
    # 供外部快速构造 DataLoader（与原接口保持一致）
    @staticmethod
    def DATALoader(
        dataset_name: str,
        split: str,
        batch_size: int,
        w_vectorizer,
        num_workers: int = 8,
        unit_length: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = True,
        max_motion_len_override: Optional[int] = None,
        verbose_missing_cot: bool = False,
        data_root: Optional[str] = None,
        meta_dir: Optional[str] = None,
    ):
        ds = Text2MotionDatasetRL(
            dataset_name=dataset_name,
            split=split,
            w_vectorizer=w_vectorizer,
            unit_length=unit_length,
            max_motion_len_override=max_motion_len_override,
            verbose_missing_cot=verbose_missing_cot,
            data_root=data_root,
            meta_dir=meta_dir,
        )
        return data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_rl,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
