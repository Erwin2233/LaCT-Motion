# import torch
# from torch.utils import data
# import numpy as np
# from os.path import join as pjoin
# import random
# import codecs as cs
# from tqdm import tqdm
# import os
from _paths import project_path
# from . import paramUtil
# from torch.utils.data._utils.collate import default_collate


# def collate_fn(batch):
#     # 不改：仍然按 sent_len 排序，保持你原逻辑兼容
#     batch.sort(key=lambda x: x[3], reverse=True)
#     return default_collate(batch)


# class Text2MotionDataset(data.Dataset):
#     """
#     训练用 Text2Motion 数据集
#     变更：
#     - 新增 m2t 的 chain-of-thought 读取（cot_m2t），路径默认 data_root/final_m2t/{name}_{i}.txt
#     - 保留原有 t2m 的 chain-of-thought（cot 或 cot_t2m），路径默认 data_root/final/{name}_{i}.txt
#     - __getitem__ 末尾多返回一个字段：cot_m2t（不影响原有字段索引）
#     """
#     def __init__(
#         self, dataset_name, type, w_vectorizer,
#         feat_bias=5, max_text_len=20, unit_length=4,
#         cot_t2m_dir: str = "final",
#         cot_m2t_dir: str = "m2t_cot",
#     ):
#         self.max_length = 20
#         self.pointer = 0
#         self.dataset_name = dataset_name
#         self.type = type
#         self.max_text_len = max_text_len
#         self.unit_length = unit_length
#         self.w_vectorizer = w_vectorizer

#         # ---------- 数据集路径 ----------
#         if dataset_name == 't2m':
#             self.data_root = project_path('dataset', 'HumanML3D')
#             self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
#             self.text_dir = pjoin(self.data_root, 'texts')
#             self.joints_num = 22
#             fps = 20
#             self.max_motion_length = 196
#             dim_pose = 263
#             kinematic_chain = paramUtil.t2m_kinematic_chain
#             self.meta_dir = project_path('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta')
#         elif dataset_name == 'kit':
#             self.data_root = project_path('dataset', 'KIT-ML')
#             self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
#             self.text_dir = pjoin(self.data_root, 'texts')
#             self.joints_num = 21
#             fps = 12.5
#             dim_pose = 251
#             self.max_motion_length = 196
#             kinematic_chain = paramUtil.kit_kinematic_chain
#             self.meta_dir = project_path('checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta')
#         else:
#             raise ValueError(f"Unknown dataset_name: {dataset_name}")

#         # 记录 CoT 目录（相对 data_root）
#         self.cot_t2m_dir = cot_t2m_dir
#         self.cot_m2t_dir = cot_m2t_dir

#         mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
#         std = np.load(pjoin(self.meta_dir, 'std.npy'))

#         # ---------- 划分 ----------
#         if type == 'test':
#             split_file = pjoin(self.data_root, 'test.txt')
#         elif type == 'val':
#             split_file = pjoin(self.data_root, 'val.txt')
#         elif type == 'train':
#             split_file = pjoin(self.data_root, 'train.txt')
#         else:
#             raise ValueError('Invalid type')

#         min_motion_len = 40 if self.dataset_name == 't2m' else 24

#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())

#         new_name_list = []
#         length_list = []

#         # 只在第一次缺失时打印 warning，避免刷屏
#         missing_warned_t2m = set()
#         missing_warned_m2t = set()

#         def _read_cot_safe(base_dir: str, name: str, idx: int, tag: str):
#             """读取 CoT；不存在则返回空串"""
#             fp = pjoin(self.data_root, base_dir, f"{name}_{idx}.txt")
#             if os.path.exists(fp):
#                 try:
#                     with cs.open(fp, "r", encoding="utf-8") as cf:
#                         cot = cf.read().strip()
#                         # 常见标签清理
#                         cot = (cot
#                                .replace("<thinking>", "")
#                                .replace("</thinking>", "")
#                                .replace("<think>", "")
#                                .replace("</think>", ""))
#                         return cot
#                 except Exception as e:
#                     # 有文件但读失败
#                     if (name, idx) not in missing_warned_t2m and tag == "t2m":
#                         print(f"[WARN] read t2m CoT failed: {fp} ({e})")
#                         missing_warned_t2m.add((name, idx))
#                     if (name, idx) not in missing_warned_m2t and tag == "m2t":
#                         print(f"[WARN] read m2t CoT failed: {fp} ({e})")
#                         missing_warned_m2t.add((name, idx))
#                     return ""
#             else:
#                 # 文件不存在
#                 if tag == "t2m" and (name, idx) not in missing_warned_t2m:
#                     print(f"[WARN] t2m CoT missing: {fp}")
#                     missing_warned_t2m.add((name, idx))
#                 if tag == "m2t" and (name, idx) not in missing_warned_m2t:
#                     print(f"[WARN] m2t CoT missing: {fp}")
#                     missing_warned_m2t.add((name, idx))
#                 return ""

#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(self.motion_dir, name + '.npy'))
#                 if (len(motion)) < min_motion_len or (len(motion) >= 200):
#                     continue

#                 text_data = []
#                 flag = False
#                 with cs.open(pjoin(self.text_dir, name + '.txt'), encoding="utf-8") as f:
#                     for i, line in enumerate(f.readlines()):
#                         text_dict = {}
#                         line_split = line.strip().split('#')
#                         caption = line_split[0]
#                         tokens = line_split[1].split(' ')
#                         f_tag = float(line_split[2])
#                         to_tag = float(line_split[3])
#                         f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                         to_tag = 0.0 if np.isnan(to_tag) else to_tag

#                         # 读取 t2m / m2t CoT
#                         cot_t2m = _read_cot_safe(self.cot_t2m_dir, name, i, "t2m")
#                         cot_m2t = _read_cot_safe(self.cot_m2t_dir, name, i, "m2t")

#                         text_dict['caption'] = caption
#                         text_dict['tokens'] = tokens
#                         text_dict['cot'] = cot_t2m          # 兼容原字段名
#                         text_dict['cot_t2m'] = cot_t2m      # 新增明确命名
#                         text_dict['cot_m2t'] = cot_m2t      # 新增 m2t 思维链

#                         if f_tag == 0.0 and to_tag == 0.0:
#                             flag = True
#                             text_data.append(text_dict)
#                         else:
#                             try:
#                                 fps_local = fps
#                                 n_motion = motion[int(f_tag * fps_local): int(to_tag * fps_local)]
#                                 if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
#                                     continue
#                                 new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
#                                 while new_name in data_dict:
#                                     new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
#                                 data_dict[new_name] = {
#                                     'motion': n_motion,
#                                     'length': len(n_motion),
#                                     'text': [text_dict],
#                                 }
#                                 new_name_list.append(new_name)
#                                 length_list.append(len(n_motion))
#                             except Exception as e:
#                                 print(line_split)
#                                 print(line_split[2], line_split[3], f_tag, to_tag, name)
#                                 # 可选：print(e)

#                 if flag:
#                     data_dict[name] = {
#                         'motion': motion,
#                         'length': len(motion),
#                         'text': text_data
#                     }
#                     new_name_list.append(name)
#                     length_list.append(len(motion))

#             except Exception:
#                 # 静默跳过坏样本
#                 pass

#         name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
#         self.mean = mean
#         self.std = std
#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.name_list = name_list
#         self.reset_max_len(self.max_length)

#     def reset_max_len(self, length):
#         assert length <= self.max_motion_length
#         self.pointer = np.searchsorted(self.length_arr, length)
#         print("Pointer Pointing at %d" % self.pointer)
#         self.max_length = length

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def forward_transform(self, data):
#         return (data - self.mean) / self.std

#     def __len__(self):
#         return len(self.data_dict) - self.pointer

#     def __getitem__(self, item):
#         idx = self.pointer + item
#         name = self.name_list[idx]
#         data = self.data_dict[name]
#         motion, m_length, text_list = data['motion'], data['length'], data['text']

#         # 随机选一条 caption
#         text_data = random.choice(text_list)
#         caption = text_data['caption']
#         tokens = text_data['tokens']
#         cot_t2m = text_data.get('cot_t2m', text_data.get('cot', ""))  # 兼容
#         cot_m2t = text_data.get('cot_m2t', "")

#         # 词向量与 POS
#         if len(tokens) < self.max_text_len:
#             tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#             sent_len = len(tokens)
#             tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
#         else:
#             tokens = tokens[:self.max_text_len]
#             tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#             sent_len = len(tokens)

#         pos_one_hots = []
#         word_embeddings = []
#         for token in tokens:
#             word_emb, pos_oh = self.w_vectorizer[token]
#             pos_one_hots.append(pos_oh[None, :])
#             word_embeddings.append(word_emb[None, :])
#         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#         word_embeddings = np.concatenate(word_embeddings, axis=0)

#         # 子片段裁剪
#         if self.unit_length < 10:
#             coin2 = np.random.choice(['single', 'single', 'double'])
#         else:
#             coin2 = 'single'
#         if coin2 == 'double':
#             m_length = (m_length // self.unit_length - 1) * self.unit_length
#         elif coin2 == 'single':
#             m_length = (m_length // self.unit_length) * self.unit_length
#         idx = random.randint(0, len(motion) - m_length)
#         motion = motion[idx:idx + m_length]

#         # 归一化 & pad 到 max_motion_length
#         motion = (motion - self.mean) / self.std
#         if m_length < self.max_motion_length:
#             motion = np.concatenate(
#                 [motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))],
#                 axis=0
#             )

#         # === 返回结构保持前 9 项不变，末尾新增第 10 项为 cot_m2t ===
#         # 0: word_embeddings
#         # 1: pos_one_hots
#         # 2: caption
#         # 3: sent_len
#         # 4: motion
#         # 5: m_length
#         # 6: '_'.join(tokens)
#         # 7: name
#         # 8: cot_t2m (兼容原 'cot')
#         # 9: cot_m2t  (新增)
#         return (
#             word_embeddings, pos_one_hots, caption, sent_len,
#             motion, m_length, '_'.join(tokens), name,
#             cot_t2m, cot_m2t
#         )


# def DATALoader(
#     dataset_name, is_test, batch_size, w_vectorizer,
#     num_workers=8, unit_length=4,
#     cot_t2m_dir: str = "final",
#     cot_m2t_dir: str = "final_m2t",
# ):
#     ds = Text2MotionDataset(
#         dataset_name, is_test, w_vectorizer,
#         unit_length=unit_length,
#         cot_t2m_dir=cot_t2m_dir,
#         cot_m2t_dir=cot_m2t_dir,
#     )
#     val_loader = torch.utils.data.DataLoader(
#         ds,
#         batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         collate_fn=collate_fn,
#         drop_last=True
#     )
#     return val_loader


# def cycle(iterable):
#     while True:
#         for x in iterable:
#             yield x
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from _paths import project_path
from . import paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, type, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.type = type
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        if dataset_name == 't2m':
            self.data_root = project_path('dataset', 'HumanML3D')
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = project_path('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta')
        elif dataset_name == 'kit':
            self.data_root = project_path('dataset', 'KIT-ML')
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = project_path('checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta')

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
        if type == 'test':
            split_file = pjoin(self.data_root, 'test.txt')
        elif type == 'val':
            split_file = pjoin(self.data_root, 'val.txt')
        elif type == 'train':
            split_file = pjoin(self.data_root, 'train.txt')
        else:
            raise ValueError('Invalid type')

        min_motion_len = 40 if self.dataset_name =='t2m' else 24
        # min_motion_len = 64

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for i, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens

                        # 读取思维链文件 (可选)
                        cot_file = pjoin(self.data_root, "final", f"{name}_{i}.txt")
                        if os.path.exists(cot_file):
                            with cs.open(cot_file, "r", encoding="utf-8") as cf:
                                chain_of_thought = cf.read().strip()
                                chain_of_thought = chain_of_thought.replace("<thinking>", "").replace("</thinking>", "")
                        else:
                            print("error!!!")
                            chain_of_thought = ""
                        text_dict['cot'] = chain_of_thought   # 新增字段
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        # data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens,cot = text_data['caption'], text_data['tokens'],text_data['cot']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name,cot




def DATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4) : 
    
    val_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length),
                                              batch_size,
                                              shuffle = False,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
