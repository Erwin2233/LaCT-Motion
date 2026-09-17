"""
评估器模型包装器

用于计算文本-动作的联合嵌入，用于语义相似度和动作相似度奖励计算。
"""

import torch
from os.path import join as pjoin
import numpy as np
from third_party.trl_motion.t2m.models.modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo
from third_party.trl_motion.t2m.utils.word_vectorizer import POS_enumerator


def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        """
        输入:
        word_embs: [B, L, E]  float
        pos_ohot : [B, L, P]  float
        cap_lens : [B]        int / Tensor
        motions  : [B, T, D]  float
        m_lens   : [B]        int / Tensor
        返回:
        text_embedding  : [B, D_txt]
        motion_embedding: [B, D_mot]
        说明:
        - 文本侧在这里做长度降序排序 -> 编码 -> 再按原顺序还原
        - 动作侧保持原顺序（motion_encoder 内部已处理 pack 的排序/或 enforce_sorted=False）
        """
        with torch.no_grad():
            dev = self.device

            # ---- to device / dtype ----
            word_embs = (word_embs if isinstance(word_embs, torch.Tensor) else torch.tensor(word_embs)).to(dev, dtype=torch.float32)
            pos_ohot = (pos_ohot if isinstance(pos_ohot, torch.Tensor) else torch.tensor(pos_ohot)).to(dev, dtype=torch.float32)
            motions = (motions if isinstance(motions, torch.Tensor) else torch.tensor(motions)).to(dev, dtype=torch.float32)

            # ---- Motion encoding ----
            movements = self.movement_encoder(motions[..., :-4]).detach()
            if not isinstance(m_lens, torch.Tensor):
                m_lens = torch.as_tensor(m_lens)
            m_lens = (m_lens // self.opt.unit_length).to(dev, dtype=torch.long)
            m_lens = torch.clamp(m_lens, min=1)
            motion_embedding = self.motion_encoder(movements, m_lens)  # [B, D_mot]

            # ---- Text encoding (sort by length desc -> encode -> unsort) ----
            if not isinstance(cap_lens, torch.Tensor):
                cap_lens = torch.as_tensor(cap_lens)

            # pack_padded_sequence 要求 lengths 在 CPU，int64
            cap_lens_cpu = cap_lens.detach().to('cpu', dtype=torch.long).view(-1)
            cap_lens_cpu = torch.clamp(cap_lens_cpu, min=1)

            # 降序排序（保持稳定），得到排序索引 & 逆索引
            sorted_len, sort_idx = torch.sort(cap_lens_cpu, descending=True)
            # 逆置映射: inv_idx[sort_idx] = arange(B)
            inv_idx = torch.empty_like(sort_idx)
            inv_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)

            # 按排序重排文本输入（在 GPU 上按样本维 0 索引）
            sort_idx_dev = sort_idx.to(dev)
            word_embs_sorted = word_embs.index_select(0, sort_idx_dev)
            pos_ohot_sorted = pos_ohot.index_select(0, sort_idx_dev)

            # 送入文本编码器（lengths 用 CPU Long）
            text_embedding_sorted = self.text_encoder(word_embs_sorted, pos_ohot_sorted, sorted_len)  # [B, D_txt]

            # 还原原批顺序
            inv_idx_dev = inv_idx.to(text_embedding_sorted.device)
            text_embedding = text_embedding_sorted.index_select(0, inv_idx_dev)  # [B, D_txt]

        return text_embedding, motion_embedding

    def get_text_embeddings(self, word_embs, pos_ohot, cap_lens):
        """
        单独获取文本嵌入，用于 M2T 任务的语义相似度计算。

        输入:
        word_embs: [B, L, E]  float - 词向量
        pos_ohot : [B, L, P]  float - POS one-hot
        cap_lens : [B]        int / Tensor - 文本长度

        返回:
        text_embedding: [B, D_txt] - 文本嵌入向量
        """
        with torch.no_grad():
            dev = self.device

            # ---- to device / dtype ----
            word_embs = (word_embs if isinstance(word_embs, torch.Tensor) else torch.tensor(word_embs)).to(dev, dtype=torch.float32)
            pos_ohot = (pos_ohot if isinstance(pos_ohot, torch.Tensor) else torch.tensor(pos_ohot)).to(dev, dtype=torch.float32)

            # ---- Text encoding (sort by length desc -> encode -> unsort) ----
            if not isinstance(cap_lens, torch.Tensor):
                cap_lens = torch.as_tensor(cap_lens)

            # pack_padded_sequence 要求 lengths 在 CPU，int64
            cap_lens_cpu = cap_lens.detach().to('cpu', dtype=torch.long).view(-1)
            cap_lens_cpu = torch.clamp(cap_lens_cpu, min=1)

            # 降序排序（保持稳定），得到排序索引 & 逆索引
            sorted_len, sort_idx = torch.sort(cap_lens_cpu, descending=True)
            # 逆置映射: inv_idx[sort_idx] = arange(B)
            inv_idx = torch.empty_like(sort_idx)
            inv_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)

            # 按排序重排文本输入（在 GPU 上按样本维 0 索引）
            sort_idx_dev = sort_idx.to(dev)
            word_embs_sorted = word_embs.index_select(0, sort_idx_dev)
            pos_ohot_sorted = pos_ohot.index_select(0, sort_idx_dev)

            # 送入文本编码器（lengths 用 CPU Long）
            text_embedding_sorted = self.text_encoder(word_embs_sorted, pos_ohot_sorted, sorted_len)  # [B, D_txt]

            # 还原原批顺序
            inv_idx_dev = inv_idx.to(text_embedding_sorted.device)
            text_embedding = text_embedding_sorted.index_select(0, inv_idx_dev)  # [B, D_txt]

        return text_embedding

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
