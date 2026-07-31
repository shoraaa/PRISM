'''
Copyright (C) 2026  CIAM Group, Southern University of Science and Technology, Shenzhen, China.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Model_LIB import *
from problem.ProblemSet import ProblemSet

class Model(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)

        embedding_dim = self.model_params['embedding_dim']
        self.position_embedding  = nn.Linear(3, embedding_dim, bias=False) # coordinate embedding
        self.attribute_embedding = nn.Linear(6, embedding_dim, bias=False)  # attribute embedding
        self.node_type_embedding = nn.Linear(5, embedding_dim, bias=False)  # node embedding

        self.problem_name = None
        self.problem_representation = None

        self.encoded_nodes = None
        # shape: (batch, node, embedding)

    def set_decoder_type(self,decoder_type):
        self.model_params['eval_type'] = decoder_type

    def pre_forward(self, reset_state, problem_name,problem_representation):

        self.problem_name = problem_name
        self.problem_representation = problem_representation
        dist = reset_state.dist

        log_scale = reset_state.log_scale
        self.decoder.log_scale = log_scale # for decoder

        # distance normalization
        ##################################################
        dist_normed = distance_normalization(dist, dist_norm_style="all_max")
        dist_normed_transpose = distance_normalization(dist.transpose(1, 2), dist_norm_style="all_max")

        negative_scale_dist = -1 * log_scale * dist_normed
        negative_scale_dist_transpose = -1 * log_scale * dist_normed_transpose
        # shape: (batch, problem, problem)

        #relation matrix
        ##################################################
        relation = reset_state.relation
        negative_scale_relation = None
        if relation is not None:
            assert 'pd' in problem_name, "relation matrix is only for pd problem"
            negative_scale_relation = -1 * relation

        # unified node feature extraction
        ##################################################
        # position embedding
        position_features = unified_node_position_construction(reset_state.problems, problem_name)
        # shape: (batch, problem, 3)  # 3: random identifier, x, y
        position_embedded = self.position_embedding(position_features)
        # shape: (batch, problem, embedding)

        attribute_node_type_features = unified_node_attribute_construction(reset_state.problems, problem_name,demand_max1=self.model_params['demand_max1'])
        attribute_embedded = self.attribute_embedding(attribute_node_type_features[:, :, :6])
        # shape: (batch, problem, embedding), attribute embedded
        node_type_embedded = self.node_type_embedding(attribute_node_type_features[:, :, 6:])
        # shape: (batch, problem, embedding), node type embedded

        init_emb = position_embedded + attribute_embedded + node_type_embedded
        # shape: (batch, problem, embedding)
        self.encoded_nodes = self.encoder(init_emb,
                                          negative_scale_dist,
                                          negative_scale_dist_transpose,
                                          negative_scale_relation,
                                          problem_representation)
        # shape: (batch, problem, embedding_dim)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state,cur_dist):
        batch_size = state.batch_size
        pomo_size = state.pomo_size
        if state.selected_count == 0:  # First Move, depot
            if self.problem_name in ProblemSet.get(excluded="md") and self.problem_name not in ['tsp','atsp']:  # For problem with single depot, we only select the depot node as the first Move
                # For problem with one depot, we need to select the depot node first
                selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
                prob = torch.ones(size=(batch_size, pomo_size))
            elif self.problem_name in ['tsp','atsp']:
                selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                prob = torch.ones(size=(batch_size, pomo_size))
            elif "md" in self.problem_name:
                selected = torch.arange(state.depot_num).repeat_interleave(pomo_size // state.depot_num)
                selected = selected.unsqueeze(0).expand(batch_size, -1)
                prob = torch.ones(size=(batch_size, pomo_size))
            else:
                raise NotImplementedError(f"problem_name: {self.problem_name} is not implemented!")

            encoded_first_node = get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        elif state.selected_count == 1 and pomo_size > 1 and self.problem_name not in ['tsp','atsp']:  # Second Move, POMO
            # Following the MTPOMO and MVMoE design, ensure that backhaul nodes are not selected as the first starting node
            # Note that it is not mandatory, as even if the first node is a backhaul node, we ensure that the sub-route does not revisit any linehaul nodes.
            if self.problem_name in ProblemSet.get(included="b", excluded="bp"):
                #For VRPB, node with negative demand can not be selected as second Move
                selected = state.START_NODE
                prob = torch.ones(size=(batch_size, pomo_size))
            elif "md" in self.problem_name:
                problem_size = cur_dist.shape[-1]-state.depot_num
                selected = torch.arange(start=state.depot_num, end=problem_size + state.depot_num)
                selected = selected.repeat(batch_size, (pomo_size + problem_size - 1) // problem_size)
                selected = selected[:, :pomo_size]
                prob = torch.ones(size=(batch_size, pomo_size))
            else:
                selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
                prob = torch.ones(size=(batch_size, pomo_size))
        else:
            encoded_last_node = get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            if self.problem_name in ['tsp','atsp','pdtsp','apdtsp']:
                constraint = None
            elif "vrp" in self.problem_name:
                constraint = state.load.unsqueeze(-1)
                # shape: (batch, pomo, 1)
            elif self.problem_name in ['op']:
                constraint = (state.tour_maxlength / 4.0).unsqueeze(-1)
            elif self.problem_name in ['aop']:
                constraint = state.tour_maxlength.unsqueeze(-1)
            elif self.problem_name in ['pctsp', 'spctsp','apctsp', 'aspctsp']:
                constraint = (1.0 - state.collected_prize).unsqueeze(-1)
            else:
                raise NotImplementedError(f"problem_name: {self.problem_name} is not implemented!")


            probs = self.decoder(encoded_last_node,
                                 cur_dist,
                                 ninf_mask=state.ninf_mask,
                                 constraint=constraint)
            # shape: (batch, pomo, problem)
            if self.training:
                assert self.model_params['eval_type'] == 'sampling', "During training, only sampling is allowed"
            selected, prob = select_next_node(probs, decoding_strategy=self.model_params['eval_type'])

        return selected, prob


########################################
# ENCODER
########################################

class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder_layer_num = self.model_params['encoder_layer_num']

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.encoder_layer_num)])

    def forward(self, init_emb,negative_scale_dist, negative_scale_dist_transpose,negative_scale_relation,problem_representation):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # dist.shape: (batch, row_cnt, col_cnt)

        out = init_emb
        for layer in self.layers:
            out = layer(out,negative_scale_dist, negative_scale_dist_transpose,negative_scale_relation,problem_representation)

        return out

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.Wq_row = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_row = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_row = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_col = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_col = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_col = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_relation = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_relation = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_relation = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.adaptation_bias_module_row = Adaptation_Bias_Module(**model_params)
        self.adaptation_bias_module_col = Adaptation_Bias_Module(**model_params)
        self.adaptation_bias_module_relation = Adaptation_Bias_Module(**model_params)

        self.row_col_relation_combine = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)

        self.alpha_attn_row = None  # adaptation bias for attention
        self.alpha_attn_col = None  # adaptation bias for attention
        self.alpha_attn_relation = None  # adaptation bias for attention

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = Feed_Forward_Module(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input_emb, negative_scale_dist, negative_scale_dist_transpose,negative_scale_relation,problem_representation):
        # input_emb.shape: (batch, problem, embedding)
        # cost_mat.shape: (batch, problem, problem)

        # row, standard distance
        ########################################################
        q_row = self.Wq_row(input_emb)
        k_row = self.Wk_row(input_emb)
        v_row = self.Wv_row(input_emb)
        # shape: (batch, problem, embedding_dim)
        self.alpha_attn_row = self.adaptation_bias_module_row(problem_representation)
        alpha_adaptation_bias_attn_row = self.alpha_attn_row * negative_scale_dist
        out_attn_row = adaptation_attention_free_module(q_row, k_row, v_row, alpha_adaptation_bias_attn_row)
        # shape: (batch, problem, embedding)

        # column, distance transpose
        #########################################################
        q_col = self.Wq_col(input_emb)
        k_col = self.Wk_col(input_emb)
        v_col = self.Wv_col(input_emb)
        # shape: (batch, problem, embedding_dim)
        self.alpha_attn_col = self.adaptation_bias_module_col(problem_representation)
        alpha_adaptation_bias_attn_col = self.alpha_attn_col * negative_scale_dist_transpose
        out_attn_col = adaptation_attention_free_module(q_col, k_col, v_col, alpha_adaptation_bias_attn_col)
        # shape: (batch, problem, embedding)

        if negative_scale_relation is not None:
            q_relation = self.Wq_relation(input_emb)
            k_relation = self.Wk_relation(input_emb)
            v_relation = self.Wv_relation(input_emb)
            # shape: (batch, problem, embedding_dim)
            self.alpha_attn_relation = self.adaptation_bias_module_relation(problem_representation)
            alpha_adaptation_bias_attn_relation = self.alpha_attn_relation * negative_scale_relation #
            out_attn_relation = adaptation_attention_free_module(q_relation, k_relation, v_relation, alpha_adaptation_bias_attn_relation)
        else:
            out_attn_relation = torch.zeros_like(out_attn_row)

        # combine row, column and relation
        out_attn = self.row_col_relation_combine(torch.cat([out_attn_row, out_attn_col,out_attn_relation], dim=-1))
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input_emb, out_attn)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)

########################################
# DECODER
########################################

class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        hyper_input_dim = 13
        hyper_hidden_embd_dim = 256
        self.embd_dim = hyper_input_dim  
        self.hyper_output_dim = 5 * self.embd_dim

        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)  # problem_type_num -> 256
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)  # 256->256
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)  # 256-> 45

        self.hyper_Wq_first = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wq_last = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wc = nn.Linear(self.embd_dim, 1 * embedding_dim, bias=False)  # constraints embedding

        self.Wq_first_para = None
        self.Wq_last_para = None
        self.Wk_para = None
        self.Wv_para = None
        self.Wc_para = None

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

        self.adaptation_bias_module_attn = Adaptation_Bias_Module(**model_params)
        self.adaptation_bias_module_com = Adaptation_Bias_Module(**model_params)
        self.alpha_attn = None  # adaptation bias for attention
        self.alpha_com = None  # adaptation bias for compatibility
        self.logit_clipping = self.model_params['logit_clipping']

    def assign(self, problem_representation):  # assign->pre_forward->forward
        embedding_dim = self.model_params['embedding_dim']

        hyper_embd = self.hyper_fc1(problem_representation)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        # mid_embd.shape: (hyper_output_dim,)

        self.Wq_first_para = self.hyper_Wq_first(mid_embd[:self.embd_dim]).reshape(embedding_dim, embedding_dim)
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[self.embd_dim: 2 * self.embd_dim]).reshape(embedding_dim,embedding_dim)
        self.Wk_para = self.hyper_Wk(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim,embedding_dim)
        self.Wv_para = self.hyper_Wv(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(embedding_dim,embedding_dim)
        # Note that F.linear execute calculation in the form of xW^T + b, so we need to transpose the weight matrix.
        self.Wc_para = self.hyper_Wc(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(embedding_dim, 1)

        self.alpha_attn = self.adaptation_bias_module_attn(problem_representation)
        self.alpha_com = self.adaptation_bias_module_com(problem_representation)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)

        self.k = F.linear(encoded_nodes, self.Wk_para)
        self.v = F.linear(encoded_nodes, self.Wv_para)
        # shape: (batch, problem, embedding)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q_first = F.linear(encoded_q1, self.Wq_first_para)
        # shape: (batch, head_num, pomo, qkv_dim)

    def forward(self, encoded_last_node, cur_dist,ninf_mask, constraint=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        # cur_dist.shape: (batch, pomo, problem)
        # constraints.shape: (batch, pomo,x)
        cur_dist = distance_normalization(cur_dist, dist_norm_style="sep_max")
        negative_scale_dist = -1 * self.log_scale * cur_dist # smaller value means better

        q_last = F.linear(encoded_last_node, self.Wq_last_para)
        # shape: (batch, pomo, embedding)

        if constraint is None:
            q = self.q_first + q_last
            # shape: (batch, pomo, embedding_dim)
        else:
            constraint_embedded = F.linear(constraint.clone(), self.Wc_para)
            # shape: (batch, pomo, embedding_dim)
            q = self.q_first + q_last + constraint_embedded
            # shape: (batch, pomo, embedding_dim)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        alpha_adaptation_bias_attn = self.alpha_attn * negative_scale_dist
        out_attn = adaptation_attention_free_module(q, self.k, self.v, alpha_adaptation_bias_attn, ninf_mask)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(out_attn, self.single_head_key)
        # shape: (batch, pomo, problem)
        score_scaled = score / torch.sqrt(torch.tensor(self.model_params['embedding_dim'], dtype=torch.float))
        # shape: (batch, pomo, problem)

        alpha_adaptation_bias_com = self.alpha_com * negative_scale_dist
        score_scaled = score_scaled + alpha_adaptation_bias_com
        # shape: (batch, pomo, problem)
        score_clipped = self.logit_clipping * torch.tanh(score_scaled)
        # shape: (batch, pomo, problem)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=-1)
        # shape: (batch, pomo, problem)

        return probs



