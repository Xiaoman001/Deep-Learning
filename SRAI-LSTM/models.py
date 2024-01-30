'''
Main Models
Author: Pu Zhang
Date: 2019/7/1
'''

import torch
import torch.nn as nn
from basemodels import *
from utils import get_noise


class SRA_LSTM(nn.Module):
    def __init__(self, args):
        super(SRA_LSTM, self).__init__()
        self.args = args
        self.ifdropout = args.ifdropout
        self.dropratio = args.dropratio
        self.rela_dropratio = args.rela_dropratio
        self.using_cuda = args.using_cuda
        self.noise_dim = args.noise_dim
        self.inputLayer = nn.Linear(args.input_size, args.input_embed_size)
        self.rela_encoder = RelationEncoder(embedding_dim=args.rela_embed_size, h_dim=args.rela_hidden_size, dropratio=self.rela_dropratio)
        self.social_model = SocialInteraction(
            r_dim=args.rela_hidden_size, m_dim=args.rnn_size, s_dim=args.social_tensor_size)
        self.cell = nn.LSTMCell(args.input_embed_size + args.social_tensor_size, args.rnn_size)
        if self.noise_dim is not None:
            self.fusion = nn.Linear(args.rnn_size + self.noise_dim, args.rnn_size)
        self.outputLayer = nn.Linear(args.rnn_size, args.output_size)
        self.dropout = nn.Dropout(self.dropratio)

        self.input_Ac = nn.ReLU()

        if args.using_cuda:
            self = self.cuda(device=args.gpu)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.inputLayer.bias, 0.0)
        nn.init.normal_(self.inputLayer.weight, std=self.args.std_in)

        nn.init.xavier_uniform_(self.cell.weight_ih)
        nn.init.orthogonal_(self.cell.weight_hh, gain=0.001)

        nn.init.constant_(self.cell.bias_ih, 0.0)
        nn.init.constant_(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant_(self.cell.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant_(self.outputLayer.bias, 0.0)
        nn.init.normal_(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs, iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]
        # print(nodes_abs.shape)

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)
        rela_hidden_states = torch.zeros(num_Ped, num_Ped, self.args.rela_hidden_size)
        rela_cell_states = torch.zeros(num_Ped, num_Ped, self.args.rela_hidden_size)
        noise = get_noise((1, self.noise_dim), 'gaussian')
        if self.using_cuda:
            outputs = outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            rela_hidden_states = rela_hidden_states.cuda()
            rela_cell_states = rela_cell_states.cuda()

        # For each frame in the sequence
        for framenum in range(self.args.seq_length - 1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index].clone()

                nodes_abs = shift_value[framenum, node_index] + nodes_current

                nodes_abs = nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index = nodes_abs.transpose(0, 1).contiguous() - nodes_abs
            else:
                node_index = seq_list[framenum] > 0
                nodes_current = nodes_norm[framenum, node_index]
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0, 1).contiguous() - corr
                nei_num_index = nei_num[framenum, node_index]

            hidden_states_current = hidden_states[node_index]
            cell_states_current = cell_states[node_index]

            rela_ht_current_nodes = rela_hidden_states[node_index, :, :]
            rela_ct_current_nodes = rela_cell_states[node_index, :, :]

            rela_ht_current_nei = rela_ht_current_nodes[:, node_index, :]
            rela_ct_current_nei = rela_ct_current_nodes[:, node_index, :]

            rela_ht_current, rela_ct_current = self.rela_encoder(
                corr_index, rela_ht_current_nei, rela_ct_current_nei, nei_index
            )

            social_tensor = self.social_model(hidden_states_current, rela_ht_current, corr_index, nei_index)
            input_embedding = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))
            cat_tensor = torch.cat((input_embedding, social_tensor), 1)

            if self.noise_dim is not None and framenum == self.args.obs_length - 1:
                noise_to_cat = noise.repeat(hidden_states_current.shape[0], 1)
                hidden_states_current = torch.cat((hidden_states_current, noise_to_cat), dim=1)
                hidden_states_current = self.fusion(hidden_states_current)

            hidden_states_current, cell_states_current = self.cell.forward(
                cat_tensor, (hidden_states_current, cell_states_current)
            )
            # print(hidden_states_current.shape)

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum, node_index] = outputs_current
            hidden_states[node_index] = hidden_states_current
            cell_states[node_index] = cell_states_current
            rela_ht_current_nodes[:, node_index, :] = rela_ht_current
            rela_ct_current_nodes[:, node_index, :] = rela_ct_current
            rela_hidden_states[node_index, :, :] = rela_ht_current_nodes
            rela_cell_states[node_index, :, :] = rela_ct_current_nodes

        return outputs


class RelationEncoder(nn.Module):
    def __init__(
        self, embedding_dim=32, h_dim=64, dropratio=0.0
    ):
        super(RelationEncoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.relative_embedding = nn.Linear(2, embedding_dim)
        self.relation_encoder = nn.LSTMCell(
            embedding_dim, h_dim
        )
        self.ac = nn.ReLU()
        self.dropout = nn.Dropout(dropratio)

        nn.init.constant_(self.relative_embedding.bias, 0.0)
        nn.init.normal_(self.relative_embedding.weight, mean=0, std=0.3)

    def forward(self, corr_index, rela_ht, rela_ct, nei_index):

        ped_num = corr_index.shape[0]
        nei_index_t = nei_index.view(-1)
        corr_t = corr_index.view((ped_num * ped_num, -1))
        rela_ht_t = rela_ht.view(ped_num * ped_num, -1)
        rela_ct_t = rela_ct.view(ped_num * ped_num, -1)
        # (nei_index)

        rela_embedding = self.dropout(self.ac(self.relative_embedding(corr_t[nei_index_t > 0])))
        rela_ht_tt, rela_ct_tt = self.relation_encoder(
            rela_embedding, (rela_ht_t[nei_index_t > 0], rela_ct_t[nei_index_t > 0])
        )

        rela_ht_t[nei_index_t > 0] = rela_ht_tt
        rela_ct_t[nei_index_t > 0] = rela_ct_tt

        rela_hidden_state = rela_ht_t.view(ped_num, ped_num, -1)
        rela_cell_state = rela_ct_t.view(ped_num, ped_num, -1)

        return rela_hidden_state, rela_cell_state


class RA_LSTM(nn.Module):
    # relative position attention
    def __init__(self, args):
        super(RA_LSTM, self).__init__()
        self.args = args
        self.ifdropout = args.ifdropout
        self.using_cuda = args.using_cuda
        self.inputLayer = nn.Linear(args.input_size, args.input_embed_size)
        self.social_model = SocialInteraction2(
            r_dim=args.input_embed_size, m_dim=args.rnn_size, s_dim=args.social_tensor_size)
        self.cell = nn.LSTMCell(args.input_embed_size + args.social_tensor_size, args.rnn_size)

        self.outputLayer = nn.Linear(args.rnn_size, args.output_size)
        self.dropout = nn.Dropout(0.1)

        self.input_Ac = nn.ReLU()

        if args.using_cuda:
            self = self.cuda(device=args.gpu)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.inputLayer.bias, 0.0)
        nn.init.normal_(self.inputLayer.weight, std=self.args.std_in)

        nn.init.xavier_uniform_(self.cell.weight_ih)
        nn.init.orthogonal_(self.cell.weight_hh, gain=0.001)

        nn.init.constant_(self.cell.bias_ih, 0.0)
        nn.init.constant_(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant_(self.cell.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant_(self.outputLayer.bias, 0.0)
        nn.init.normal_(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs, iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)

        if self.using_cuda:
            outputs = outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()

        # For each frame in the sequence
        for framenum in range(self.args.seq_length - 1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index].clone()

                nodes_abs = shift_value[framenum, node_index] + nodes_current

                nodes_abs = nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index = nodes_abs.transpose(0, 1).contiguous() - nodes_abs
            else:
                node_index = seq_list[framenum] > 0
                nodes_current = nodes_norm[framenum, node_index]
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0, 1).contiguous() - corr
                # nei_num_index = nei_num[framenum, node_index]

            hidden_states_current = hidden_states[node_index]
            cell_states_current = cell_states[node_index]

            social_tensor = self.social_model(hidden_states_current, corr_index, nei_index)
            input_embedding = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))
            cat_tensor = torch.cat((input_embedding, social_tensor), 1)

            hidden_states_current, cell_states_current = self.cell.forward(
                cat_tensor, (hidden_states_current, cell_states_current)
            )

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum, node_index] = outputs_current
            hidden_states[node_index] = hidden_states_current
            cell_states[node_index] = cell_states_current

        return outputs


class SA_LSTM(nn.Module):
    # relative position attention
    def __init__(self, args):
        super(SA_LSTM, self).__init__()
        self.args = args
        self.ifdropout = args.ifdropout
        self.using_cuda = args.using_cuda
        self.inputLayer = nn.Linear(args.input_size, args.input_embed_size)

        self.social_model = SocialInteraction3(
            m_dim=args.rnn_size, s_dim=args.social_tensor_size)
        self.cell = nn.LSTMCell(args.input_embed_size + args.social_tensor_size, args.rnn_size)

        self.outputLayer = nn.Linear(args.rnn_size, args.output_size)
        self.dropout = nn.Dropout(0.1)

        self.input_Ac = nn.ReLU()

        if args.using_cuda:
            self = self.cuda(device=args.gpu)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.inputLayer.bias, 0.0)
        nn.init.normal_(self.inputLayer.weight, std=self.args.std_in)

        nn.init.xavier_uniform_(self.cell.weight_ih)
        nn.init.orthogonal_(self.cell.weight_hh, gain=0.001)

        nn.init.constant_(self.cell.bias_ih, 0.0)
        nn.init.constant_(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant_(self.cell.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant_(self.outputLayer.bias, 0.0)
        nn.init.normal_(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs, iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)

        if self.using_cuda:
            outputs = outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()

        # For each frame in the sequence
        for framenum in range(self.args.seq_length - 1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index].clone()

                nodes_abs = shift_value[framenum, node_index] + nodes_current

                nodes_abs = nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index = nodes_abs.transpose(0, 1).contiguous() - nodes_abs
            else:
                node_index = seq_list[framenum] > 0
                nodes_current = nodes_norm[framenum, node_index]
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0, 1).contiguous() - corr
                # nei_num_index = nei_num[framenum, node_index]

            hidden_states_current = hidden_states[node_index]
            cell_states_current = cell_states[node_index]

            social_tensor = self.social_model(hidden_states_current, corr_index, nei_index)
            input_embedding = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))
            cat_tensor = torch.cat((input_embedding, social_tensor), 1)

            hidden_states_current, cell_states_current = self.cell.forward(
                cat_tensor, (hidden_states_current, cell_states_current)
            )

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum, node_index] = outputs_current
            hidden_states[node_index] = hidden_states_current
            cell_states[node_index] = cell_states_current

        return outputs


class SRLA_LSTM(nn.Module):
    def __init__(self, args):
        super(SRLA_LSTM, self).__init__()
        self.args = args
        self.ifdropout = args.ifdropout
        self.dropratio = args.dropratio
        self.rela_dropratio = args.rela_dropratio
        self.using_cuda = args.using_cuda
        self.inputLayer = nn.Linear(args.input_size, args.input_embed_size)
        self.rela_encoder = RelationEncoder(embedding_dim=args.rela_embed_size, h_dim=args.rela_hidden_size, dropratio=self.rela_dropratio)
        self.social_model = SocialInteraction4(
            r_dim=args.rela_hidden_size, m_dim=args.rnn_size, s_dim=args.social_tensor_size)
        self.cell = nn.LSTMCell(args.input_embed_size + args.social_tensor_size, args.rnn_size)

        self.outputLayer = nn.Linear(args.rnn_size, args.output_size)
        self.dropout = nn.Dropout(self.dropratio)

        self.input_Ac = nn.ReLU()

        if args.using_cuda:
            self = self.cuda(device=args.gpu)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.inputLayer.bias, 0.0)
        nn.init.normal_(self.inputLayer.weight, std=self.args.std_in)

        nn.init.xavier_uniform_(self.cell.weight_ih)
        nn.init.orthogonal_(self.cell.weight_hh, gain=0.001)

        nn.init.constant_(self.cell.bias_ih, 0.0)
        nn.init.constant_(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant_(self.cell.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant_(self.outputLayer.bias, 0.0)
        nn.init.normal_(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs, iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)
        rela_hidden_states = torch.zeros(num_Ped, num_Ped, self.args.rela_hidden_size)
        rela_cell_states = torch.zeros(num_Ped, num_Ped, self.args.rela_hidden_size)

        if self.using_cuda:
            outputs = outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            rela_hidden_states = rela_hidden_states.cuda()
            rela_cell_states = rela_cell_states.cuda()

        # For each frame in the sequence
        for framenum in range(self.args.seq_length - 1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index].clone()

                nodes_abs = shift_value[framenum, node_index] + nodes_current

                nodes_abs = nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index = nodes_abs.transpose(0, 1).contiguous() - nodes_abs
            else:
                node_index = seq_list[framenum] > 0
                nodes_current = nodes_norm[framenum, node_index]
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0, 1).contiguous() - corr
                nei_num_index = nei_num[framenum, node_index]

            hidden_states_current = hidden_states[node_index]
            cell_states_current = cell_states[node_index]

            rela_ht_current_nodes = rela_hidden_states[node_index, :, :]
            rela_ct_current_nodes = rela_cell_states[node_index, :, :]

            rela_ht_current_nei = rela_ht_current_nodes[:, node_index, :]
            rela_ct_current_nei = rela_ct_current_nodes[:, node_index, :]

            rela_ht_current, rela_ct_current = self.rela_encoder(
                corr_index, rela_ht_current_nei, rela_ct_current_nei, nei_index
            )

            social_tensor = self.social_model(hidden_states_current, rela_ht_current, corr_index, nei_index)
            input_embedding = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))
            cat_tensor = torch.cat((input_embedding, social_tensor), 1)

            hidden_states_current, cell_states_current = self.cell.forward(
                cat_tensor, (hidden_states_current, cell_states_current)
            )

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum, node_index] = outputs_current
            hidden_states[node_index] = hidden_states_current
            cell_states[node_index] = cell_states_current
            rela_ht_current_nodes[:, node_index, :] = rela_ht_current
            rela_ct_current_nodes[:, node_index, :] = rela_ct_current
            rela_hidden_states[node_index, :, :] = rela_ht_current_nodes
            rela_cell_states[node_index, :, :] = rela_ct_current_nodes

        return outputs


class NonA_LSTM(nn.Module):
    # relative position attention
    def __init__(self, args):
        super(NonA_LSTM, self).__init__()
        self.args = args
        self.ifdropout = args.ifdropout
        self.using_cuda = args.using_cuda
        self.inputLayer = nn.Linear(args.input_size, args.input_embed_size)
        self.social_model = SocialInteraction5(
            m_dim=args.rnn_size, s_dim=args.social_tensor_size)
        self.cell = nn.LSTMCell(args.input_embed_size + args.social_tensor_size, args.rnn_size)

        self.outputLayer = nn.Linear(args.rnn_size, args.output_size)
        self.dropout = nn.Dropout(0.1)

        self.input_Ac = nn.ReLU()

        if args.using_cuda:
            self = self.cuda(device=args.gpu)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.inputLayer.bias, 0.0)
        nn.init.normal_(self.inputLayer.weight, std=self.args.std_in)

        nn.init.xavier_uniform_(self.cell.weight_ih)
        nn.init.orthogonal_(self.cell.weight_hh, gain=0.001)

        nn.init.constant_(self.cell.bias_ih, 0.0)
        nn.init.constant_(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant_(self.cell.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant_(self.outputLayer.bias, 0.0)
        nn.init.normal_(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs, iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)

        if self.using_cuda:
            outputs = outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()

        # For each frame in the sequence
        for framenum in range(self.args.seq_length - 1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index].clone()

                nodes_abs = shift_value[framenum, node_index] + nodes_current

                nodes_abs = nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index = nodes_abs.transpose(0, 1).contiguous() - nodes_abs
            else:
                node_index = seq_list[framenum] > 0
                nodes_current = nodes_norm[framenum, node_index]
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0, 1).contiguous() - corr
                # nei_num_index = nei_num[framenum, node_index]

            hidden_states_current = hidden_states[node_index]
            cell_states_current = cell_states[node_index]

            social_tensor = self.social_model(hidden_states_current, corr_index, nei_index)
            input_embedding = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))
            cat_tensor = torch.cat((input_embedding, social_tensor), 1)

            hidden_states_current, cell_states_current = self.cell.forward(
                cat_tensor, (hidden_states_current, cell_states_current)
            )

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum, node_index] = outputs_current
            hidden_states[node_index] = hidden_states_current
            cell_states[node_index] = cell_states_current

        return outputs


class SRA_LSTM_M(nn.Module):
    def __init__(self, args):
        super(SRA_LSTM_M, self).__init__()
        self.args = args
        self.ifdropout = args.ifdropout
        self.dropratio = args.dropratio
        self.rela_dropratio = args.rela_dropratio
        self.using_cuda = args.using_cuda
        self.noise_dim = args.noise_dim
        self.seq_length = args.obs_length + args.pred_length
        self.inputLayer = nn.Linear(args.input_size, args.input_embed_size)
        self.rela_encoder = RelationEncoder(embedding_dim=args.rela_embed_size, h_dim=args.rela_hidden_size, dropratio=self.rela_dropratio)
        self.social_model = SocialInteraction(
            r_dim=args.rela_hidden_size, m_dim=args.rnn_size, s_dim=args.social_tensor_size)
        self.cell = nn.LSTMCell(args.input_embed_size + args.social_tensor_size, args.rnn_size)
        self.outputLayer = nn.Linear(args.rnn_size + args.noise_dim, args.output_size)
        self.dropout = nn.Dropout(self.dropratio)

        self.input_Ac = nn.ReLU()

        if args.using_cuda:
            self = self.cuda(device=args.gpu)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.inputLayer.bias, 0.0)
        nn.init.normal_(self.inputLayer.weight, std=self.args.std_in)

        nn.init.xavier_uniform_(self.cell.weight_ih)
        nn.init.orthogonal_(self.cell.weight_hh, gain=0.001)

        nn.init.constant_(self.cell.bias_ih, 0.0)
        nn.init.constant_(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant_(self.cell.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant_(self.outputLayer.bias, 0.0)
        nn.init.normal_(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs, iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]
        # print(nodes_abs.shape)

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)
        rela_hidden_states = torch.zeros(num_Ped, num_Ped, self.args.rela_hidden_size)
        rela_cell_states = torch.zeros(num_Ped, num_Ped, self.args.rela_hidden_size)
        noise = get_noise((1, self.noise_dim), 'gaussian')
        if self.using_cuda:
            outputs = outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            rela_hidden_states = rela_hidden_states.cuda()
            rela_cell_states = rela_cell_states.cuda()

        # For each frame in the sequence
        for framenum in range(self.seq_length - 1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index].clone()

                nodes_abs = shift_value[framenum, node_index] + nodes_current

                nodes_abs = nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index = nodes_abs.transpose(0, 1).contiguous() - nodes_abs
            else:
                node_index = seq_list[framenum] > 0
                nodes_current = nodes_norm[framenum, node_index]
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0, 1).contiguous() - corr
                nei_num_index = nei_num[framenum, node_index]

            hidden_states_current = hidden_states[node_index]
            cell_states_current = cell_states[node_index]

            rela_ht_current_nodes = rela_hidden_states[node_index, :, :]
            rela_ct_current_nodes = rela_cell_states[node_index, :, :]

            rela_ht_current_nei = rela_ht_current_nodes[:, node_index, :]
            rela_ct_current_nei = rela_ct_current_nodes[:, node_index, :]

            rela_ht_current, rela_ct_current = self.rela_encoder(
                corr_index, rela_ht_current_nei, rela_ct_current_nei, nei_index
            )

            social_tensor = self.social_model(hidden_states_current, rela_ht_current, corr_index, nei_index)
            input_embedding = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))
            cat_tensor = torch.cat((input_embedding, social_tensor), 1)

            # noise_to_cat = noise.repeat(hidden_states_current.shape[0], 1)
            # hidden_states_current = torch.cat((hidden_states_current, noise_to_cat), dim=1)

            hidden_states_current, cell_states_current = self.cell.forward(
                cat_tensor, (hidden_states_current, cell_states_current)
            )
            # print(hidden_states_current.shape)
            noise_to_cat = noise.repeat(hidden_states_current.shape[0], 1)
            temporal_input_embedded_wnoise = torch.cat((hidden_states_current, noise_to_cat), dim=1)
            outputs_current = self.outputLayer(temporal_input_embedded_wnoise)
            outputs[framenum, node_index] = outputs_current
            hidden_states[node_index] = hidden_states_current
            cell_states[node_index] = cell_states_current
            rela_ht_current_nodes[:, node_index, :] = rela_ht_current
            rela_ct_current_nodes[:, node_index, :] = rela_ct_current
            rela_hidden_states[node_index, :, :] = rela_ht_current_nodes
            rela_cell_states[node_index, :, :] = rela_ct_current_nodes

        return outputs

