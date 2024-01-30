import torch
import torch.nn as nn


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def init_weights_attr(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean=0, std=0.2)
        # nn.init.xavier_uniform_(m.weight)
        try:
            nn.init.constant_(m.bias, 0)
        except:
            pass


class SocialInteraction(nn.Module):

    def __init__(
        self, r_dim=64, m_dim=64, s_dim=64
    ):
        super(SocialInteraction, self).__init__()
        self.m_dim = m_dim
        self.r_dim = r_dim
        self.s_dim = s_dim
        self.attention = nn.Linear(m_dim * 2 + r_dim, 1, bias=True)
        self.attention.apply(init_weights_attr)
        # self.ac = nn.LeakyReLU(0.2)

    def forward(self, hidden_state, rela_state, corr_index, nei_index):

        ped_num = hidden_state.shape[0]

        nei_inputs = hidden_state.repeat(ped_num, 1)
        hi_t = nei_inputs.view((ped_num, ped_num, self.m_dim)).permute(1, 0, 2).contiguous().view(-1, self.m_dim)
        rela_t = rela_state.view(-1, self.r_dim)
        nei_index_t = nei_index.view(-1)
        corr_t = corr_index.view(ped_num * ped_num, -1)
        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return hidden_state

        cat_tensor = torch.cat((rela_t[nei_index_t > 0], hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)

        tt = self.attention(cat_tensor).view(-1)

        Pos_t = torch.zeros((ped_num * ped_num, 1)).cuda().view(-1)
        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((ped_num, ped_num))
        Pos[Pos == 0] = -1e-6
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)

        H = torch.zeros((ped_num * ped_num, self.m_dim)).cuda()
        H[nei_index_t > 0] = nei_inputs[nei_index_t > 0]
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.m_dim, 1).transpose(0, 1)
        H = H.view(ped_num, ped_num, -1)
        H_sum = torch.sum(H, 1)

        return H_sum


class SocialInteraction2(nn.Module):

    def __init__(
        self, r_dim=32, m_dim=64, s_dim=64
    ):
        super(SocialInteraction2, self).__init__()
        self.m_dim = m_dim
        self.r_dim = r_dim
        self.s_dim = s_dim

        self.attention = nn.Linear(m_dim * 2 + r_dim, 1)
        self.attention.apply(init_weights_attr)

        self.rela_embed = nn.Linear(2, r_dim)
        self.rela_embed.apply(init_weights_attr)
        self.ac = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state, corr_index, nei_index):

        ped_num = hidden_state.shape[0]

        nei_inputs = hidden_state.repeat(ped_num, 1)
        hi_t = nei_inputs.view((ped_num, ped_num, self.m_dim)).permute(1, 0, 2).contiguous().view(-1, self.m_dim)
        nei_index_t = nei_index.view(-1)
        corr_t = corr_index.view(ped_num * ped_num, -1)
        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return hidden_state
        rela_embed = self.dropout(self.ac(self.rela_embed(corr_t[nei_index_t > 0])))

        cat_tensor = torch.cat((rela_embed, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)

        tt = self.attention(cat_tensor).view(-1)

        Pos_t = torch.full((ped_num * ped_num, 1), 0, device=torch.device("cuda")).view(-1)
        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((ped_num, ped_num))
        Pos[Pos == 0] = -1e-6
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)

        H = torch.full((ped_num * ped_num, self.m_dim), 0, device=torch.device("cuda"))
        H[nei_index_t > 0] = nei_inputs[nei_index_t > 0]
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.m_dim, 1).transpose(0, 1)
        H = H.view(ped_num, ped_num, -1)
        social_tensor = torch.sum(H, 1)

        return social_tensor


class SocialInteraction3(nn.Module):

    def __init__(
        self, m_dim=64, s_dim=64
    ):
        super(SocialInteraction3, self).__init__()
        self.m_dim = m_dim
        self.s_dim = s_dim

        self.attention = nn.Linear(m_dim * 2, 1)
        self.attention.apply(init_weights_attr)

    def forward(self, hidden_state, corr_index, nei_index):

        ped_num = hidden_state.shape[0]

        nei_inputs = hidden_state.repeat(ped_num, 1)
        hi_t = nei_inputs.view((ped_num, ped_num, self.m_dim)).permute(1, 0, 2).contiguous().view(-1, self.m_dim)
        nei_index_t = nei_index.view(-1)
        corr_t = corr_index.view(ped_num * ped_num, -1)
        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return hidden_state

        cat_tensor = torch.cat((hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)

        tt = self.attention(cat_tensor).view(-1)

        Pos_t = torch.full((ped_num * ped_num, 1), 0, device=torch.device("cuda")).view(-1)
        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((ped_num, ped_num))
        Pos[Pos == 0] = -1e-6
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)

        H = torch.full((ped_num * ped_num, self.m_dim), 0, device=torch.device("cuda"))
        H[nei_index_t > 0] = nei_inputs[nei_index_t > 0]
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.m_dim, 1).transpose(0, 1)
        H = H.view(ped_num, ped_num, -1)
        social_tensor = torch.sum(H, 1)

        return social_tensor


class SocialInteraction4(nn.Module):

    def __init__(
        self, r_dim=64, m_dim=64, s_dim=64
    ):
        super(SocialInteraction4, self).__init__()
        self.m_dim = m_dim
        self.r_dim = r_dim
        self.s_dim = s_dim
        self.attention = nn.Linear(r_dim, 1, bias=True)
        self.attention.apply(init_weights_attr)

    def forward(self, hidden_state, rela_state, corr_index, nei_index):

        ped_num = hidden_state.shape[0]

        nei_inputs = hidden_state.repeat(ped_num, 1)
        # hi_t = nei_inputs.view((ped_num, ped_num, self.m_dim)).permute(1, 0, 2).contiguous().view(-1, self.m_dim)
        rela_t = rela_state.view(-1, self.r_dim)
        nei_index_t = nei_index.view(-1)
        corr_t = corr_index.view(ped_num * ped_num, -1)
        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return hidden_state

        # cat_tensor = torch.cat((rela_t[nei_index_t > 0], hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)

        tt = self.attention(rela_t[nei_index_t > 0]).view(-1)

        Pos_t = torch.full((ped_num * ped_num, 1), 0, device=torch.device("cuda")).view(-1)
        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((ped_num, ped_num))
        Pos[Pos == 0] = -1e-6
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)

        H = torch.full((ped_num * ped_num, self.m_dim), 0, device=torch.device("cuda"))
        H[nei_index_t > 0] = nei_inputs[nei_index_t > 0]
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.m_dim, 1).transpose(0, 1)
        H = H.view(ped_num, ped_num, -1)
        H_sum = torch.sum(H, 1)

        return H_sum


class SocialInteraction5(nn.Module):

    def __init__(
        self, m_dim=64, s_dim=64
    ):
        super(SocialInteraction5, self).__init__()
        self.m_dim = m_dim
        self.s_dim = s_dim

        # self.attention = nn.Linear(m_dim * 2, 1)
        # self.attention.apply(init_weights_attr)

    def forward(self, hidden_state, corr_index, nei_index):

        ped_num = hidden_state.shape[0]

        nei_inputs = hidden_state.repeat(ped_num, 1)
        hi_t = nei_inputs.view((ped_num, ped_num, self.m_dim)).permute(1, 0, 2).contiguous().view(-1, self.m_dim)
        nei_index_t = nei_index.view(-1)
        corr_t = corr_index.view(ped_num * ped_num, -1)
        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return hidden_state

        # cat_tensor = torch.cat((hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)

        # tt = self.attention(cat_tensor).view(-1)

        Pos_t = torch.full((ped_num * ped_num, 1), 0, device=torch.device("cuda")).view(-1)
        Pos_t[nei_index_t > 0] = 1
        Pos = Pos_t.view((ped_num, ped_num))
        Pos[Pos == 0] = -1e-6
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)

        H = torch.full((ped_num * ped_num, self.m_dim), 0, device=torch.device("cuda"))
        H[nei_index_t > 0] = nei_inputs[nei_index_t > 0]
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.m_dim, 1).transpose(0, 1)
        H = H.view(ped_num, ped_num, -1)
        social_tensor = torch.sum(H, 1)

        return social_tensor
