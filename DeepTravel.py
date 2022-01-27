import torch
import torch.nn as nn
import numpy as np

from models.ShortLongTrafficFeatures import ShortTermLSTM, LongTermLSTM
from models.SpatialTemporal import SpatialTemporal
from models.PredictionBiLSTM import PredictionBiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepTravel(nn.Module):
    def __init__(self, alpha=0.3):
        super(DeepTravel, self).__init__()

        self.alpha = alpha  # alpha=0.3
        self.build()  # 搭建神经网络
        self.init_weight()  # 初始化网络的权重

    def build(self):
        self.spatial_temporal = SpatialTemporal()  # 时空嵌入：时间因素Vtp向量、空间因素Vsp向量对应每一个方格cell
        self.short_term_lstm = ShortTermLSTM()  # Short-term traffic features
        self.long_term_lstm = LongTermLSTM()  # long-term traffic features

        self.prediction = PredictionBiLSTM()  # 预测层：输入当前方格的[Vsp,Vtp,Vdri,Vshort,Vlong],输出

        self.linear = torch.nn.Linear(in_features=200, out_features=1)

    def evaluate(self, stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers):
        H_cells = self(stats, temporal, spatial, dr_state, short_ttf, long_ttf)
        loss = self.dual_loss(H_cells, helpers['time_gap'], helpers['borders'], helpers['mask'])

        return loss

    def forward(self, stats, temporal, spatial, dr_state, short_ttf, long_ttf):
        V_sp, V_tp = self.spatial_temporal(stats, temporal, spatial)  # [path_len, 10]; [path_len, 12]

        V_dri = dr_state.view(-1, 4)  # [path_len, 4]
        # print('xxxx')
        # print(short_ttf) 1*3个tensor
        V_short = self.short_term_lstm(short_ttf)  # [300] for each cell
        # print('DeepTravel_forward_V_short')
        V_long = self.long_term_lstm(long_ttf)  # [100] for each cell

        V_cells = []
        for i in range(len(short_ttf)):
            V_cells.append(torch.cat([V_sp[0], V_tp[0], V_dri[0], V_short[0], V_long[0]]))  # path_len x [904]

        H_cells = self.prediction(V_cells)
        # print(H_cells)
        return H_cells

    def dual_loss(self, H_cells, times, borders, mask):

        n = len(H_cells)

        h_f = []
        h_b = []

        for i in range(1, n):
            h_f.append(sum(H_cells[:i]))
            h_b.append(sum(H_cells[i:]))

        h_f.append(sum(H_cells))

        T_f_hat = self.linear(torch.stack(h_f)).squeeze(dim=1)
        T_b_hat = self.linear(torch.stack(h_b)).squeeze(dim=1)

        # T_b_hat = torch.cat([T_b_hat, torch.Tensor([0]).cuda()])
        T_b_hat = torch.cat([T_b_hat, torch.Tensor([0]).to(device)])

        M = np.zeros(n)
        T_f = np.ones(n)
        T_b = np.ones(n)

        for ind in range(len(H_cells)):
            if not mask[ind]:
                borders = np.insert(borders, ind, 0)
            if mask[ind]:
                M[ind] = 1
                T_f[ind] = borders[ind]
                T_b[ind] = times[-1] - borders[ind]

        M = torch.Tensor(M).to(device)
        T_f = torch.Tensor(T_f).to(device)
        T_b = torch.Tensor(T_b).to(device)
        print("T_f_hat")
        print(T_f_hat)
        print("T_f")
        print(T_f)
        # loss = (M * ((T_f_hat - T_f) / T_f) ** 2 + M * ((T_b_hat - T_b) / T_b) ** 2) / (M * 2)
        loss = (M * ((T_f_hat - T_f) / (T_f + 1e-4)) ** 2 + M * ((T_b_hat - T_b) / (T_b + 1e-4)) ** 2) / (
                    (M + 1e-4) * 2)
        # Temporary forced operation
        loss[loss != loss] = 0

        return loss

    def init_weight(self):
        for name, param in self.named_parameters():  # 给出网络层的名字和参数的迭代器
            if name.find('.bias') != -1:  # 包含字符串.bias
                param.data.fill_(0)
            elif name.find('.weight') != -1:  # 包含字符串.weight
                nn.init.uniform_(param.data)
