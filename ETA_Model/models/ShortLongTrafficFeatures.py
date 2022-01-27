import torch
import torch.nn as nn


class ShortTermLSTM(nn.Module):
    def __init__(self, ):
        super(ShortTermLSTM, self).__init__()

        # input_size 输入数据的特征维数，通常就是embedding_dim(词向量的维度)
        # hidden_size　LSTM中隐层的维度
        # num_layers　循环神经网络的层数
        # batch_first 这个要注意，通常我们输入的数据shape=(batch_size,seq_length,embedding_dim),
        # 而batch_first默认是False,所以我们的输入数据最好送进LSTM之前将batch_size与seq_length这两个维度调换
        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=300,
            num_layers=1,
            batch_first=True
        )

        nn.init.uniform_(self.lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)

    def forward(self, short_ttf):
        length = len(short_ttf)
        # V_short = []
        # y = 0
        cell_tensor = torch.zeros(size=[1, length, 12]).to('cpu')
        # for cell in short_ttf:
        for idx, cell in enumerate(short_ttf):
            # inputs_0, inputs_1, inputs_2 = cell[0], cell[1], cell[2]
            inputs_0, inputs_1, inputs_2 = cell[0][0, :], cell[1][0, :], cell[2][0, :]
            # print(inputs_0, inputs_1, inputs_2)  # 1*4
            # inputs_0 = torch.unsqueeze(inputs_0, dim=0)
            # inputs_1 = torch.unsqueeze(inputs_1, dim=0)
            # inputs_2 = torch.unsqueeze(inputs_2, dim=0)
            # print(inputs_0, inputs_1, inputs_2)
            # inputs_0:1*1*4
            # -->(batch_size,seq_len,input_size)
            # batch_size个句子，每个句子seq_len个单词，单词用input_size维的向量表示
            # y += 1
            # print(y)
            input = torch.cat((inputs_0, inputs_1, inputs_2), dim=-1)
            cell_tensor[:, idx, :] = input

            # outputs_0, (h_n, c_n) = self.lstm(inputs_0)
            # outputs_1, (h_n, c_n) = self.lstm(inputs_1)
            # outputs_2, (h_n, c_n) = self.lstm(inputs_2)
            # print(outputs_0, outputs_1, outputs_2)
            # print(torch.isnan(outputs_0))
            # hiddens_v = torch.squeeze(torch.cat([outputs_0[:, -1], outputs_1[:, -1], outputs_2[:, -1]], dim=1), dim=0)

            # V_short.append(hiddens_v)
        output, (h_n, c_n) = self.lstm(cell_tensor)
        output = torch.squeeze(output, dim=0)
        return output


class LongTermLSTM(nn.Module):

    def __init__(self, ):
        super(LongTermLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        nn.init.uniform_(self.lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)

    def forward(self, long_ttf):
        # V_long = []
        #
        # for cell in long_ttf:
        #     inputs = torch.unsqueeze(cell, dim=0)
        #     outputs, (h_n, c_n) = self.lstm(inputs)
        #
        #     V_long.append(torch.squeeze(outputs[:, -1], dim=0))
        #
        # return V_long

        length = len(long_ttf)
        cell_tensor = torch.zeros(size=[1, length, 4]).to('cpu')
        for idx, cell in enumerate(long_ttf):
            cell_tensor[:, idx, :] = cell[0, :]
        output, (h_n, c_n) = self.lstm(cell_tensor)
        output = torch.squeeze(output, dim=0)
        return output
