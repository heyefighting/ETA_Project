import torch
import torch.nn as nn


class SpatialTemporal(nn.Module):  # 继承Module类
    """Generates feature vectors Vsp and Vtp for each particular cell"""

    spatial_emb_dims = [
        ('G_X', 256, 100),
        ('G_Y', 256, 100)
    ]

    temporal_emb_dims = [
        ('day_bin', 7, 100),
        ('hour_bin', 24, 100),
        ('time_bin', 288, 100)
    ]

    def __init__(self):
        super(SpatialTemporal, self).__init__()
        # print(SpatialTemporal.spatial_emb_dims + SpatialTemporal.temporal_emb_dims) :
        # [('G_X', 256, 100), ('G_Y', 256, 100), ('day_bin', 7, 100), ('hour_bin', 24, 100), ('time_bin', 288, 100)]
        self.build()

    def build(self):
        # G_X_em,G_Y_em:将一个child module 添加到当前module, 被添加的module可以通过name属性来获取即build.G_X_em,G_Y_em
        #  nn.Embedding(256，100):256表示有256个词，100表示100维度,字典中有256个词，词向量维度为100
        for name, dim_in, dim_out in (SpatialTemporal.spatial_emb_dims + SpatialTemporal.temporal_emb_dims):
            # print(name,dim_in,dim_out)
            # G_X 256 100
            # G_Y 256 100
            # day_bin 7 100
            # hour_bin 24 100
            # time_bin 288 100
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))  # 矩阵

        for module in self.modules():
            if type(module) is not nn.Embedding:
                continue
            # pytorch 中的 state_dict 是一个简单的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
            nn.init.uniform_(module.state_dict()['weight'], a=-1, b=1)  # 服从均匀分布~U(a,b)

    def forward(self, stats, temporal, spatial):

        V_tp = []
        for name, dim_in, dim_out in SpatialTemporal.temporal_emb_dims:
            embed = getattr(self, name + '_em')
            temporal_t = temporal[name].view(-1, 1)
            temporal_t = torch.squeeze(embed(temporal_t))
            V_tp.append(temporal_t)

        V_sp = []
        for name, dim_in, dim_out in SpatialTemporal.spatial_emb_dims:
            embed = getattr(self, name + '_em')
            spatial_t = spatial[name].view(-1, 1)
            spatial_t = torch.squeeze(embed(spatial_t))
            V_sp.append(spatial_t)

        V_tp = torch.cat(V_tp, dim=1)  # [300]
        V_sp = torch.cat(V_sp, dim=1)  # [200]
        return V_sp, V_tp

# SpatialTemporal()
