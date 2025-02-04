import torch
import torch.nn as nn
from .layers import *
from .AGCN import AVWDCRNN,AGCRNCell
import config
# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np
# from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = torch.device('cuda:0')

local_model_path = "/root/autodl-tmp/project/models/GPT-2"

# model = GPT2Model.from_pretrained(local_model_path)
# tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
# text = "Hello, world!"
# input_ids = tokenizer.encode(text, return_tensors="pt")
# outputs = model(input_ids)
#
# print(outputs.last_hidden_state)

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        tem_emb = time_day.transpose(1, 2).unsqueeze(-1)

        return tem_emb

class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2", output_attentions=True, output_hidden_states=True)
        # self.gpt2 = GPT2Model.from_pretrained(local_model_path)

        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U
        #
        # for layer_index, layer in enumerate(self.gpt2.h):
        #     for name, param in layer.named_parameters():
        #         if layer_index < gpt_layers - self.U:
        #             if "ln" in name or "wpe" in name:
        #                 param.requires_grad = True
        #             else:
        #                 param.requires_grad = False
        #         else:
        #             if "mlp" in name:
        #                 param.requires_grad = False
        #             else:
        #                 param.requires_grad = True


        # 冻结位置嵌入参数 (wpe)
        for name, param in self.gpt2.named_parameters():
            if "wpe" in name or "wte" in name:
                param.requires_grad = True

        # 冻结或训练每一层的参数
        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                # 对前 gpt_layers - U 层进行冻结
                if layer_index < gpt_layers - self.U:
                    # 冻结 MLP 层，但保留 LayerNorm 和 Self-Attention 参数进行训练
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                else:
                    # 对最后 U 层，训练所有的参数
                        param.requires_grad = True



    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state


# class EnhancedFeatureFusion(nn.Module):
#     def __init__(self, gpt_dim=768, stgcn_dim=512):
#         super().__init__()
#         self.hidden_dim = 256  # 中间隐层维度
#
#         # GPT特征投影
#         self.gpt_proj = nn.Sequential(
#             nn.Linear(gpt_dim, self.hidden_dim),
#             nn.LayerNorm(self.hidden_dim),
#             nn.ReLU()
#         )
#
#         # STGCN特征投影
#         self.stgcn_proj = nn.Sequential(
#             nn.Linear(stgcn_dim, self.hidden_dim),
#             nn.LayerNorm(self.hidden_dim),
#             nn.ReLU()
#         )
#
#         # 交叉注意力
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.hidden_dim,
#             num_heads=8,
#             dropout=0.1
#         )
#
#         # 最终融合
#         self.final_fusion = nn.Sequential(
#             nn.Linear(2 * self.hidden_dim, gpt_dim),
#             nn.LayerNorm(gpt_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, gpt_feat, stgcn_feat):
#         # 1. 特征投影
#         gpt_feat = self.gpt_proj(gpt_feat)  # [B, N, 256]
#         stgcn_feat = self.stgcn_proj(stgcn_feat)  # [B, N, 256]
#
#         # 2. 交叉注意力
#         # 转换维度顺序以适应MultiheadAttention
#         gpt_feat = gpt_feat.transpose(0, 1)  # [N, B, 256]
#         stgcn_feat = stgcn_feat.transpose(0, 1)  # [N, B, 256]
#
#         attn_output, _ = self.cross_attn(
#             query=gpt_feat,
#             key=stgcn_feat,
#             value=stgcn_feat
#         )
#
#         # 恢复维度顺序
#         attn_output = attn_output.transpose(0, 1)  # [B, N, 256]
#         gpt_feat = gpt_feat.transpose(0, 1)  # [B, N, 256]
#
#         # 3. 特征融合
#         fused = torch.cat([attn_output, gpt_feat], dim=-1)  # [B, N, 512]
#         output = self.final_fusion(fused)  # [B, N, 768]
#
#         return output


class EnhancedFeatureFusion(nn.Module):
    def __init__(self, gpt_dim=768, stgcn_dim=512):
        super().__init__()
        self.hidden_dim = 256

        # 特征降维
        self.gpt_down = nn.Sequential(
            nn.Linear(gpt_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.stgcn_down = nn.Sequential(
            nn.Linear(stgcn_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 简单但有效的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # 特征融合和升维
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, gpt_dim)
        )

        # 残差连接的缩放因子
        self.alpha = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, gpt_feat, stgcn_feat):
        # 特征降维
        gpt_hidden = self.gpt_down(gpt_feat)
        stgcn_hidden = self.stgcn_down(stgcn_feat)

        # 计算注意力权重
        combined = torch.cat([gpt_hidden, stgcn_hidden], dim=-1)
        weights = self.attention(combined)

        # 加权融合
        weighted_feat = weights[..., 0:1] * gpt_hidden + weights[..., 1:2] * stgcn_hidden

        # 特征融合并添加残差连接
        fused_feat = self.fusion(weighted_feat)
        output = self.alpha * fused_feat + (1 - self.alpha) * gpt_feat

        return output




class EnhancedSTLayer(nn.Module):
    def __init__(self, window_size, in_features, out_features, dropout=0.1):
        super().__init__()
        self.spatial_attn = GraphAttentionLayer(
            window_size, in_features, out_features, dropout
        )

        # 添加时序注意力
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=out_features,
            num_heads=8,
            dropout=dropout
        )

        # 添加FFN
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 4, out_features)
        )

        # 添加LayerNorm
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.norm3 = nn.LayerNorm(out_features)

    def forward(self, x, A, att_mask=None):
        """
        x: [batch_size, seq_len, num_nodes, features]
        A: [batch_size, seq_len, num_nodes, num_nodes]
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # 1. 空间注意力
        spatial_out = self.spatial_attn(A, x, att_mask)
        spatial_out = self.norm1(spatial_out + x)

        # 2. 时序注意力
        temp = spatial_out.permute(2, 0, 1, 3)  # [N, B, T, F]
        temp = temp.reshape(num_nodes * batch_size, seq_len, -1)
        temp = temp.permute(1, 0, 2)  # [T, B*N, F]

        temporal_out, _ = self.temporal_attn(temp, temp, temp)
        temporal_out = temporal_out.permute(1, 0, 2)  # [B*N, T, F]
        temporal_out = temporal_out.reshape(num_nodes, batch_size, seq_len, -1)
        temporal_out = temporal_out.permute(1, 2, 0, 3)  # [B, T, N, F]

        out = self.norm2(temporal_out + spatial_out)

        # 3. FFN
        ffn_out = self.ffn(out.view(-1, out.size(-1)))
        ffn_out = ffn_out.view(batch_size, seq_len, num_nodes, -1)

        return self.norm3(ffn_out + out)

class CNN_Generator(nn.Module):
    def __init__(self, STG, window_size, n_pred, in_features, out_features, out_size=2, embedding_dim=64, n_stgcnn=1, n_txpcnn=5, max_node_num = 57, **kwargs):
        super(CNN_Generator, self).__init__()
        self.window_size = window_size
        self.pred_len = n_pred
        self.in_features = in_features
        self.out_features = out_features  # 64
        self.out_size = out_size     # 5
        self.embedding_dim = embedding_dim
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.max_node_num = max_node_num

        self.input_dim = 7



        self.spatial_embedding = nn.Linear(self.in_features, self.embedding_dim)
        #


        self.stgcns = nn.ModuleList()
        self.stgcns.append(STG(self.window_size, self.embedding_dim, out_features, 3))
        for j in range(1, self.n_stgcnn):
            self.stgcns.append(STG(window_size, out_features, out_features, 3))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(window_size, self.pred_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(self.pred_len, self.pred_len, 3, padding=1))

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())
        self.spatial_decoder = nn.Conv2d(self.out_features, self.out_size, 3, 1, 1)


        self.node_embeddings = nn.Parameter(torch.randn(self.max_node_num, config.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.max_node_num, self.in_features, config.dim_out, config.cheb_k, config.embed_dim, config.num_layers)

        # predictor
        self.end_conv = nn.Conv2d(1, self.pred_len * self.out_size, kernel_size=(1, self.embedding_dim), bias=True)


        #  LLM
        gpt_channel = 256
        to_gpt_channel = 768
        time = self.window_size

        self.start_conv = nn.Conv2d(
            self.in_features * self.window_size, to_gpt_channel, kernel_size=(1, 1)
        )
        self.embedding_input = nn.Conv2d(
            self.input_dim * self.window_size, to_gpt_channel, kernel_size=(1, 1)
        )

        self.encoder_togpt = nn.Conv2d(
            self.embedding_dim * self.window_size, to_gpt_channel, kernel_size=(1, 1)
        )
        self.Temb = TemporalEmbedding(time, gpt_channel)

        self.node_emb = nn.Parameter(torch.empty(self.max_node_num, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.feature_fusion = nn.Conv2d(
            gpt_channel * 3 + self.embedding_dim * self.window_size, to_gpt_channel, kernel_size=(1, 1)
        )

        # embedding layer
        self.gpt = PFA(device= "cuda:0", gpt_layers=config.gpt_layers, U=config.U)

        # regression
        self.regression_layer = nn.Conv2d(
            gpt_channel * 3, self.pred_len * self.out_size, kernel_size=(1, 1)
        )
        # 修改特征融合模块
        # self.feature_fusion = EnhancedFeatureFusion(
        #     gpt_dim=768,  # GPT特征维度
        #     stgcn_dim=self.embedding_dim * self.window_size  # STGCN特征维度
        # )

        # 替换原有的STGCN层
        self.st_layers = nn.ModuleList([
            EnhancedSTLayer(
                window_size=window_size,
                in_features=self.embedding_dim if i == 0 else out_features,
                out_features=out_features
            ) for i in range(n_stgcnn)
        ])

    def forward(self, V_obs, A, att_mask=None):
        """
        :param V_obs, shape [batch_Size, n_his, node_num, in_feat]
        :param att_mask, shape [batch, node, seq]
        :return shape [batch_size, n_pred, node_num, inf_feat]
        """
        batch_size, window_size, node_num, in_feature = V_obs.shape
        # input_data = V_obs
        # print(torch.histc(V_obs))

        ################################# noise #############################################
        # noise = torch.randn_like(V_obs) * 1e-5
        # V_obs = V_obs + noise
        # print(f"V_obs max: {V_obs.max()}, min: {V_obs.min()}, mean: {V_obs.mean()}")
        # if torch.all(V_obs == 0):
        #     print("Warning: V_obs is all zeros!")
        # print("V_obs sample values:", V_obs.view(-1)[:10])  # 打印前 10 个值
        # noise = (torch.rand(batch_size, window_size, node_num, node_num, device=A.device))  # 构造随机的邻接矩阵
        # noise = noise * (noise > 0.5)
        # adjacency = A * noise
        adjacency = A

        ################################# AGCN #############################################
        # init_state = self.encoder.init_hidden(V_obs.shape[0])
        # # print(init_state)
        # # print(f"init_state max: {init_state.max()}, min: {init_state.min()}, mean: {init_state.mean()}")
        #
        # output, _ = self.encoder(V_obs, init_state, self.node_embeddings)
        # # print(output)
        #
        # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        #
        # #CNN based predictor
        # output = self.end_conv((output))                         #B, T*C, N, 1  torch.Size([64, 1, 57, 256])
        # output = output.squeeze(-1).reshape(-1, self.pred_len, 2, self.max_node_num)
        # output = output.permute(0, 1, 3, 2)
        # print(output.shape)   #  (64,57,12,64)

        ################################# GTGAN #############################################
        # v = self.spatial_embedding(V_obs.view(-1, in_feature)).view(batch_size, window_size, node_num, -1)
        # # print(v.shape,output.shape)  torch.Size([64, 8, 57, 64])
        #
        #
        # for k in range(self.n_stgcnn):
        #     # v = self.stgcns[k](output, adjacency, att_mask).contiguous()
        #     v = self.stgcns[k](v, adjacency, att_mask).contiguous() # torch.Size([64, 8, 57, 64])
        # stgcn_output = v.permute(0, 1, 3, 2).contiguous()  #  [64, 8, 64, 57]
        # v = self.prelus[0](self.tpcnns[0](stgcn_output))
        # print("v.shape:",v.shape)  # [64, 12, 64, 57]
        #
        # for k in range(1, self.n_txpcnn):
        #     v = self.prelus[k](self.tpcnns[k](v)) + v    # [64, 12, 64, 57]
        # output = self.spatial_decoder(v.permute(0, 2, 1, 3))  #  torch.Size([64, 5, 12, 57])
        # output = output.permute(0, 2, 3, 1)  # torch.Size([64, 12, 57, 5])

        ################################# GPT #############################################
        # input_data = V_obs
        # input_data = input_data.permute(0, 3, 2, 1)  # torch.Size([64, 8, 57, 2])
        # batch_size, _, node_num, _ = input_data.shape
        # input_data = input_data.transpose(1,2).contiguous()
        # input_data = (
        #     input_data.view(batch_size,node_num,-1).transpose(1,2).unsqueeze(-1)
        # )    # 64, 16, 57, 1
        # input_data = self.start_conv(input_data)  # torch.Size([64, 256, 57, 1])
        #
        # tem_emb = self.Temb(V_obs)  # torch.Size([64, 256, 57, 1])
        #
        # node_emb = []
        # node_emb.append(
        #     self.node_emb.unsqueeze(0)
        #     .expand(batch_size, -1, -1)
        #     .transpose(1, 2)
        #     .unsqueeze(-1)
        # )
        #
        # data_st = torch.cat(
        #     [input_data] + [tem_emb] + node_emb, dim=1
        # )            # torch.Size([64, 768, 57, 1])
        # # print("input_data:{}, tem_emb:{}, node_emb:{}".format(input_data.shape, tem_emb.shape, len(node_emb)))
        # #  input_data:torch.Size([64, 256, 57, 1]), tem_emb:torch.Size([64, 256, 57, 1]), node_emb:1
        # data_st = self.feature_fusion(data_st)  # torch.Size([64, 768, 57, 1])
        # data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)  # torch.Size([64, 57, 768])
        # # print(data_st.shape)
        # data_st = self.gpt(data_st)  # # torch.Size([64, 57, 768])
        # data_st = data_st.permute(0, 2, 1).unsqueeze(-1)  # torch.Size([64, 768, 57, 1])
        #
        # output = self.regression_layer(data_st)  # torch.Size([64, 60, 57, 1])
        # output = output.view(batch_size, self.pred_len, self.out_size, node_num).permute(0, 1, 3, 2)  # torch.Size([64, 12, 57, 5])
        #


        #####################  GPT +  AGCN ##########################

        #
        # input_data = V_obs
        # input_data = input_data.permute(0, 3, 2, 1)
        # batch_size, _, node_num, _ = input_data.shape
        # input_data = input_data.transpose(1, 2).contiguous()  # [64,57,2,8]
        # input_data = (
        #     input_data.view(batch_size, node_num, -1).transpose(1, 2).unsqueeze(-1)     # [64, 57, 16]  -> [64, 16, 57]  -> [64,16,57,1]
        # )  # 64, 16, 57, 1
        # input_data = self.start_conv(input_data)
        #
        # tem_emb = self.Temb(V_obs)  # time emb (batch_size, gpt_channel, num_nodes, 1)
        # node_emb = []
        # node_emb.append(
        #     self.node_emb.unsqueeze(0)
        #     .expand(batch_size, -1, -1)
        #     .transpose(1, 2)
        #     .unsqueeze(-1)
        # )
        #
        # init_state = self.encoder.init_hidden(V_obs.shape[0])
        # # print(init_state)
        # # print(f"init_state max: {init_state.max()}, min: {init_state.min()}, mean: {init_state.mean()}")
        #
        # node_emb, _ = self.encoder(V_obs, init_state, self.node_embeddings)
        # # print(output)
        #
        # node_emb = node_emb[:, -1:, :, :]                                   #B, 1, N, hidden torch.Size([64, 1, 57, 256])
        # node_emb = node_emb.permute(0, 3, 2, 1)
        #
        # data_st = torch.cat(
        #     [input_data] + [tem_emb] + [node_emb], dim=1
        # )
        #
        # # data_st = torch.cat(
        # #     [input_data] + [tem_emb] + node_emb, dim=1
        # # )
        # # print("input_data:{}, tem_emb:{}, node_emb:{}".format(input_data.shape, tem_emb.shape, len(node_emb)))
        # #  input_data:torch.Size([64, 256, 57, 1]), tem_emb:torch.Size([64, 256, 57, 1]), node_emb:1
        # data_st = self.feature_fusion(data_st)
        # data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        # # print(data_st.shape)
        # data_st = self.gpt(data_st)
        # data_st = data_st.permute(0, 2, 1).unsqueeze(-1)  # torch.Size([64, 768, 57, 1])
        #
        # output = self.regression_layer(data_st)  # torch.Size([64, 24, 57, 1])
        # output = output.view(batch_size, self.pred_len, self.out_size, node_num).permute(0, 1, 3,2)  # torch.Size([64, 12, 2, 57])




        ################################# GTGAN+ GPT #############################################

        # # tem_emb = self.Temb(V_obs)  # time emb (batch_size, gpt_channel, num_nodes, 1)
        # data_st = torch.cat(
        #     [input_data] + [stgcn_output], dim=1
        # )
        # # print("input_data:{}, tem_emb:{}, node_emb:{}".format(input_data.shape, tem_emb.shape, len(node_emb)))
        # #  input_data:torch.Size([64, 256, 57, 1]), tem_emb:torch.Size([64, 256, 57, 1]), node_emb:1
        # data_st = self.feature_fusion(data_st)
        # data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        # # print(data_st.shape)
        # data_st = self.gpt(data_st)
        # data_st = data_st.permute(0, 2, 1).unsqueeze(-1)  # torch.Size([64, 768, 57, 1])
        #
        # output = self.regression_layer(data_st)  # torch.Size([64, 24, 57, 1])
        # output = output.view(batch_size, self.pred_len, self.out_size, node_num).permute(0, 1, 3, 2)  # torch.Size([64, 12, , 57])

        ################################# GTGAN+ GPT ---new fusion  #############################################


        input_data = V_obs
        input_data = input_data.permute(0, 3, 2, 1)  # torch.Size([64, 8, 57, 2])
        input_data = input_data.transpose(1,2).contiguous()
        input_data = (
            input_data.view(batch_size,node_num,-1).transpose(1,2).unsqueeze(-1)
        )    # 64, 16, 57, 1
        input_data = self.start_conv(input_data)  # torch.Size([64, 256, 57, 1])

        # 1.2 GPT处理
        gpt_features = input_data.squeeze(-1).transpose(1, 2)  # [B, N, 768]
        gpt_features = self.gpt(gpt_features)  # 使用GPT进行特征提取 [B, N, 768]


        # 2. 时空特征提取
        # 2.1 空间嵌入
        v = self.spatial_embedding(V_obs.view(-1, in_feature))  # [(B*T*N), embed_dim]
        v = v.view(batch_size, window_size, node_num, -1)  # [B, T, N, embed_dim]

        # 2.2 增强的时空特征提取
        for layer in self.st_layers:
            v = layer(v, A, att_mask)  # [B, T, N, embed_dim]

        # 2.3 准备STGAT特征用于融合
        stgcn_output = v.permute(0, 2, 1, 3)  # [B, N, T, embed_dim]
        stgcn_output = stgcn_output.reshape(batch_size, node_num, -1)  # [B, N, T*embed_dim]
        # print("gpt_features, stgcn_output",gpt_features.shape, stgcn_output.shape)
        stgcn_output = (
            stgcn_output.transpose(1,2).unsqueeze(-1)
        )    # 64, 512, 57, 1
        gpt_features = gpt_features.transpose(1,2).unsqueeze(-1)
        # print("gpt_features, stgcn_output",gpt_features.shape, stgcn_output.shape)
        data_st = torch.cat(
            [gpt_features] + [stgcn_output], dim=1
        )
        # print("input_data:{}, tem_emb:{}, node_emb:{}".format(input_data.shape, tem_emb.shape, len(node_emb)))
        #  input_data:torch.Size([64, 256, 57, 1]), tem_emb:torch.Size([64, 256, 57, 1]), node_emb:1
        fused_features = self.feature_fusion(data_st)

        # 4. 使用改进的特征融合
        # fused_features = self.feature_fusion(gpt_features, stgcn_output)
        # fused_features = fused_features.transpose(1, 2).unsqueeze(-1)  # [B, 768, N, 1]

        # 5. 解码预测轨迹

        output = self.regression_layer(fused_features)
        output = output.view(batch_size, self.pred_len, self.out_size, node_num).permute(0, 1, 3, 2)

        ################################# STGCNN + GPT -------version2   #############################################
        #
        # input_data = V_obs
        # input_data = input_data.permute(0, 3, 2, 1)  # torch.Size([64, 8, 57, 2])
        # input_data = input_data.transpose(1,2).contiguous()
        # input_data = (
        #     input_data.view(batch_size,node_num,-1).transpose(1,2).unsqueeze(-1)
        # )    # 64, 16, 57, 1
        # input_data = self.start_conv(input_data)  # torch.Size([64, 768, 57, 1])
        #
        #
        # data_st = input_data.permute(0, 2, 1, 3).squeeze(-1)
        # # print(data_st.shape)
        # data_st = self.gpt(data_st)   # torch.Size([64, 57, 768])
        # data_st = data_st.permute(0, 2, 1).unsqueeze(-1)  # torch.Size([64, 768, 57, 1])
        #
        #
        #
        #
        # v = self.spatial_embedding(V_obs.view(-1, in_feature)).view(batch_size, window_size, node_num, -1)
        # # print(v.shape,output.shape)  torch.Size([64, 8, 57, 64])
        #
        #
        # for k in range(self.n_stgcnn):
        #     # v = self.stgcns[k](output, adjacency, att_mask).contiguous()
        #     v = self.stgcns[k](v, adjacency, att_mask).contiguous() # torch.Size([64, 8, 57, 64])
        # stgcn_output = v.permute(0, 2, 1, 3).contiguous()  #  [64, 57, 8, 64]
        #
        #
        # stgcn_output = (
        #     stgcn_output.view(batch_size,node_num,-1).transpose(1,2).unsqueeze(-1)
        # )    # 64, 512, 57, 1
        # # stgcn_output = self.encoder_togpt(stgcn_output)  # 64, 512, 57
        #
        # # tem_emb = self.Temb(V_obs)  # time emb (batch_size, gpt_channel, num_nodes, 1)
        # data_st = torch.cat(
        #     [data_st] + [stgcn_output], dim=1
        # )   # torch.Size([64, 1280, 57, 1])
        # # print("input_data:{}, tem_emb:{}, node_emb:{}".format(input_data.shape, tem_emb.shape, len(node_emb)))
        # #  input_data:torch.Size([64, 256, 57, 1]), tem_emb:torch.Size([64, 256, 57, 1]), node_emb:1
        # data_st = self.feature_fusion(data_st)
        #
        # output = self.regression_layer(data_st)  # torch.Size([64, 24, 57, 1])
        #
        # output = output.view(batch_size, self.pred_len, self.out_size, node_num).permute(0, 1, 3, 2)  # torch.Size([64, 12, , 57])

        ################################# STGCNN + GPT -------version3   #############################################

        # input_data = V_obs
        # input_data = input_data.permute(0, 3, 2, 1)  # torch.Size([64, 8, 57, 2])
        # input_data = input_data.transpose(1,2).contiguous()
        # input_data = (
        #     input_data.view(batch_size,node_num,-1).transpose(1,2).unsqueeze(-1)
        # )    # 64, 16, 57, 1
        # input_data = self.start_conv(input_data)  # torch.Size([64, 768, 57, 1])
        #
        #
        # data_st = input_data.permute(0, 2, 1, 3).squeeze(-1)
        # # print(data_st.shape)
        # data_st = self.gpt(data_st)   # torch.Size([64, 57, 768])
        # data_st = data_st.permute(0, 2, 1).unsqueeze(-1)  # torch.Size([64, 768, 57, 1])
        #
        #
        #
        #
        # v = self.spatial_embedding(V_obs.view(-1, in_feature)).view(batch_size, window_size, node_num, -1)
        # # print(v.shape,output.shape)  torch.Size([64, 8, 57, 64])
        #
        #
        # for k in range(self.n_stgcnn):
        #     # v = self.stgcns[k](output, adjacency, att_mask).contiguous()
        #     v = self.stgcns[k](v, adjacency, att_mask).contiguous() # torch.Size([64, 8, 57, 64])
        # stgcn_output = v.permute(0, 2, 1, 3).contiguous()  #  [64, 57, 8, 64]
        #
        #
        # stgcn_output = (
        #     stgcn_output.view(batch_size,node_num,-1).transpose(1,2).unsqueeze(-1)
        # )    # 64, 512, 57, 1
        # # stgcn_output = self.encoder_togpt(stgcn_output)  # 64, 512, 57
        #
        # # tem_emb = self.Temb(V_obs)  # time emb (batch_size, gpt_channel, num_nodes, 1)
        # data_st = torch.cat(
        #     [data_st] + [stgcn_output], dim=1
        # )   # torch.Size([64, 1280, 57, 1])
        # # print("input_data:{}, tem_emb:{}, node_emb:{}".format(input_data.shape, tem_emb.shape, len(node_emb)))
        # #  input_data:torch.Size([64, 256, 57, 1]), tem_emb:torch.Size([64, 256, 57, 1]), node_emb:1
        # data_st = self.feature_fusion(data_st)   # torch.Size([64, 768, 57, 1])
        #
        # # output = self.regression_layer(data_st)  # torch.Size([64, 24, 57, 1])
        # v = self.prelus[0](self.tpcnns[0](data_st))
        # print("v.shape:",v.shape)  # [64, 12, 64, 57]
        #
        # for k in range(1, self.n_txpcnn):
        #     v = self.prelus[k](self.tpcnns[k](v)) + v    # [64, 12, 64, 57]
        # output = self.spatial_decoder(v.permute(0, 2, 1, 3))  #  torch.Size([64, 5, 12, 57])
        #
        # output = output.view(batch_size, self.pred_len, self.out_size, node_num).permute(0, 1, 3, 2)  # torch.Size([64, 12, , 57])

        ################################# GMM  #############################################



        # GMM(V_obs)

        # role_labels_tensor = role_labels_tensor.to(device)
        # role_probs_tensor = role_probs_tensor.to(device)
        # # 将角色标签和角色概率分布添加到输入数据中
        # enhanced_input_data = torch.cat((V_obs, role_labels_tensor, role_probs_tensor), dim=-1)
        # # print(enhanced_input_data.shape)


        # batch_size, _, node_num, _ = V_obs.shape
        # input_data = V_obs.transpose(1,2).contiguous()
        # input_data = (
        #     V_obs.view(batch_size,node_num,-1).transpose(1,2).unsqueeze(-1)
        # )    # 64, 56, 57, 1
        # input_data = self.embedding_input(input_data)  # torch.Size([64, 256, 57, 1])
        #
        # tem_emb = self.Temb(V_obs)  # torch.Size([64, 256, 57, 1])
        #
        # node_emb = []
        # node_emb.append(
        #     self.node_emb.unsqueeze(0)
        #     .expand(batch_size, -1, -1)
        #     .transpose(1, 2)
        #     .unsqueeze(-1)
        # )
        #
        # data_st = torch.cat(
        #     [input_data] + [tem_emb] + node_emb, dim=1
        # )            # torch.Size([64, 768, 57, 1])
        # # print("input_data:{}, tem_emb:{}, node_emb:{}".format(input_data.shape, tem_emb.shape, len(node_emb)))
        # #  input_data:torch.Size([64, 256, 57, 1]), tem_emb:torch.Size([64, 256, 57, 1]), node_emb:1
        # data_st = self.feature_fusion(data_st)  # torch.Size([64, 768, 57, 1])
        # data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)  # torch.Size([64, 57, 768])
        # # print(data_st.shape)
        # data_st = self.gpt(data_st)  # # torch.Size([64, 57, 768])
        # data_st = data_st.permute(0, 2, 1).unsqueeze(-1)  # torch.Size([64, 768, 57, 1])
        #
        # output = self.regression_layer(data_st)  # torch.Size([64, 60, 57, 1])
        # output = output.view(batch_size, self.pred_len, self.out_size, node_num).permute(0, 1, 3, 2)  # torch.Size([64, 12, 57, 5])
        #



        return output



def preprocess_data(input_data):
    # 将数据从 GPU 移到 CPU，并转换为 NumPy 数组
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.cpu().numpy()

    # 将数据展平到二维平面上，以便 GMM 处理
    # 假设 input_data 维度为 [batch_size, time_step, human_id, location]
    batch_size, time_step, human_id, _ = input_data.shape

    # 提取位置特征
    location_data = input_data.reshape(batch_size * time_step * human_id, -1)
    return location_data

def GMM(V_obs):
    _, window_size, node_num, _ = V_obs.shape
    location_data = preprocess_data(V_obs)

    # 定义和训练 GMM 模型
    n_components = 4  # 群体角色的数量，可以根据需要调整
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(location_data)

    # 为每个样本预测角色标签和概率分布
    role_labels = gmm.predict(location_data)  # 每个位置的角色标签
    role_probs = gmm.predict_proba(location_data)  # 每个位置的角色概率分布

    # 结果展示
    # print("Role Labels:", role_labels[:10])  # 角色标签
    # print("Role Probabilities:", role_probs[:10])  # 角色概率分布

    # 假设 input_data.shape = [batch_size, time_step, human_id, location]
    # 将 GMM 的输出转换为张量形式，以便与原始输入数据整合
    role_labels_tensor = torch.tensor(role_labels, dtype=torch.long).view(V_obs.shape[0], V_obs.shape[1],
                                                                          V_obs.shape[2], 1)
    role_probs_tensor = torch.tensor(role_probs, dtype=torch.float32).view(V_obs.shape[0], V_obs.shape[1],
                                                                           V_obs.shape[2], -1)

    # 选择一个批次进行可视化
    batch_index = 0
    current_trajectory_data = V_obs[batch_index].cpu().numpy()  # [time_steps, num_pedestrians, location_dim]
    current_role_labels = role_labels_tensor[batch_index].cpu().numpy()  # [time_steps, num_pedestrians, 1]
    current_role_probs = role_probs_tensor[batch_index].cpu().numpy()  # [time_steps, num_pedestrians, n_components]
    print("Trajectory Data Shape:", current_trajectory_data.shape)
    print("Role Labels Shape:", current_role_labels.shape)
    print("Role Probabilities Shape:", current_role_probs.shape)

    # 颜色映射
    colors = ['r', 'g', 'b', 'y']
    labels = [f'Role {i}' for i in range(n_components)]

    # 创建动画
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # 初始化散点图和文本
    scatter = ax.scatter([], [], s=100)
    texts = [ax.text(0, 0, "", ha='center', va='center', fontsize=8) for _ in range(node_num)]

    # 初始化函数
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        for text in texts:
            text.set_text("")
        return scatter, texts

    # 更新函数
    def update(frame):
        x = current_trajectory_data[frame, :, 0]
        y = current_trajectory_data[frame, :, 1]

        # 设置颜色和大小
        role_indices = current_role_labels[frame, :, 0]  # 当前时间步的角色标签
        sizes = 300 * current_role_probs[frame, np.arange(node_num), role_indices]  # 角色概率决定大小
        colors_mapped = [colors[role] for role in role_indices]

        # 更新散点图
        scatter.set_offsets(np.c_[x, y])
        scatter.set_sizes(sizes)
        scatter.set_color(colors_mapped)

        # 更新文本标签以显示角色和概率
        for i, text in enumerate(texts):
            text.set_position((x[i], y[i]))
            prob = current_role_probs[frame, i, role_indices[i]]
            text.set_text(f"{labels[role_indices[i]]}\nProb: {prob:.2f}")

        return scatter, texts

    # 创建动画
    ani = FuncAnimation(fig, update, frames=window_size, init_func=init, blit=True, repeat=False)
    plt.show()

class CNN_GAT_Generator(CNN_Generator):
    def __init__(self, **kwargs):
        super(CNN_GAT_Generator, self).__init__(STGAT, **kwargs)


class CNN_GCN_Generator(CNN_Generator):
    def __init__(self, **kwargs):
        super(CNN_GCN_Generator, self).__init__(STGCN, **kwargs)

    # def forward(self, V_obs, A, att_mask=None):
    #     batch_size, window_size, node_num, in_feature = V_obs.shape
    #
    #     noise = (torch.rand(batch_size, window_size, node_num, node_num, device=A.device))  # 构造随机的邻接矩阵
    #     noise = noise * (noise > 0.5)
    #     adjacency = A * noise
    #
    #     eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num,
    #                                                                           node_num)
    #     adjacency = adjacency + eye
    #     diag = adjacency.sum(dim=-1, keepdim=True).pow(-0.5).expand(adjacency.size()) * eye
    #     adjacency = diag.matmul(adjacency).matmul(diag)
    #
    #     v = self.spatial_embedding(V_obs.view(-1, in_feature)).view(batch_size, window_size, node_num, -1)
    #     for k in range(self.n_stgcnn):
    #         v = self.stgcns[k](v, adjacency, att_mask)
    #     stgcn_output = v.permute(0, 2, 1, 3).contiguous()  # [batch, n_his, in_feature, node_num]
    #     v = self.prelus[0](self.tpcnns[0](stgcn_output))  # 预测下12帧
    #     for k in range(1, self.n_txpcnn):
    #         v = self.prelus[k](self.tpcnns[k](v)) + v
    #     output = self.spatial_decoder(v.permute(0, 2, 1, 3))  # [batch, in_feature, n_pred, node_num]
    #     output = output.permute(0, 2, 3, 1)
    #     return output
