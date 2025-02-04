import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import numpy as np


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.num_nodes = num_nodes

    def forward(self, X, A):
        # X 的维度为 (batch_size, node_num, in_features)
        # A 的维度为 (batch_size, node_num, node_num)

        batch_size = X.size(0)
        node_num = X.size(1)

        # 创建单位矩阵 I，并扩展到 batch_size 的大小
        I = torch.eye(node_num).to(A.device)  # I 的大小为 (node_num, node_num)
        I = I.unsqueeze(0).repeat(batch_size, 1, 1)  # 扩展 I 为 (batch_size, node_num, node_num)

        # 计算 A_hat = A + I
        A_hat = A + I  # A 和 I 的维度匹配

        # 计算 D_hat
        D_hat = torch.diag_embed(torch.sum(A_hat, dim=-1))  # D_hat 的维度为 (batch_size, node_num, node_num)

        # 归一化邻接矩阵
        D_hat_inv_sqrt = torch.inverse(torch.sqrt(D_hat))
        A_normalized = torch.bmm(torch.bmm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)  # A_normalized 的维度为 (batch_size, node_num, node_num)

        # 图卷积操作：A_normalized * X * W
        output = torch.bmm(A_normalized, X)  # A_normalized * X，得到的维度为 (batch_size, node_num, in_features)
        output = torch.matmul(output, self.weight)  # (A_normalized * X) * W，得到的维度为 (batch_size, node_num, out_features)

        return output

class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers = 6, U = 1):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state

class AdaptiveGCN(nn.Module):
    def __init__(self, num_nodes, obs_seq_len, in_features, out_features, pred_seq_len, U):
        super(AdaptiveGCN, self).__init__()
        # 动态学习的邻接矩阵
        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes))  # 共享邻接矩阵
        self.gcn1 = GCNLayer(in_features, 64, num_nodes)
        self.gcn2 = GCNLayer(64, out_features, num_nodes)
        self.obs_seq_len = obs_seq_len
        self.in_features = in_features
        self.pred_seq_len = pred_seq_len
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.U = U
        # embedding layer
        self.gpt = PFA(device='cuda:0', gpt_layers=6, U=self.U)
        # 用于生成未来序列的全连接层
        self.fc = nn.Linear(out_features, out_features)




        gpt_channel = 64
        to_gpt_channel = 768

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        self.Temb = nn.Parameter(torch.empty(self.obs_seq_len, gpt_channel))
        self.regression_layer = nn.Conv2d(
            gpt_channel * 12, self.pred_seq_len, kernel_size=(1, 1)
        )

        self.start_conv = nn.Conv2d(
            self.in_features * self.obs_seq_len, gpt_channel, kernel_size=(1, 1)
        )
        # self.Temb = TemporalEmbedding(time, gpt_channel)
        # self.feature_fusion = nn.Conv2d(
        #     gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1)
        # )

    def forward(self, X):
        batch_size, obs_seq_len, node_num, in_features = X.shape
        # 将输入 X 从 (batch_size, obs_seq_len, node_num, in_features)
        # 变换为 (batch_size * obs_seq_len, node_num, in_features)
        X_reshaped = X.view(batch_size * obs_seq_len, node_num, in_features)

        # 创建邻接矩阵，并适应新的 batch 维度
        A_normalized = F.softmax(self.A, dim=-1).unsqueeze(0).repeat(batch_size * obs_seq_len, 1, 1)

        # 通过两层图卷积
        X_out = self.gcn1(X_reshaped, A_normalized)
        X_out = F.relu(X_out)
        X_out = self.gcn2(X_out, A_normalized)


        # 将输出重新变形回 (batch_size, obs_seq_len, node_num, out_features)
        X_out = X_out.view(batch_size, obs_seq_len, node_num, self.out_features)




        # tem_emb = self.Temb(g_in)

        tem_emb = []
        tem_emb.append(
            self.Temb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )


        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )

        # node_emb, tem_emb = np.array(node_emb,tem_emb)
        # print(node_emb.shape,tem_emb.shape)


        # input_data = X_out.transpose(1, 2).contiguous()
        # input_data = (
        #     input_data.view(batch_size, node_num, -1).transpose(1, 2).unsqueeze(-1)
        # )
        # input_data = self.start_conv(input_data)

        # print(type(tem_emb),type(X_out), type(node_emb))


        input_data = X_out.transpose(1,2).contiguous()
        input_data = (
            input_data.view(batch_size, node_num, -1).transpose(1, 2).unsqueeze(-1)
        )
        input_data = self.start_conv(input_data)

        print(type(tem_emb),type(X_out), type(node_emb))
        data_st = torch.cat(
            [input_data] + tem_emb + node_emb, dim=1
        )


        data_st = self.feature_fusion(data_st)

        data_st =  data_st.permute(0, 2, 1, 3).squeeze(-1)
        data_st = self.gpt(data_st)
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)

        prediction = self.regression_layer(data_st)
        return prediction


