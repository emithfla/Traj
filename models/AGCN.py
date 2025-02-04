import torch
import torch.nn.functional as F
import torch.nn as nn




class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in,  dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        # print(f"x.shape: {x.shape}, expected node_num: {self.node_num}, expected input_dim: {self.input_dim}")

        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        # print(f"x max: {x.max()}, min: {x.min()}, mean: {x.mean()}")
        # print(
        #     f"node_embeddings111111111 max: {node_embeddings.max()}, min: {node_embeddings.min()}, mean: {node_embeddings.mean()}")

        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        # print(self.node_num,self.num_layers,self.input_dim)
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                # print(f"state max: {state.max()}, min: {state.min()}, mean: {state.mean()}")
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                # print(f"state max: {state.max()}, min: {state.min()}, mean: {state.mean()}")
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # print(f"current_inputs max: {current_inputs.max()}, min: {current_inputs.min()}, mean: {current_inputs.mean()}")
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        # print(f"h max: {h.max()}, min: {h.min()}, mean: {h.mean()}")
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        # print("dim_in:{},dim_out:{}".format(dim_in,dim_out))
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        # 初始化权重和偏置
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]

        torch.autograd.set_detect_anomaly(True)




        node_num = node_embeddings.shape[0]
        # print(
            # f"node_embeddings max: {node_embeddings.max()}, min: {node_embeddings.min()}, mean: {node_embeddings.mean()}")

        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # print(f"supports max: {supports.max()}, min: {supports.min()}, mean: {supports.mean()}")
        # 避免全零输入
        # supports = F.softmax(F.leaky_relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # print(f"supports2 max: {supports.max()}, min: {supports.min()}, mean: {supports.mean()}")
        # print(
            # f"weights_pool max: {self.weights_pool.max()}, min: {self.weights_pool.min()}, mean: {self.weights_pool.mean()}")



        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out

        # print(f"weights max: {weights.max()}, min: {weights.min()}, mean: {weights.mean()}")

        bias = torch.matmul(node_embeddings, self.bias_pool) #N, dim_out
        # print(f"bias max: {bias.max()}, min: {bias.min()}, mean: {bias.mean()}")
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in


        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        # print("node_embeddings:{}, weights_pool:{}, x_g:{}, bias:{}".format(node_embeddings.shape,self.weights_pool.shape,x_g.shape, bias.shape))
        # print(f"x_gconv max: {x_gconv.max()}, min: {x_gconv.min()}, mean: {x_gconv.mean()}")
        # weights = torch.clamp(weights, min=-1, max=1)
        # print(f"weightsclamp max: {weights.max()}, min: {weights.min()}, mean: {weights.mean()}")
        return x_gconv