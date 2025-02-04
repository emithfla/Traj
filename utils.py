import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_adaptive_weight(weight_matrix, epoch):
    plt.figure(figsize=(8, 6))
    sns.heatmap(weight_matrix.cpu().detach().numpy(), cmap='viridis', annot=False)
    plt.title(f'Adaptive Weight Matrix at Epoch {epoch}')
    plt.show()

def seq_to_nodes(seq_):
    """
    seq_: shape (batch, node, feat, seq)
    """
    batch_size = seq_.shape[0]
    max_nodes = seq_.shape[1]
    seq_len = seq_.shape[3]

    V = torch.zeros((batch_size, seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, :, s]
        for h in range(step_.shape[1]):
            V[:, s, h, :] = step_[:, h, :]
    return V

def nodes_rel_to_node_abs(nodes, init_node):
    """
    nodes: shape (batch, seq, node, feat)
    init_node: shape (batch, node, feat)
    """
    init_node = init_node.unsqueeze(1).repeat(1,nodes.shape[1],1,1)
    cums = torch.cumsum(nodes, axis=1)
    nodes_ = init_node + cums
    return nodes_

def bivariate_loss(V_pred, V_trgt):
    """
    V_pred: tensor, shape (1, seq, node, feat)
    """
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    # V_pred = V_pred.squeeze(0)
    # V_trgt = V_trgt.squeeze(0)

    normx = V_trgt[..., 0] - V_pred[..., 0]
    normy = V_trgt[..., 1] - V_pred[..., 1]

    sx = torch.exp(V_pred[..., 2])  # sx
    sy = torch.exp(V_pred[..., 3])  # sy
    corr = torch.tanh(V_pred[..., 4])  # corr
    # print('sx:', sx.mean(), 'sy:', sy.mean(), 'corr:', corr.mean())

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho)) + 1e-10

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result

def get_pos_loss_k(pred_, gt_, loss_mask_, mode='avg',mask_loss=True):
    """
    :param loss_mask: Tensor of shape (batch, node, seq)
    """
    # batch, seq_len, node_num, _ = pred_.shape
    if mask_loss:
        loss = (loss_mask_.permute(0,2,1).unsqueeze(3)*(pred_-gt_)**2)
    else:
        loss = (pred_-gt_)**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'avg':
        return torch.sum(loss) / torch.sum(loss_mask_.data)
    elif mode == 'raw':
        return loss.sum(dim=3).sum(dim=2)