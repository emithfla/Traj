import torch
import torch.distributions.multivariate_normal as torchdist
import math
from scipy.stats import gaussian_kde
# from utils import to_theta
# from sklearn.mixture import GaussianMixture
from scipy.special import erf
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




# def interaction_regularization_loss(predictions, interaction_matrix, threshold=0.5, lambda_low=0.1, lambda_high=1.0):
#     """
#     基于时间步的社会交互矩阵的正则化损失函数，支持批处理。
#     Args:
#         predictions: Tensor of shape (batch_size, N, T, 5)，批处理的行人预测轨迹，其中前两维为 x, y 坐标。
#         interaction_matrix: Tensor of shape (batch_size, T, N, N)，时间步的社会交互矩阵。
#         threshold: float，高交互强度的阈值。
#         lambda_low: float，低交互行人对的惩罚权重。
#         lambda_high: float，高交互行人对的奖励权重。
#
#     Returns:
#         regularization_loss: float，正则化损失值。
#     """
#     batch_size, T, N, _ = predictions.size()
#     regularization_loss = 0.0
#     # print("predictions and  interaction_matrix", predictions.shape, interaction_matrix.shape)
#
#
#     for b in range(batch_size):  # 遍历每个批次
#         for t in range(T):  # 遍历每个时间步
#             for i in range(N):
#                 for j in range(N):
#                     if i != j:  # 只考虑不同行人之间的交互
#                         # 提取预测的 x, y 坐标
#                         pred_i = predictions[b, t, i, :2]  # (2,)
#                         pred_j = predictions[b, t, j, :2]  # (2,)
#
#                         # 计算两行人之间的预测轨迹的欧几里得距离
#                         pred_dist = torch.norm(pred_i - pred_j)  # 单个时间步的欧几里得距离
#
#                         # 提取 interaction_matrix 的值
#                         interaction_value = interaction_matrix[b, t, i, j]
#
#                         # 根据交互矩阵的值判断高交互或低交互
#                         if interaction_value > threshold:
#                             # 高交互：鼓励轨迹更接近
#                             regularization_loss += lambda_high * pred_dist * interaction_value
#                         else:
#                             # 低交互：惩罚不必要的相似性
#                             regularization_loss += lambda_low * pred_dist * (1 - interaction_value)
#
#     return regularization_loss / batch_size  # 对批次归一化

# def interaction_regularization_loss(predictions, interaction_matrix, threshold=0.7, lambda_low=0.1, lambda_high=1.0):
#     """
#     基于时间步的社会交互矩阵的正则化损失函数，支持批处理。
#         predictions: Tensor of shape (batch_size, T, N, 5)，行人预测轨迹，其中前两维为 x, y 坐标。
#         interaction_matrix: Tensor of shape (batch_size, T, N, N)，社会交互矩阵。
#         threshold: float，高交互强度的阈值。
#         lambda_dist: float，距离损失的权重。
#         lambda_angle: float，方向一致性损失的权重。
#
#     Returns:
#         regularization_loss: float，正则化损失值。
#     """
#     epsilon = 1e-6
#     batch_size, T, N, _ = predictions.size()  # 从 predictions 提取形状
#
#     # 提取 x, y 坐标 (batch_size, T, N, 2)
#     coords = predictions[..., :2]
#
#     # 计算行人之间的欧几里得距离 (batch_size, T, N, N)
#     # 使用广播方式计算所有行人之间的距离
#     diff = coords.unsqueeze(3) - coords.unsqueeze(2)  # (batch_size, T, N, N, 2)
#     dist = torch.norm(diff, dim=-1) + epsilon  # 加上 epsilon 防止除以零，(batch_size, T, N, N)
#
#     # 获取高交互和低交互掩码
#     high_interaction_mask = interaction_matrix > threshold  # (batch_size, T, N, N)
#     low_interaction_mask = ~high_interaction_mask  # (batch_size, T, N, N)
#
#     # 对交互矩阵进行归一化处理，确保交互值在 [0, 1] 之间
#     # interaction_matrix = interaction_matrix / (interaction_matrix.max() + epsilon)
#
#     # 高交互损失：鼓励高交互行人对之间的距离更接近
#     high_interaction_loss = lambda_high * (
#                 dist * interaction_matrix * high_interaction_mask).sum() / high_interaction_mask.sum().clamp(min=1)
#
#     # 低交互损失：惩罚不必要的接近
#     low_interaction_loss = lambda_low * (
#                 dist * (1 - interaction_matrix) * low_interaction_mask).sum() / low_interaction_mask.sum().clamp(min=1)
#
#     # 合并损失并对 batch_size 归一化
#     regularization_loss = (high_interaction_loss + low_interaction_loss) / batch_size
#
#     return regularization_loss

def interaction_regularization_loss(predictions, interaction_matrix, threshold=0.5, lambda_low=0.1, lambda_high=1.0, lambda_direction=0.1):
    """
    改进的基于时间步的社会交互矩阵的正则化损失函数，支持批处理。
    Args:
        predictions: Tensor of shape (batch_size, T, N, 2)，批处理的行人预测轨迹，其中前两维为 x, y 坐标。
        interaction_matrix: Tensor of shape (batch_size, T, N, N)，社会交互矩阵。
        threshold: float，高交互强度的阈值。
        lambda_low: float，低交互行人对的惩罚权重。
        lambda_high: float，高交互行人对的奖励权重。
        lambda_direction: float，方向一致性损失的权重。
        epsilon: float，小数值，用于防止除零问题。

    Returns:
        regularization_loss: float，正则化损失值。
    """
    epsilon = 1e-6
    # 输入验证
    assert len(predictions.shape) == 4, "predictions should be 4-dimensional"
    assert len(interaction_matrix.shape) == 4, "interaction_matrix should be 4-dimensional"
    assert predictions.size(0) == interaction_matrix.size(0), "batch_size mismatch"
    assert predictions.size(1) == interaction_matrix.size(1), "sequence length mismatch"
    assert predictions.size(2) == interaction_matrix.size(2), "number of pedestrians mismatch"

    # 获取张量维度
    batch_size, T, N, dim = predictions.size()

    # 提取坐标 (batch_size, T, N, 2)
    coords = predictions[..., :2]

    # 创建对角线掩码 (用于排除自身交互)
    diagonal_mask = ~torch.eye(N, dtype=torch.bool, device=predictions.device)
    diagonal_mask = diagonal_mask.unsqueeze(0).unsqueeze(0)  # 扩展到 (1, 1, N, N)

    # 计算行人间距离
    diff = coords.unsqueeze(3) - coords.unsqueeze(2)  # (batch_size, T, N, N, 2)
    dist = torch.norm(diff, dim=-1) + epsilon  # (batch_size, T, N, N)

    # 创建交互掩码
    high_interaction_mask = (interaction_matrix > threshold) & diagonal_mask
    low_interaction_mask = (~high_interaction_mask) & diagonal_mask

    # 计算高交互损失 (鼓励高交互行人靠近)
    high_interaction_loss = lambda_high * (
            dist * interaction_matrix * high_interaction_mask
    ).sum() / high_interaction_mask.sum().clamp(min=1)

    # 计算低交互损失 (避免不必要的接近)
    low_interaction_loss = lambda_low * (
            dist * (1 - interaction_matrix) * low_interaction_mask
    ).sum() / low_interaction_mask.sum().clamp(min=1)

    # 计算速度和方向一致性
    velocities = coords[:, 1:] - coords[:, :-1]  # (batch_size, T-1, N, 2)
    velocities_a = velocities.unsqueeze(3)  # (batch_size, T-1, N, 1, 2)
    velocities_b = velocities.unsqueeze(2)  # (batch_size, T-1, 1, N, 2)

    # 计算速度向量的点积
    dot_product = torch.sum(velocities_a * velocities_b, dim=-1)  # (batch_size, T-1, N, N)

    # 计算速度向量的范数
    norm_a = torch.norm(velocities_a, dim=-1)  # (batch_size, T-1, N, 1)
    norm_b = torch.norm(velocities_b, dim=-1)  # (batch_size, T-1, 1, N)

    # 计算余弦相似度 (考虑数值稳定性)
    norm_product = norm_a * norm_b + epsilon
    valid_norms = norm_product > epsilon
    cos_sim = torch.zeros_like(dot_product)
    cos_sim[valid_norms] = dot_product[valid_norms] / norm_product[valid_norms]

    # 调整高交互掩码以匹配速度时间步
    high_interaction_mask_vel = high_interaction_mask[:, 1:]

    # 计算方向一致性损失
    high_interaction_direction_loss = lambda_direction * (
            (1 - cos_sim) * high_interaction_mask_vel
    ).sum() / high_interaction_mask_vel.sum().clamp(min=1)

    # 合并所有损失项并对batch size归一化
    regularization_loss = (
                                  high_interaction_loss +
                                  low_interaction_loss +
                                  high_interaction_direction_loss
                          ) / batch_size

    return regularization_loss

def calculate_velocity(seq_):
    """
    通过绝对位置计算速度（相对位置变化率）。
    Args:
        seq_: Tensor of shape (batch_size, N, 2, T)，绝对位置。
              其中 batch_size 是批量大小，N 是行人数，2 是 (x, y) 坐标，T 是时间步数。
    Returns:
        seq_rel: Tensor of shape (batch_size, N, 2, T-1)，速度。
    """
    # 在时间维度上计算相邻位置的差分
    seq_rel = seq_[..., 1:] - seq_[..., :-1]
    return seq_rel

def interaction_loss(
    predicted_trajectories,  # (batch_size, T, N, 2)
    actual_trajectories,     # (batch_size, T, N, 2)
    actual_interaction_matrix,  # (batch_size, T, N, N)
    visual_range=5.0, lambda_weight=1.0, epsilon=1e-6
):
    """
    修复张量布尔操作的交互损失函数。
    """
    # 计算速度
    seq_rel = calculate_velocity(actual_trajectories)  # (batch_size, T-1, N, 2)

    # 裁剪时间步
    predicted_trajectories = predicted_trajectories[:, :-1]
    actual_interaction_matrix = actual_interaction_matrix[:, :-1]

    # 初始化交互矩阵
    batch_size, T_minus_1, N, _ = predicted_trajectories.size()
    predicted_interaction_matrix = torch.zeros((batch_size, T_minus_1, N, N), device=predicted_trajectories.device)

    for t in range(T_minus_1):
        # 计算相对位置差分
        delta_pos = predicted_trajectories[:, t, :, None, :] - predicted_trajectories[:, t, None, :, :]  # (batch_size, N, N, 2)
        dist = torch.norm(delta_pos, dim=-1) + epsilon  # (batch_size, N, N)

        # 布尔掩码：只选择交互范围内的行人对
        mask = dist < visual_range

        # 方向一致性
        norm_i = torch.norm(seq_rel[:, t, :, None, :], dim=-1) + epsilon
        norm_j = torch.norm(seq_rel[:, t, None, :, :], dim=-1) + epsilon
        cos_alpha_ij = torch.sum(delta_pos * seq_rel[:, t, :, None, :], dim=-1) / (dist * norm_i)
        cos_alpha_ji = torch.sum(-delta_pos * seq_rel[:, t, None, :, :], dim=-1) / (dist * norm_j)

        # 更新交互矩阵
        predicted_interaction_matrix[:, t] = torch.where(
            mask,
            (norm_i * cos_alpha_ij + norm_j * cos_alpha_ji) / dist,
            torch.tensor(0.0, device=dist.device)
        )

    # 对交互矩阵归一化
    predicted_interaction_matrix = predicted_interaction_matrix / (predicted_interaction_matrix.max() + epsilon)

    # 计算交互矩阵差异
    interaction_diff = (predicted_interaction_matrix - actual_interaction_matrix).pow(2)


    # 返回损失
    loss = lambda_weight * interaction_diff.mean()
    return loss


def calculate_nonzero_density(matrix):
    """
    计算矩阵非零值的密度。
    Args:
        matrix: Tensor of shape (seq_len, max_nodes, max_nodes)，社会交互矩阵。
    Returns:
        density: float，非零值密度。
    """
    nonzero_count = torch.count_nonzero(matrix)
    total_elements = matrix.numel()
    density = nonzero_count.item() / total_elements
    return density

def ade(pred, target, num_of_objs=None):
    """
    num_peds: shape (batch,)
    """
    batch_size = len(pred)
    pred_time = pred.shape[2]
    if num_of_objs is None:
        num_of_objs = [pred.shape[1]] * batch_size
    sum_all = 0
    for s in range(batch_size):
        sum_ = 0
        node_num = num_of_objs[s]
        for i in range(int(node_num)):
            for t in range(pred_time):
                sum_ += math.sqrt((pred[s][i][t][0] - target[s][i][t][0]) ** 2 + (pred[s][i][t][1] - target[s][i][t][1]) ** 2)
        sum_all += sum_ / (pred_time*node_num)
    return sum_all/batch_size

def fde(pred, target, num_of_objs=None):
    batch_size = len(pred)
    pred_time = pred.shape[2]
    if num_of_objs is None:
        num_of_objs = [pred.shape[1]] * batch_size
    sum_all = 0
    for s in range(batch_size):
        sum_ = 0
        node_num = num_of_objs[s]
        for i in range(int(node_num)):
            for t in range(pred_time - 1, pred_time):
                sum_ += math.sqrt((pred[s][i][t][0] - target[s][i][t][0]) ** 2 + (pred[s][i][t][1] - target[s][i][t][1]) ** 2)
        sum_all += sum_/(node_num)
    return sum_all/batch_size

def ade_sstgcnn(pred, target, num_of_objs=None,mean=True, theta_=False):
    """
    num_peds: shape (batch,)
    """
    batch_size = len(pred)
    pred_time = pred.shape[2]
    if num_of_objs is None:
        num_of_objs = [pred.shape[1]] * batch_size
    sum_all = 0
    for s in range(batch_size):
        # sum_ = 0
        node_num = num_of_objs[s]
        for i in range(int(node_num)):
            sum_ = 0
            for t in range(pred_time):
                if theta_:
                    epsilon = 1e-10
                    pred_v = pred[s][i,t]
                    target_v = target[s][i,t]
                    pred_v_norm = np.sqrt((pred_v**2).sum())
                    target_v_norm = np.sqrt((target_v**2).sum())
                    if pred_v_norm > epsilon and target_v_norm > epsilon:
                        sum_ += 1.0 - np.sum(pred_v*target_v)/(pred_v_norm*target_v_norm)
                    else:
                        sum_ += (1.0/(1+np.exp(-np.abs(pred_v_norm-target_v_norm)))-0.5)*4.0
                else:
                    sum_ += math.sqrt((pred[s][i][t][0] - target[s][i][t][0]) ** 2 + (pred[s][i][t][1] - target[s][i][t][1]) ** 2)
            sum_all += sum_ / pred_time
    if mean:
        return sum_all / sum(num_of_objs)
    else:
        return sum_all

def fde_sstgcnn(pred, target, num_of_objs=None, mean=True):
    batch_size = len(pred)
    pred_time = pred.shape[2]
    if num_of_objs is None:
        num_of_objs = [pred.shape[1]] * batch_size
    sum_all = 0
    for s in range(batch_size):
        # sum_ = 0
        node_num = num_of_objs[s]
        for i in range(int(node_num)):
            sum_ = 0
            for t in range(pred_time - 1, pred_time):
                sum_ += math.sqrt((pred[s][i][t][0] - target[s][i][t][0]) ** 2 + (pred[s][i][t][1] - target[s][i][t][1]) ** 2)
            sum_all += sum_
    if mean:
        return sum_all / sum(num_of_objs)
    else:
        return sum_all

def cal_aae(pred_traj, pred_traj_gt, mode='sum', epsilon=1e-12):
    """ calculating AAE
    pred_traj: ndarray of shape (seq, batch, xy)
    ppred_traj_gt: ndarray of shape (seq, batch, xy)
    """
    # def _sigmoid(x_):
    #     x = x_.copy()
    #     _mask = x < 0
    #     x[_mask] = np.exp(x[_mask])/(1.0+np.exp(x[_mask]))
    #     x[~_mask] = 1.0/(1.0+np.exp(-x[~mask]))
    #     return x

    assert pred_traj.shape[0] == 1 and pred_traj_gt.shape[0] == 1
    pred_traj = pred_traj.squeeze()
    pred_traj_gt = pred_traj_gt.squeeze()
    
    pred_traj = pred_traj.permute(1,0,2)
    pred_traj_gt = pred_traj_gt.permute(1,0,2)
    loss = (pred_traj * pred_traj_gt).sum(axis=2)
    pred_traj_norm = torch.sqrt((pred_traj**2).sum(axis=2))
    pred_traj_gt_norm = torch.sqrt((pred_traj_gt**2).sum(axis=2))
    
    mask = (pred_traj_norm > epsilon) & (pred_traj_gt_norm > epsilon)
    loss[mask] = 1.0 - loss[mask]/(pred_traj_norm[mask]*pred_traj_gt_norm[mask])
    loss[~mask] = (torch.sigmoid(torch.abs(pred_traj_norm[~mask]-pred_traj_gt_norm[~mask])) - 0.5) * 4.0
    loss = loss.sum(dim=1)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def aae_1(pred_traj, pred_traj_gt, epsilon=1e-12):
    """
    pred: shape (1, seq, 1, 2)
    gt: shape (1, seq, 1, 2)
    """
    loss = (pred_traj * pred_traj_gt).sum(axis=-1)
    pred_traj_norm = torch.sqrt((pred_traj**2).sum(axis=-1))
    pred_traj_gt_norm = torch.sqrt((pred_traj_gt**2).sum(axis=-1))
    
    mask = (pred_traj_norm > epsilon) & (pred_traj_gt_norm > epsilon)
    # print("pred_traj_norm:{},pred_traj_gt_norm:{}".format(pred_traj_norm.shape,pred_traj_gt_norm.shape))
    #pred_traj_norm:[1, 1, 1],pred_traj_gt_norm:[1, 12, 1]
    loss[mask] = 1.0 - loss[mask]/(pred_traj_norm[mask]*pred_traj_gt_norm[mask])
    loss[~mask] = (torch.sigmoid(torch.abs(pred_traj_norm[~mask]-pred_traj_gt_norm[~mask])) - 0.5) * 4.0
    loss = loss.mean()
    return loss

def ade_1(pred, target, count, loss_mask=None, theta_=False):
    assert pred[0].shape[0] == 1
    n = len(pred)
    sum_all = 0
    for s in range(n):
        pred_ = np.swapaxes(pred[s][:,:,:count[s],:],1,2)  # [batch, node, sq, feat]
        target_ = np.swapaxes(target[s][:,:,:count[s],:],1,2)
        
        pred_ = np.squeeze(pred_, 0)
        target_ = np.squeeze(target_, 0)


        N = pred_.shape[0]
        T = pred_.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                # if theta_:
                #     pred_theta = to_theta(pred_[i,t])
                #     # target_theta = to_theta(target_[i,t])
                #     # sum_ += np.abs(pred_theta-target_theta)
                #     epsilon = 1e-10
                #     pred_v = pred_[i,t]
                #     target_v = target_[i,t]
                #     pred_v_norm = np.sqrt((pred_v**2).sum())
                #     target_v_norm = np.sqrt((target_v**2).sum())
                #     if pred_v_norm > epsilon and target_v_norm > epsilon:
                #         sum_ += 1.0 - np.sum(pred_v*target_v)/(pred_v_norm*target_v_norm)
                #     else:
                #         sum_ += (1.0/(1+np.exp(-np.abs(pred_v_norm-target_v_norm)))-0.5)*4.0
                # else:
                sum_ += np.sqrt((pred_[i,t,0] - target_[i,t,0])**2+(pred_[i,t,1] - target_[i,t,1])**2)
        sum_all += sum_/(N*T)
    return sum_all/n

def fde_1(pred, target, count, theta_=False):
    assert pred[0].shape[0] == 1
    n = len(pred)
    sum_all = 0
    for s in range(n):
        pred_ = np.swapaxes(pred[s][:,:,:count[s],:],1,2)  # [batch, node, sq, feat]
        target_ = np.swapaxes(target[s][:,:,:count[s],:],1,2)
        
        pred_ = np.squeeze(pred_, 0)
        target_ = np.squeeze(target_, 0)

        N = pred_.shape[0]
        T = pred_.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T-1,T):
                if theta_:
                    pred_theta = to_theta(pred_[i,t])
                    target_theta = to_theta(target_[i,t])
                    sum_ += np.abs(pred_theta-target_theta)
                else:
                    sum_ += np.sqrt((pred_[i,t,0] - target_[i,t,0])**2+(pred_[i,t,1] - target_[i,t,1])**2)
        sum_all += sum_/(N)
    return sum_all/n

def MSE(input, target):
    mse = ((input-target)**2).mean()
    return float(mse)

def EdgeWiseKL(input, target):
    mask = (input > 0) & (target > 0)
    input = input[mask]
    target = target[mask]
    kl = (target * np.log(target / input)).mean()
    return float(kl)

def EdgeWiseKL_T(input, target):
    mask = (input > 0) & (target > 0)
    input = input[mask]
    target = target[mask]
    kl = (target * torch.log(target / input)).mean()
    return float(kl)

def get_class_acc(y, pred, n_classes):
    y = y.data.cpu()
    pred = pred.data.cpu()
    y = y.reshape(-1, n_classes)
    pred = pred.reshape(-1, n_classes)
    xi, yi = torch.where(y!=0)
    a, b = torch.where(pred[xi]==pred[xi].max(dim=1)[0].unsqueeze(1))
    pred_ = [b[0]]
    for i in range(1, len(a)):
        if a[i] != a[i-1]:
            pred_.append(b[i])
    pred = torch.LongTensor(pred_)
    acc = torch.mean((pred==yi).type(torch.float))
    return acc

def bivariate_loss(V_pred,V_trgt):
    """
    V_pred: tensor, shape (1, seq, node, feat)
    """

    normx = V_trgt[...,0]- V_pred[...,0]
    normy = V_trgt[...,1]- V_pred[...,1]

    # sx = torch.exp(V_pred[...,2]) #sx
    # sy = torch.exp(V_pred[...,3]) #sy
    # corr = torch.tanh(V_pred[...,4]) #corr

    # 提取并确保协方差矩阵的稳定性
    sx = torch.exp(torch.clamp(V_pred[..., 2], max=10))  # 限制最大值防止数值溢出
    sy = torch.exp(torch.clamp(V_pred[..., 3], max=10))
    corr = torch.tanh(V_pred[..., 4])  # 相关系数在[-1,1]之间

    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # # 计算概率密度
    # result = 0.5 * (z / negRho + 2 * torch.log(sx) + 2 * torch.log(sy) + torch.log(negRho)) + torch.log(
    #     2 * torch.tensor(np.pi))


    #
    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho)) + 1e-10

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))


    return torch.mean(result)


def bivariate_gaussian_nll(gt, mu, sigma_x, sigma_y, rho):
    """
    Args:
        gt: [batch_size, num_nodes, pred_len, 2]
        mu: [batch_size, num_nodes, pred_len, 2]
        sigma_x: [batch_size, num_nodes, pred_len]
        sigma_y: [batch_size, num_nodes, pred_len]
        rho: [batch_size, num_nodes, pred_len]
    """
    # 计算偏差
    normx = gt[..., 0] - mu[..., 0]  # [batch_size, num_nodes, pred_len]
    normy = gt[..., 1] - mu[..., 1]  # [batch_size, num_nodes, pred_len]

    # 添加数值稳定性
    eps = 1e-6
    sigma_x = sigma_x + eps
    sigma_y = sigma_y + eps

    # 计算指数项
    z = (normx / sigma_x) ** 2 + (normy / sigma_y) ** 2 \
        - 2 * rho * normx * normy / (sigma_x * sigma_y)

    # 计算行列式项
    neg_rho = 1 - rho ** 2 + eps

    # 计算负对数似然
    nll = 0.5 * (z / neg_rho + 2 * torch.log(sigma_x) + 2 * torch.log(sigma_y) \
                 + torch.log(neg_rho) + 2 * np.log(2 * np.pi))

    return nll  # [batch_size, num_nodes, pred_len]


def gmm_loss(pred_params, pred_weights, gt_trajectory, reduce_mode='min'):
    """
    Args:
        pred_params: [batch_size, num_nodes, num_modes, pred_len, 5]
        pred_weights: [batch_size, num_nodes, num_modes]
        gt_trajectory: [batch_size, pred_len, num_nodes, 2]
    """
    # 转换gt_trajectory维度以匹配预测
    gt_trajectory = gt_trajectory.permute(0, 2, 1, 3)  # [batch_size, num_nodes, pred_len, 2]

    # 计算每个模态的负对数似然
    nll_loss = []
    for m in range(pred_params.size(2)):  # 遍历每个模态
        mode_params = pred_params[:, :, m]  # [batch_size, num_nodes, pred_len, 5]

        # 提取参数
        mu = mode_params[..., :2]  # [batch_size, num_nodes, pred_len, 2]
        sigma_x = torch.exp(mode_params[..., 2])  # [batch_size, num_nodes, pred_len]
        sigma_y = torch.exp(mode_params[..., 3])  # [batch_size, num_nodes, pred_len]
        rho = torch.tanh(mode_params[..., 4])  # [batch_size, num_nodes, pred_len]

        # 计算负对数似然
        nll = bivariate_gaussian_nll(
            gt_trajectory,  # [batch_size, num_nodes, pred_len, 2]
            mu,  # [batch_size, num_nodes, pred_len, 2]
            sigma_x,  # [batch_size, num_nodes, pred_len]
            sigma_y,  # [batch_size, num_nodes, pred_len]
            rho  # [batch_size, num_nodes, pred_len]
        )  # [batch_size, num_nodes, pred_len]

        nll = nll.sum(dim=-1)  # [batch_size, num_nodes]
        nll_loss.append(nll.unsqueeze(-1))

    nll_loss = torch.cat(nll_loss, dim=-1)  # [batch_size, num_nodes, num_modes]

    if reduce_mode == 'min':
        # 只使用最佳模态的损失
        min_nll, _ = nll_loss.min(dim=-1)  # [batch_size, num_nodes]
        loss = min_nll.mean()
    else:
        # 使用所有模态的加权损失
        loss = (pred_weights * nll_loss).sum(dim=-1).mean()

    return loss

def kde_nll_1(predicted_trajs, gt_traj):
    """
    :param predicted_trajs: ndarray, shape [1, num_samples, seq_len, 2]
    :param gt_traj: ndarray, shape [seq_len, 2]
    """
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf
            except np.linalg.LinAlgError:
                kde_ll = np.nan
    kde_ll = kde_ll / (num_timesteps * num_batches)
    return -kde_ll

# AMD / AMV
def calc_amd_amv(gt, pred):
    total = 0
    m_collect = []
    gmm_cov_all = 0
    for i in range(pred.shape[0]):  #per time step
        for j in range(pred.shape[1]):
            #do the method of finding the best bic
            temp = pred[i, j, :, :]

            gmm = get_best_gmm2(pred[i, j, :, :])
            center = np.sum(np.multiply(gmm.means_, gmm.weights_[:,
                                                                 np.newaxis]),
                            axis=0)
            gmm_cov = 0
            for cnt in range(len(gmm.means_)):
                gmm_cov += gmm.weights_[cnt] * (
                    gmm.means_[cnt] - center)[..., None] @ np.transpose(
                        (gmm.means_[cnt] - center)[..., None])
            gmm_cov = np.sum(gmm.weights_[..., None, None] * gmm.covariances_,
                             axis=0) + gmm_cov

            dist, _ = mahalanobis_d(
                center, gt[i, j], len(gmm.weights_), gmm.covariances_,
                gmm.means_, gmm.weights_
            )  #assume it will be the true value, add parameters

            total += dist
            gmm_cov_all += gmm_cov
            m_collect.append(dist)

    gmm_cov_all = gmm_cov_all / (pred.shape[0] * pred.shape[1])
    return total / (pred.shape[0] *
                    pred.shape[1]), None, None, m_collect, np.abs(
                        np.linalg.eigvals(gmm_cov_all)).max()

def mahalanobis_d(x, y, n_clusters, ccov, cmeans, cluster_p):  #ccov
    v = np.array(x - y)
    Gnum = 0
    Gden = 0
    for i in range(0, n_clusters):
        ck = np.linalg.pinv(ccov[i])
        u = np.array(cmeans[i] - y)
        val = ck * cluster_p[i]
        b2 = 1 / (v.T @ ck @ v)
        a = b2 * v.T @ ck @ u
        Z = u.T @ ck @ u - b2 * (v.T @ ck @ u)**2
        pxk = np.sqrt(np.pi * b2 / 2) * np.exp(-Z / 2) * (erf(
            (1 - a) / np.sqrt(2 * b2)) - erf(-a / np.sqrt(2 * b2)))
        Gnum += val * pxk
        Gden += cluster_p[i] * pxk
    G = Gnum / Gden
    mdist = np.sqrt(v.T @ G @ v)
    if np.isnan(mdist):
        # print(Gnum, Gden)
        '''
        print("is nan")
        print(v)
        print("Number of clusters", n_clusters)
        print("covariances", ccov)
        '''
        return 0, 0

    # print( "Mahalanobis distance between " + str(x) + " and "+str(y) + " is "+ str(mdist) )
    return mdist, G

def get_best_gmm(X):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(
        1, 7)  ## stop based on fit/small BIC change/ earlystopping
    cv_types = ['full']
    best_gmm = GaussianMixture()
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    return best_gmm

def get_best_gmm2(X):  #early stopping gmm
    lowest_bic = np.infty
    bic = []
    cv_types = ['full']  #changed to only looking for full covariance
    best_gmm = GaussianMixture()
    for cv_type in cv_types:
        p = 1  #Decide a value
        n_comps = 1
        j = 0
        while j < p and n_comps < 5:  # if hasn't improved in p times, then stop. Do it for each cv type and take the minimum of all of them
            gmm = GaussianMixture(n_components=n_comps,
                                  covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                j = 0  #reset counter
            else:  #increment counter
                j += 1
            n_comps += 1

    bic = np.array(bic)
    return best_gmm

def kde_lossf(gt, pred):
    #(12, objects, samples, 2)
    # 12, 1600,1000,2
    kde_ll = 0
    kde_ll_f = 0
    n_u_c = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            temp = pred[i, j, :, :]
            n_unique = len(np.unique(temp, axis=0))
            if n_unique > 2:
                kde = gaussian_kde(pred[i, j, :, :].T)
                t = np.clip(kde.logpdf(gt[i, j, :].T), a_min=-20,
                            a_max=None)[0]
                kde_ll += t
                if i == (pred.shape[0] - 1):
                    kde_ll_f += t
            else:
                n_u_c += 1
    if n_u_c == pred.shape[0] * pred.shape[1]:
        return 0
    return -kde_ll / (pred.shape[0] * pred.shape[1])


def get_trajectory_angles(x):
    '''
        Input: (batch_size, time_steps, human_id, 2D_coordinates)
        Output: (batch_size, time_steps-2, human_id)  # Angles between consecutive trajectory vectors for each human
    '''
    eps = 1e-7
    # 计算每个人相邻时间步之间的轨迹向量 (N, T-1, human_id, 2)
    traj_vectors = x[:, 1:, :, :] - x[:, :-1, :, :]  # 每个时间步的相邻向量

    # 计算相邻向量之间的角度（余弦相似度）
    angles = F.cosine_similarity(traj_vectors[:, 1:, :, :], traj_vectors[:, :-1, :, :], dim=-1)  # (N, T-2, human_id)

    # 计算余弦相似度对应的角度
    return torch.acos(angles.clamp(-1 + eps, 1 - eps))  # 返回角度，维度为 (N, T-2, human_id)


def loss_trajectory_angle(x, gt):
    '''
        Input: (batch_size, time_steps, human_id, 2), (batch_size, time_steps, human_id, 2)
        Output: Scalar loss value
    '''
    # 获取预测轨迹和真实轨迹的角度
    pred_angles = get_trajectory_angles(x)  # (N, T-2, human_id)
    gt_angles = get_trajectory_angles(gt)  # (N, T-2, human_id)

    # 计算每个行人的角度差异的 L1 损失
    return nn.L1Loss()(pred_angles, gt_angles)


def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative for each pedestrian)

    Input:
    - x: (batch_size, time_steps, human_id, 2), predicted 2D trajectory
    - gt: (batch_size, time_steps, human_id, 2), ground truth 2D trajectory

    Output:
    - Scalar loss value representing the L1 loss of angle velocities between predicted and ground truth.
    """



    # 如果时间步少于2步，不计算损失
    if x.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)

    # 获取轨迹中的角度变化
    x_a = get_trajectory_angles(x)  # (batch_size, time_steps-2, human_id)
    gt_a = get_trajectory_angles(gt)  # (batch_size, time_steps-2, human_id)

    # 计算相邻时间步之间的角度差分（角速度）
    x_av = x_a[:, 1:, :] - x_a[:, :-1, :]  # 计算预测角速度，维度为 (batch_size, time_steps-3, human_id)
    gt_av = gt_a[:, 1:, :] - gt_a[:, :-1, :]  # 计算真实角速度，维度为 (batch_size, time_steps-3, human_id)

    # 计算角速度之间的 L1 损失
    return nn.L1Loss()(x_av, gt_av)


def loss_velocity(predicted, target):
    """
    Mean velocity error for multi-pedestrian trajectory prediction.

    Input:
    - predicted: (batch_size, time_steps, human_id, 2), predicted 2D trajectory
    - target: (batch_size, time_steps, human_id, 2), ground truth 2D trajectory

    Output:
    - Scalar loss value representing the mean velocity error.
    """
    # 保证预测值和真实值的形状一致
    # assert predicted.shape == target.shape

    # 如果时间步小于等于1，无法计算速度差分，返回零损失
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.).to(predicted.device)

    # 计算预测轨迹和真实轨迹的速度（相邻时间步的位置差分）
    velocity_predicted = predicted[:, 1:, :, :2] - predicted[:, :-1, :, :2]  # (batch_size, time_steps-1, human_id, 2)
    velocity_target = target[:, 1:, :, :] - target[:, :-1, :, :]  # (batch_size, time_steps-1, human_id, 2)

    # 计算速度差异的欧氏距离
    velocity_diff = velocity_predicted - velocity_target  # (batch_size, time_steps-1, human_id, 2)
    velocity_error = torch.norm(velocity_diff, dim=-1)  # 计算每个时间步的速度差，维度变为 (batch_size, time_steps-1, human_id)

    # 返回平均速度误差
    return torch.mean(velocity_error)

def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """

    norm_predicted = torch.mean(torch.sum(predicted[:, :, :, :2] ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted[:, :, :, :2], dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    temp = scale * predicted[:, :, :, :2] - target
    return torch.mean(torch.norm(temp, dim=len(target.shape)-1))