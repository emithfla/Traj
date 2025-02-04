import os
import math
import numpy as np
from tqdm import tqdm
import pickle as pkl
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import PillowWriter


# 设置无显示后端
# plt.use('Agg')




def anorm(p1, p2):
    if (np.abs(p1[0]) + np.abs(p1[1])) > 0 and (np.abs(p2[0]) + np.abs(p2[1])) > 0:
        return 1
    return 0



def get_V(seq_, seq_rel):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    if max_nodes == 0 or seq_len == 0:  # 检查无效输入
        raise ValueError("Invalid input: seq_ or seq_rel has zero size")

    V = np.zeros((seq_len, max_nodes, 2))
    # theta = np.zeros((seq_len, max_nodes, 1))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :2] = step_rel[h, :2]
            # theta[s, h, 0] = to_theta(step_rel[h, :2])
            # V[s, h, 3] = np.sqrt(step_rel[h,0]**2+step_rel[h,1]**2)
    # V[1:,:,2:4] = V[1:,:,:2] - V[:-1,:,:2]
    # V[1:,:,2] = theta[1:,:,0]-theta[:-1,:,0]
    return torch.from_numpy(V).type(torch.float)



def get_A(seq_, seq_rel, flag):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    A = np.zeros((seq_len, max_nodes, max_nodes))
    adjusted_range = 5.0

    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_[h], step_[k])  # 基于两点偏移量的距离计算权重
                # if l2_norm > 0:
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        # if norm_lap_matr:
        # G = nx.from_numpy_matrix(A[s, :, :])
        # A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()


    return torch.from_numpy(A).type(torch.float)

def visualize_interaction(seq_, seq_rel, A, visual_range, output_file="interaction.gif"):
    """
    可视化行人轨迹和交互关系，生成动态动图。

    Args:
        seq_: Tensor of shape (max_nodes, 2, seq_len)，行人绝对位置 (x, y)。
        seq_rel: Tensor of shape (max_nodes, 2, seq_len)，行人速度 (vx, vy)。
        A: Tensor of shape (seq_len, max_nodes, max_nodes)，交互矩阵。
        visual_range: float，行人视觉范围。
        output_file: str，保存的动图文件名。
    """
    seq_ = seq_.numpy()
    seq_rel = seq_rel.numpy()
    max_nodes, _, seq_len = seq_.shape

    # 初始化画布
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-20, 20)  # 根据轨迹数据范围调整
    ax.set_ylim(-20, 20)
    ax.set_title("Pedestrian Interaction Visualization")

    # 行人轨迹点、方向箭头和视觉范围
    points = ax.scatter([], [], c='blue', s=50, label="Pedestrian")
    directions = [ax.quiver(0, 0, 0, 0, scale=10, color="blue") for _ in range(max_nodes)]
    ranges = [plt.Circle((0, 0), visual_range, color='blue', alpha=0.2, fill=False) for _ in range(max_nodes)]

    # 交互权重线条
    interaction_lines = []
    for i in range(max_nodes):
        for j in range(max_nodes):
            if i < j:  # 避免重复线条
                line, = ax.plot([], [], color='red', alpha=0.5, lw=1)
                interaction_lines.append((i, j, line))

    # 添加视觉范围到画布
    for circle in ranges:
        ax.add_artist(circle)

    def update(frame):
        """更新每一帧的内容"""
        step_ = seq_[:, :, frame]
        step_rel = seq_rel[:, :, frame]
        A_step = A[frame]

        # 更新行人位置
        valid_indices = np.any(step_ != 0, axis=1)  # 检查哪些行人的位置不是零

        # 更新行人位置，仅更新有效行人
        valid_step_ = step_[valid_indices]
        points.set_offsets(valid_step_)

        # 更新行人方向箭头和视觉范围
        for i, (dx, dy) in enumerate(step_rel):
            if valid_indices[i]:  # 仅更新有效行人
                directions[i].set_offsets(step_[i])
                directions[i].set_UVC(dx, dy)
                ranges[i].set_center(step_[i])
            else:
                # 对于无效的行人，将箭头和圆设置为不可见
                directions[i].set_offsets([np.nan, np.nan])
                ranges[i].set_center([np.nan, np.nan])

        # 更新交互线条
        for i, j, line in interaction_lines:
            if valid_indices[i] and valid_indices[j] and A_step[i, j] > 0:  # 确保交互是有效的
                line.set_data([step_[i, 0], step_[j, 0]], [step_[i, 1], step_[j, 1]])
                line.set_linewidth(A_step[i, j])  # 根据权重调整线条粗细
                line.set_alpha(0.8)
            else:
                line.set_data([], [])

    return points, directions, ranges, interaction_lines

        # 创建 PillowWriter 实例

    writer = PillowWriter(fps=5)

    # 使用 writer 进行保存
    writer.setup(fig, output_file, dpi=100)
    for frame in range(seq_len):
        update(frame)
        writer.grab_frame()  # 保存当前帧
    writer.finish()

    # 关闭图形窗口，释放资源
    plt.close(fig)


def get_average_and_max_speed(seq_rel):
    # 计算每个时间步、每个行人的速度模长
    speed_magnitudes = np.linalg.norm(seq_rel, axis=1)  # (max_nodes, seq_len)
    average_speed = np.mean(speed_magnitudes)
    max_speed = np.max(speed_magnitudes)
    return average_speed, max_speed


# def get_A(seq_, seq_rel, flag):
#     """
#     计算社会交互矩阵 A，基于行人速度、方向角和距离。
#     Args:
#         seq_: Tensor of shape (max_nodes, 2, seq_len)，表示行人的绝对位置 (x, y)。
#         seq_rel: Tensor of shape (max_nodes, 2, seq_len)，表示行人的相对位置 (vx, vy)。
#         visual_range: float，可视范围，限制交互范围。
#     Returns:
#         A: Tensor of shape (seq_len, max_nodes, max_nodes)，社会交互矩阵。
#     """
#     seq_ = seq_.squeeze()
#     seq_rel = seq_rel.squeeze()
#     seq_len = seq_.shape[2]
#     max_nodes = seq_.shape[0]
#
#     if max_nodes == 0 or seq_len == 0:  # 检查无效输入
#         raise ValueError("Invalid input: seq_ or seq_rel has zero size")
#
#     A = np.zeros((seq_len, max_nodes, max_nodes))
#     visual_range = 5.0  # 视野范围
#     epsilon = 1e-6
#     # speed_factor = 5.0
#     stop_speed_factor = 1.0
#     min_speed = 0.1
#
#     # 计算速度的统计信息
#     average_speed, max_speed = get_average_and_max_speed(seq_rel)
#
#
#     for s in range(seq_len):
#         step_ = seq_[:, :, s]  # 每个时间步 t 的绝对位置
#         step_rel = seq_rel[:, :, s]  # 每个时间步 t 的相对位置 (速度)
#
#         # with open("./log/log_%d_%d.txt" % (flag, s), "w") as log_file:
#         #     log_file.write("Debug Log for Interaction Matrix Calculation\n")
#         #     log_file.write("=" * 50 + "\n")
#         for h in range(len(step_)):
#             # A[s, h, h] = 1
#             # 判断行人 h 是否停留在原地
#             if np.linalg.norm(step_rel[h]) < min_speed:
#                 # 如果行人 h 速度低于 min_speed，使用固定的 speed_factor
#                 speed_factor = stop_speed_factor
#             else:
#                 # 计算动态放大因子
#                 speed_factor = min(3 * (np.linalg.norm(step_rel[h]) / (average_speed + epsilon)), 10.0)
#
#             for k in range(h + 1, len(step_)):
#                 # print("speed_factor is ", speed_factor)
#                 # 计算两行人之间的欧几里得距离
#                 dist = np.linalg.norm(step_[h] - step_[k]) + epsilon # 欧几里得距离
#                 # adjusted_range = visual_range + np.linalg.norm(step_rel[h]) * speed_factor
#                 adjusted_range = visual_range + speed_factor * np.linalg.norm(step_rel[h])
#                 # print(f"NONE_Filtered: dist={dist}, visual_range={adjusted_range}")
#                 if dist == 0 or dist > adjusted_range:  # 超出视觉范围或相同点，忽略交互
#                     # print(f"Filtered: dist={dist}, visual_range={adjusted_range}")
#                     continue
#
#                 # 计算方向角 (cos α)
#                 delta_pos = step_[k] - step_[h]
#                 norm_h = np.linalg.norm(step_rel[h])  # 行人 h 的速度模长
#                 norm_k = np.linalg.norm(step_rel[k])  # 行人 k 的速度模长
#
#                 if norm_h > 0 and norm_k > 0:
#                     cos_alpha_hk = np.dot(delta_pos, step_rel[h]) / (dist * norm_h)  # 行人 h 的方向角
#                     cos_alpha_kh = np.dot(-delta_pos, step_rel[k]) / (dist * norm_k)  # 行人 k 的方向角
#                 else:
#                     cos_alpha_hk = 0
#                     cos_alpha_kh = 0
#
#                 # 根据方向角计算交互权重
#                 if cos_alpha_hk > 0:
#                     A[s, h, k] = (norm_h * cos_alpha_hk + norm_k * cos_alpha_kh) / dist
#                 else:
#                     A[s, h, k] = 0 # 方向相反交互为0
#
#                 # 对称矩阵
#                 A[s, k, h] = A[s, h, k]
#
#         # 最大最小归一化处理
#         A_min = A.min(axis=(1, 2), keepdims=True)  # 找到每个时间步的最小值
#         A_max = A.max(axis=(1, 2), keepdims=True)  # 找到每个时间步的最大值
#
#         A = (A - A_min) / (A_max - A_min + epsilon)  # 归一化到 [0, 1] 范围
#
#         # # 行归一化处理
#         # row_sum = np.sum(A, axis=2, keepdims=True) + epsilon  # 计算每行的和并避免除以零
#         # A = A / row_sum  # 归一化，使每行的和为 1
#
#             #         # 写入未过滤的结果
#             #         log_file.write(
#             #             f"Kept: dist={dist:.4f}, adjusted_range={adjusted_range:.4f}, "
#             #             f"A[{s},{h},{k}]={A[s, h, k]:.4f}\n"
#             #         )
#             #
#             # log_file.write("\nMatrix Density Calculation Complete\n")
#             # log_file.write("=" * 50 + "\n")
#
#     if flag % 100 == 0:
#         print("visualize_interaction: %d" % flag)
#         visualize_interaction(seq_, seq_rel, A, adjusted_range, output_file="./gif/interaction_%d.gif" % flag)
#
#     return torch.from_numpy(A).type(torch.float)

# def get_A(seq_, seq_rel, flag):
    """
    计算社会交互矩阵 A，基于行人速度、方向角和距离。
    Args:
        seq_: Tensor of shape (max_nodes, 2, seq_len)，表示行人的绝对位置 (x, y)。
        seq_rel: Tensor of shape (max_nodes, 2, seq_len)，表示行人的相对位置 (vx, vy)。
        flag: int，标记参数，用于控制日志输出和可视化生成。
        visual_range: float，可视范围，限制交互范围。
        normalize_method: str，归一化方法，可选 "min-max" 或 "row"。
    Returns:
        A: Tensor of shape (seq_len, max_nodes, max_nodes)，社会交互矩阵。
    """
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    if max_nodes == 0 or seq_len == 0:  # 检查无效输入
        raise ValueError("Invalid input: seq_ or seq_rel has zero size")

    # 初始化交互矩阵
    A = np.zeros((seq_len, max_nodes, max_nodes))
    epsilon = 1e-6
    stop_speed_factor = 1.0
    min_speed = 0.1
    visual_range = 5.0

    # 计算速度的统计信息
    average_speed, max_speed = get_average_and_max_speed(seq_rel)

    for s in range(seq_len):
        # 提取当前时间步的绝对位置和速度
        step_ = seq_[:, :, s]  # (max_nodes, 2)
        step_rel = seq_rel[:, :, s]  # (max_nodes, 2)

        # 计算速度模长
        speeds = np.linalg.norm(step_rel, axis=1)  # (max_nodes,)

        # # 动态调整范围
        # speed_factors = np.where(speeds < min_speed, stop_speed_factor,
        #                          np.minimum(3 * speeds / (average_speed + epsilon), 10.0))  # (max_nodes,)

        # 动态调整范围（改进版）
        speed_factors = np.where(
            speeds < min_speed,
            stop_speed_factor,  # 静止状态的固定因子
            np.clip(1 + 5 * np.log1p(speeds / (average_speed + epsilon)), 1.0, 20.0)  # 使用对数函数映射
        )

        # 行人间位置差分
        delta_pos = step_[:, None, :] - step_[None, :, :]  # (max_nodes, max_nodes, 2)

        # 行人间距离
        dist = np.linalg.norm(delta_pos, axis=-1) + epsilon  # (max_nodes, max_nodes)

        # 调整视觉范围
        adjusted_ranges = visual_range + speed_factors[:, None] * speeds[:, None]  # (max_nodes, max_nodes)

        # 计算方向角 (cos α)
        norm_h = speeds[:, None]  # 行人 h 的速度模长
        norm_k = speeds[None, :]  # 行人 k 的速度模长

        valid_mask = (dist > 0) & (dist <= adjusted_ranges)  # 距离范围内且非自身交互

        delta_pos_h = delta_pos / dist[:, :, None]  # 单位化方向向量
        cos_alpha_hk = np.sum(delta_pos_h * step_rel[:, None, :], axis=-1) / (norm_h + epsilon)  # (max_nodes, max_nodes)
        cos_alpha_kh = np.sum(-delta_pos_h * step_rel[None, :, :], axis=-1) / (norm_k + epsilon)  # (max_nodes, max_nodes)

        # 计算交互权重
        interaction_weights = np.where(
            cos_alpha_hk > 0,
            (norm_h * cos_alpha_hk + norm_k * cos_alpha_kh) / dist,
            0
        )

        # 更新交互矩阵，仅保留有效交互
        A[s] = interaction_weights * valid_mask

    # 归一化交互矩阵

    row_sum = A.sum(axis=2, keepdims=True) + epsilon  # 行归一化
    A = A / row_sum

    # 可视化（可选）
    if flag % 10 == 0:
        visualize_interaction(seq_, seq_rel, A, adjusted_ranges, output_file=f"./gif/interaction_{flag}.gif")

    return torch.from_numpy(A).float()


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            # line = line.strip().split(',')
            # print(line)
            # for i in line:
                # print(i)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)







class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    # 轨迹数据集的数据加载器
    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t', norm_lap_matr=True, test='test', max_nodes=0,
            cache_train_name=None, cache_test_name=None, cache_val_name=None, dataset='eth'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        参数：
         -data_dir：目录包含格式为的数据集文件
         <frame_id> <ped_id> <x> <y>
         -obs_len：输入轨迹中的时间步数
         -pred_len：输出轨迹中的时间步数
         -skip：制作数据集时要跳过的帧数
         -threshold：非线性轨迹应考虑的最小误差
         使用线性预测器时
         -min_ped：一个序列中最小行人数量
         -delim：数据集文件中的定界符
         loss_mask 是一个掩码矩阵，
         loss_mask 通过在有效的时间步和行人位置上设置为1（其他无效位置为0），确保损失计算仅对有效的轨迹数据进行，从而避免模型因丢失或不完整的轨迹数据而计算错误的损失
         """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir  # -data_dir：目录包含格式为的数据集文件
        self.obs_len = obs_len  # -obs_len：输入轨迹中的时间步数
        self.pred_len = pred_len  # -pred_len：输出轨迹中的时间步数
        self.skip = skip  # -skip：制作数据集时要跳过的帧数
        self.seq_len = self.obs_len + self.pred_len  # 一个序列中的时间步数
        self.delim = delim  # -delim：数据集文件中的定界符
        self.norm_lap_matr = norm_lap_matr  # # 矩阵拉普拉斯规范化
        self.test = test == 'test' or test == 'val'

        if cache_train_name is None:
            cache_train_name = 'data_caches/cache_%s.pkl' % (dataset)
        if cache_test_name is None:
            cache_test_name = 'data_caches/cache_%s_test.pkl' % (dataset)
        if cache_val_name is None:
            cache_val_name = 'data_caches/cache_%s_val.pkl' % (dataset)

        f = None
        if test == 'train' and os.path.exists(cache_train_name):
            print('%s,train cache' % cache_train_name)
            f = open(cache_train_name, 'rb')
        elif test == 'val' and os.path.exists(cache_val_name):
            print('%s,val cache' % cache_val_name)
            f = open(cache_val_name, 'rb')
        elif test == 'test' and os.path.exists(cache_test_name):
            print('%s,test cache' % cache_test_name)
            f = open(cache_test_name, 'rb')

        if f is not None:
            data_ = pkl.load(f)
            self.seq_start_end = data_['seq_start_end']
            self.v_obs = data_['v_obs']
            self.A_obs = data_['A_obs']
            self.v_pred = data_['v_pred']
            self.A_pred = data_['A_pred']
            self.num_seq = data_['num_seq']
            self.obs_traj = data_['obs_traj']
            self.pred_traj = data_['pred_traj']
            self.obs_traj_rel = data_['obs_traj_rel']
            self.pred_traj_rel = data_['pred_traj_rel']
            self.loss_mask = data_['loss_mask']
            self.non_linear_ped = data_['non_linear_ped']
            self.num_peds_in_seq = data_['num_peds_in_seq']
            f.close()



        else:
            all_files = os.listdir(self.data_dir)  # 返回文件名的列表
            all_files = [os.path.join(self.data_dir, _path) for _path in all_files]  # 拼合路径名形成可访问的路径
            all_files = [path for path in all_files if os.path.isfile(path)]

            print("Processing Data .....")
            num_peds_in_seq = []  # 一个序列中的行人数
            seq_list = []  #
            seq_list_rel = []  #
            loss_mask_list = []  #
            non_linear_ped = []

            # data_list = []
            frame_data_list = []
            frames_list = []
            num_seq_list = []
            # uid_index = {}
            # ui = 0
            # obs_range_list = []

            for path in all_files:
                data = read_file(path, delim)
                # data_list.append(data)
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
                num_seq_list.append(num_sequences)
                frame_data_list.append(frame_data)
                frames_list.append(frames)

                max_peds_in_frame = 0
                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(
                        frame_data[idx:idx + self.seq_len], axis=0)  #
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 去重取出每个序列的行人
                    # for ped_id in peds_in_curr_seq:
                    #     if ped_id not in uid_index:
                    #         uid_index[ped_id] = ui
                    #         ui += 1
                    num_peds_considered = 0
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                     ped_id, :]  # 把当前人的20个序列xy取出[20*2]
                        # curr_ped_seq = np.around(curr_ped_seq, decimals=4)  # 小数点后4位取整
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # 首帧
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1  # 尾帧
                        if pad_end - pad_front != self.seq_len:
                            continue
                        num_peds_considered += 1
                    max_peds_in_frame = max(max_peds_in_frame, num_peds_considered)
                print(path, max_peds_in_frame)
                self.max_peds_in_frame = max(self.max_peds_in_frame, max_peds_in_frame)
            # self.max_peds_in_frame = len(uid_index)
            # if max_ped == -1:
            #     max_ped = self.max_peds_in_frame
            # print('max_ped:', max_ped)
            if max_nodes != 0:
                self.max_peds_in_frame = max_nodes
            else:
                max_nodes = self.max_peds_in_frame

            print('max_peds_in_frame:', self.max_peds_in_frame)
            for i in range(len(num_seq_list)):
                # for path in all_files:
                #     data = read_file(path, delim)  # 读取文件，得到数据矩阵
                #     frames = np.unique(data[:, 0]).tolist()  # 去重后一共有多少帧
                #     frame_data = []  # 按帧存储数据
                #     for frame in frames:
                #         frame_data.append(data[frame == data[:, 0], :])  # 按帧取出所有数据并生成列表
                #     num_sequences = int(
                #         math.ceil((len(frames) - self.seq_len + 1) / skip))  # 窗口滑动，求出序列数
                num_sequences = num_seq_list[i]
                frame_data = frame_data_list[i]
                frames = frames_list[i]
                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(
                        frame_data[idx:idx + self.seq_len], axis=0)  #
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 去重取出每个序列的行人
                    # self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))  # 最大行人数自我更新
                    curr_seq_rel = np.zeros((self.max_peds_in_frame, 2,
                                             self.seq_len))  # [3,2,20]
                    curr_seq = np.zeros((self.max_peds_in_frame, 2, self.seq_len))
                    curr_loss_mask = np.zeros((self.max_peds_in_frame,
                                               self.seq_len))
                    curr_obs_range = np.ones((self.max_peds_in_frame, 2))

                    num_peds_considered = 0
                    _non_linear_ped = []
                    # minx = np.inf, maxx = -np.inf
                    # miny = np.inf, maxy = -np.inf
                    # peds_considered_list = []
                    for _, ped_id in enumerate(peds_in_curr_seq):  # 单位帧内所有人
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                     ped_id, :]  # 把当前人的20个序列xy取出[20*2]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)  # 小数点后4位取整
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # 首帧
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1  # 尾帧
                        if pad_end - pad_front != self.seq_len:
                            continue
                        # maxx, maxy = curr_ped_seq[:self.obs_len, 2:4].max(0)
                        # minx, miny = curr_ped_seq[:self.obs_len, 2:4].min(0)

                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # 取第二列之后的xy

                        # curr_ped_seq = curr_ped_seq
                        # Make coordinates relative
                        # 建立相对坐标系
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = \
                            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]  # 下一帧减当前帧坐标——>求出变化量
                        _idx = num_peds_considered
                        # _idx = uid_index[ped_id]
                        # peds_considered_list.append(_idx)
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq  # 把多个行人的序列放在一个列表（3，2，20）3个行人20帧的坐标
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq  # 同上,（3，2，20）3个行人20帧的相对位移
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(
                            poly_fit(curr_ped_seq, pred_len, threshold))  # 把线性的找出来放在列表里
                        curr_loss_mask[_idx, pad_front:pad_end] = 1  # 设置掩码
                        # curr_obs_range[_idx,:] = maxx-minx, maxy-miny

                        num_peds_considered += 1
                        if num_peds_considered >= max_nodes:
                            break

                    if num_peds_considered > min_ped and num_peds_considered <= max_nodes:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)  # 记录有多少人
                        loss_mask_list.append(curr_loss_mask)  # 掩码
                        seq_list.append(curr_seq)  # 把这个序列所有人的坐标加进列表
                        seq_list_rel.append(curr_seq_rel)  # 同上
                        # obs_range_list.append(curr_obs_range)

            self.num_peds_in_seq = torch.Tensor(num_peds_in_seq)
            self.num_seq = len(seq_list)
            print(len(num_peds_in_seq), max(num_peds_in_seq))
            seq_list = np.concatenate(seq_list, axis=0)
            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            loss_mask_list = np.concatenate(loss_mask_list, axis=0)
            non_linear_ped = np.asarray(non_linear_ped)
            # obs_range_list = np.concatenate(obs_range_list, axis=0)
            # num_peds_in_seq  = 57

            # Convert numpy -> Torch Tensor
            self.obs_traj = torch.from_numpy(
                seq_list[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj = torch.from_numpy(
                seq_list[:, :, self.obs_len:]).type(torch.float)
            self.obs_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, self.obs_len:]).type(torch.float)
            self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
            self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
            # cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            cum_start_idx = [0] + np.cumsum([self.max_peds_in_frame] * len(num_peds_in_seq)).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]
            # self.obs_range = torch.from_numpy(obs_range_list).type(torch.float) # shape [node, 2]

            self.v_obs = []
            self.A_obs = []
            self.v_pred = []
            self.A_pred = []
            # print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]  # 取出来开始和结束时的人数
                # obs_traj:观测的轨迹坐标 obs_traj_rel:坐标偏移量
                v_o = get_V(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :])
                a_o = get_A(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], ss)
                if v_o.shape[1] != self.max_peds_in_frame:
                    print("v_o.shape < max_peds_in_frame:",v_o.shape)
                self.v_obs.append(v_o.clone())
                self.A_obs.append(a_o.clone())

                v_p = get_V(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :],)
                a_p = get_A(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], ss)

                self.v_pred.append(v_p.clone())
                self.A_pred.append(a_p.clone())
                # print("get_A output shape (A_obs):", a_o.shape)
                # print("get_A output shape (A_pred):", a_p.shape)

            pbar.close()

            if test == 'val':
                cache_name = cache_val_name
            elif test == 'test':
                cache_name = cache_test_name
            else:
                cache_name = cache_train_name
            if not os.path.exists('caches'):
                os.mkdir('caches')
            with open(cache_name, 'wb') as f:
                pkl.dump({'seq_start_end': self.seq_start_end,
                          'v_obs': self.v_obs,
                          'A_obs': self.A_obs,
                          'v_pred': self.v_pred,
                          'A_pred': self.A_pred,
                          'num_seq': self.num_seq,
                          'obs_traj': self.obs_traj,
                          'pred_traj': self.pred_traj,
                          'obs_traj_rel': self.obs_traj_rel,
                          'pred_traj_rel': self.pred_traj_rel,
                          'loss_mask': self.loss_mask,
                          'num_peds_in_seq': self.num_peds_in_seq,
                          'non_linear_ped': self.non_linear_ped,
                          }, f)

    def __len__(self):
        return self.num_seq

    #
    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        v_obs, a_obs = self.v_obs[index], self.A_obs[index]
        if v_obs.size(1) == 0 or a_obs.size(1) == 0:  # 检查无效数据
            raise ValueError(f"Invalid data at index {index}")
        # out = [
        #     self.obs_traj[start:end, :], self.pred_traj[start:end, :],
        #     self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
        #     self.non_linear_ped[start:end], self.loss_mask[start:end, :],
        #     self.v_obs[index], self.A_obs[index],
        #     self.v_pred[index], self.A_pred[index], self.num_peds_in_seq
        #
        # ]
        if self.test:
            out = self.v_obs[index][..., :3], self.A_obs[index], self.v_pred[index][..., :3], \
                self.loss_mask[start:end], self.obs_traj[start:end], \
                self.num_peds_in_seq[index], self.A_pred[index]
        else:
            out = self.v_obs[index][..., :3], self.A_obs[index], self.v_pred[index][..., :3], \
                self.loss_mask[start:end], self.num_peds_in_seq[index], self.A_pred[index], self.obs_traj_rel[start:end]
        return out