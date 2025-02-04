import pickle
import os
import time
import argparse
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from GNN_based_model import *
from utils import *
from metrics import *
import config
# from config import device
from visualization import Visualizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from TrajectoryDataset import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch_lr_finder import LRFinder

torch.backends.cudnn.enabled=False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2,initial')
parser.add_argument('--load_pre', action='store_true', help="load pretrain weight")
parser.add_argument('--world_size', type=int, default=2)

args = parser.parse_args()

model_name = config.model_name
max_node_num =  config.max_node_num
mask = config.mask
distributed = len(config.device_ids) > 1

vis_name = f"{config.vis_name}_{config.model_name}_{args.dataset}"
vis = Visualizer(env=vis_name)

print('*'*30)
print("Training initiating....")

#Data prep
obs_seq_len = config.obs_seq_len
pred_seq_len   = config.pred_seq_len
dataset = args.dataset
data_set = './datasets_new/'+dataset+'/'
loader_batchsize = config.loader_size
if loader_batchsize == 0:
    loader_batchsize = config.batch_size
cache_name = config.cache_name

if distributed:
    dist.init_process_group("nccl", init_method="env://")
    print('distributed.')



dset_train = TrajectoryDataset(
        data_set+'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,
        norm_lap_matr=False,
        delim='\t',
        max_nodes=max_node_num,
        test='train',
        dataset=dataset)

if distributed:
    world_size = args.world_size
    assert loader_batchsize % world_size == 0
    loader_batchsize = loader_batchsize // world_size
    train_sampler = DistributedSampler(dset_train)
    loader_train = DataLoader(dset_train, sampler=train_sampler,
        batch_size=loader_batchsize, num_workers=2)
else:
    loader_train = DataLoader(
            dset_train,
            batch_size=loader_batchsize, #This is irrelative to the args batch size parameter
            shuffle=True,
            num_workers=0)

vald = True
if vald:
    dset_val = TrajectoryDataset(
            data_set+'val/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,
            norm_lap_matr=False, 
            delim='\t', 
            max_nodes=max_node_num, 
            test='val',
            dataset=dataset)
    if distributed:
        val_sampler = DistributedSampler(dset_val)
        loader_val = DataLoader(dset_val, sampler=val_sampler, 
            batch_size=loader_batchsize, shuffle=False, num_workers=2)
    else:
        loader_val = DataLoader(
                dset_val,
                batch_size=loader_batchsize, #This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=0)

# model
generator = get_model(model_name)
print(generator)
# print(discriminator)

device_count = len(config.device_ids)
print("cuda",config.cuda, device_count)
rank = -1
if device_count > 1:
    rank = dist.get_rank()
    device_id = config.device_ids[rank % device_count]
    print(f"Running DDP on rank {rank}, device {device_id}.")
    device = torch.device('cuda', device_id)
    print("device:", device)
    generator = generator.to(device_id)
    # discriminator = discriminator.to(device_id)
    generator = DDP(generator, device_ids=[device_id],find_unused_parameters=False)
    # discriminator = DDP(discriminator, device_ids=[device_id])
else:
    # device = config.device

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print("device:",device)
    generator.to(device)
    # discriminator.to(device)

# optimizer
optimizer = optim.RMSprop(generator.parameters(), lr=config.pretrain_learning_rate,weight_decay=0)
# generator_optimizer = optim.RMSprop(generator.parameters(), lr=config.g_learning_rate,weight_decay=1e-4, alpha=0.999)
# discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=config.d_learning_rate, alpha=0.999)
# pretrain_optimizer = optimizer = torch.optim.AdamW(
#     generator.parameters(),
#     lr=config.pretrain_learning_rate,
#     weight_decay=1e-4
# )

# from adabelief_pytorch import AdaBelief

# optimizer = AdaBelief(
#     generator.parameters(),
#     lr=config.pretrain_learning_rate,
#     eps=1e-16,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     weight_decouple=True,
#     rectify=False
# )

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)




# loss
# 添加学习率调度器
# pretrain_scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=config.pretrain_learning_rate,
#     epochs=config.p_epoch,
#     steps_per_epoch=len(loader_train),
#     pct_start=0.3
# )

mse_criterion = nn.MSELoss(reduction='mean')

# lr scheduler
lr_scheduler_on = config.use_lrschd != ""
if lr_scheduler_on:
    lrschd = config.use_lrschd
    pretrain_scheduler = optim.lr_scheduler.StepLR(pretrain_optimizer, step_size=5, gamma=0.7)
    if lrschd.startswith('exp'):
        gamma = 0.8 if len(lrschd) == 3 else float(lrschd[3:])/10.0
        g_scheduler = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=gamma, last_epoch=-1)
        # d_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=gamma, last_epoch=-1)
    elif lrschd.startswith('step'):
        gamma = 0.8 if len(lrschd) == 4 else float(lrschd[4:])/10.0
        g_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=5, gamma=gamma)
        # d_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=6, gamma=gamma)
    elif lrschd == 'ronp':
        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(generator_optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
        # d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(discriminator_optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
    else:
        print(f'Not supported lr scheduler: {lrschd}')
        sys.exit(1)

checkpoint_dir = './checkpoint/'+'%s-%s/' % (model_name, dataset)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)


def get_pos_loss_mse(pred_, gt_):
    """
    :param V_tr, shape (batch, seq, node, 2)
    """
    loss1 = mse_criterion(pred_[...,:2], gt_[...,:2])
    loss = loss1

    return loss

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

        
loss_type = config.loss_type
if loss_type == "biv":
    loss_func = bivariate_loss
elif loss_type == "mse":
    loss_func = get_pos_loss_mse
else:
    print(f"loss_type '{loss_type}' not supported !!!")
    sys.exit(1)


def graph_loss(pred_, gt_, *args):
    loss = loss_func(pred_, gt_)
    return loss

def eval_pred_convert(x, *args):
    return x

def disc_pred_convert(x, *args):
    return x


def evaluate(model, epoch, name_window='validation'):
    total_ade = 0
    total_fde = 0
    total_samples = 0
    total_mse = 0
    total_kl = 0
    total_peds = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader_val):
            batch = [tensor.to(device) for tensor in data]
            V_obs, A_obs, V_tr, loss_mask, obs_traj, num_peds, A_pred,= batch
            obs_mask = loss_mask[...,:obs_seq_len]
            loss_mask = loss_mask[...,obs_seq_len:]
            if V_obs.shape[2] < 57:
                continue
            V_pred = generator(V_obs, A_obs, obs_mask)
            V_pred = V_pred * loss_mask.permute(0,2,1).unsqueeze(3)
            predicted_shot = eval_pred_convert(V_pred, V_tr[...,:2], 10)

            V_x = seq_to_nodes(obs_traj).to(device)
            V_pred_rel_to_abs = nodes_rel_to_node_abs(predicted_shot[...,:2], V_x[:,-1,:,:])
            V_y_rel_to_abs= nodes_rel_to_node_abs(V_tr[...,:2], V_x[:,-1,:,:])
            predicted_shot = V_pred_rel_to_abs
            out_shot = V_y_rel_to_abs

            batch_size =V_obs.size(0)
            total_samples += batch_size
            # total_mse += batch_size * MSE(predicted_shot, out_shot)
            total_loss += batch_size * graph_loss(V_pred, V_tr).item()

            predicted_shot1 = predicted_shot.permute(0,2,1,3)
            out_shot1 = out_shot.permute(0,2,1,3)
            predicted_shot1 = predicted_shot1.data.cpu().numpy()
            out_shot1 = out_shot1.data.cpu().numpy()
            num_peds = num_peds.data.cpu().numpy()
            ade_ = ade_sstgcnn(predicted_shot1, out_shot1, num_peds, False)
            fde_ = fde_sstgcnn(predicted_shot1, out_shot1, num_peds, False)
            total_ade += ade_
            total_fde += fde_
            total_peds += sum(num_peds)

    avg_ade = total_ade/total_peds
    avg_fde = total_fde/total_peds
    # avg_mse = total_mse/total_samples
    avg_loss = total_loss/total_samples
    print("[Epoch %d] ADE: %.4f, FDE: %.4f, Loss: %.4f" %
        (epoch, avg_ade, avg_fde, avg_loss))
    vis.plot_many_stack(epoch+1,
        {'ade': avg_ade, 'fde': avg_fde}, xlabel='epoch', name_window=name_window)
    vis.plot_one(epoch+1, avg_loss, 'loss_'+name_window, xlabel='epoch')
    return avg_ade, avg_fde, avg_loss

pre_train_count = 0
train_count = 0

def pre_train_step(data, i, best_k=4, nll=False):
    global pre_train_count
    pre_train_count += 1
    batch = [tensor.to(device) for tensor in data]
    V_obs, A_obs, V_tr, loss_mask, num_peds, A_pred, seq_rel, = batch
    obs_mask = loss_mask[...,:obs_seq_len]
    loss_mask = loss_mask[...,obs_seq_len:]
    density = calculate_nonzero_density(A_obs) # A_obs的非零值的密度

    # if best_k > 0:
    #     loss_rel = []
    #     for _ in range(best_k):
    #         predicted_shot = generator(V_obs, A_obs, obs_mask)
    #         # print("*****best_k——predicted_shot:{}******",format(predicted_shot.shape))
    #         pos_loss = get_pos_loss_k(predicted_shot[...,:2], V_tr[...,:2], loss_mask, mode='raw')
    #         loss_rel.append(pos_loss)
    #     loss_rel = torch.stack(loss_rel, dim=2)
    #     loss = torch.zeros(1).to(V_tr)
    #     for b_ in range(V_obs.shape[0]):
    #         num_ped_ = int(num_peds[b_])
    #         _loss_rel = loss_rel[b_, 0:num_ped_]
    #         _loss_rel = torch.sum(_loss_rel, dim=0)
    #         _loss_rel = torch.min(_loss_rel) / torch.sum(loss_mask[b_, 0:num_ped_])
    #         loss += _loss_rel
    #     pos_loss = loss
    # else:
    predicted_shot = generator(V_obs, A_obs)
    # print("*****predicted_shot:{}******", format(predicted_shot.shape))

    predicted_shot = predicted_shot * loss_mask.permute(0,2,1).unsqueeze(3)   # loss_mask: torch.Size([64, 57, 12])
    # V_tr = V_tr * loss_mask.permute(0,2,1).unsqueeze(3)
    pos_loss = graph_loss(predicted_shot, V_tr, pre_train_count, 'pre')
    mse_loss = get_pos_loss_mse(predicted_shot, V_tr)
    # trajectory_angle_loss = loss_trajectory_angle(predicted_shot, V_tr)
    angle_velocity_loss = loss_angle_velocity(predicted_shot, V_tr)
    velocity_loss = loss_velocity(predicted_shot, V_tr)
    mpjpe  = n_mpjpe(predicted_shot, V_tr)
    regularization_loss = interaction_loss(predicted_shot, V_tr, A_pred, visual_range=5.0, lambda_weight=1.0, epsilon=1e-6)

    # print("pos_loss:{}, mse_loss:{}".format(pos_loss, mse_loss))
    # loss = pos_loss * loss_mask.permute(0, 2, 1).unsqueeze(3)
    # loss = loss.sum() / loss_mask.sum()
    # total_loss = pos_loss + regularization_loss
    #
    total_loss= pos_loss + mse_loss + regularization_loss

    print('[epoch %d] [step %d] [density %.4f] [pos loss %.4f, regularization_loss %.4f, mse %.4f, angle_velocity_loss %.4f, velocity_loss %.4f, total_loss %.4f]'
          % (epoch, i, density, pos_loss.item(), regularization_loss.item(), mse_loss.item(), angle_velocity_loss.item(), velocity_loss.item(), total_loss.item()))

    return pos_loss, mse_loss, angle_velocity_loss, velocity_loss, total_loss, regularization_loss


def save_model(model, path):
    if distributed and rank != -1:
        if rank == 0:
            torch.save(model.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


loader_len = len(loader_train)
turn_point = loader_len -1
acc_batchsize = config.batch_size
nll=False

acc_train_on = (loader_batchsize == 1)
constant_metrics = {'pretrain':{'min_val_epoch':-1, 'min_val_loss': 1e16},
                    'train':{'min_val_epoch':-1, 'min_val_loss': 1e16}}
if args.load_pre:
    generator.load_state_dict(torch.load(checkpoint_dir+'pretrain_best.pth',map_location=device, weights_only=True))
    pepoch = 0


######################### pretrain ########################
step = 0
since = time.time()
for epoch in range(config.p_epoch):
    generator.train()
    if distributed:
        train_sampler.set_epoch(epoch)
    batch_count = 0
    for i, data in enumerate(loader_train):
        # print(f"Batch {i}: {data.shape}")
        # if data.size(1) == 0:
        #     data = torch.zeros([8, 57, 2])
        batchsize_ = len(data[0])
        batch_count += batchsize_
        pos_loss, mse_loss, angle_velocity_loss, velocity_loss, total_loss, regularization_loss = pre_train_step(data, i, best_k=config.bestk)

        optimizer.zero_grad()
        total_loss.backward()
        # if config.gradient_clip is not None:
        #     nn.utils.clip_grad_norm_(generator.parameters(), config.gradient_clip)
        optimizer.step()

        step += 1

    # print(f'Epoch {epoch}, Adaptive Weight Matrix:')
    # print(generator.adaptive_weight)
    # 检查自适应权重矩阵的梯度是否正常
    # print(generator.adaptive_weight.grad)
    # visualize_adaptive_weight(generator.adaptive_weight, epoch)


    # torch.save(generator.adaptive_weight, f'adaptive_weight_epoch_{epoch}.pt')
        vis.plot_many_stack(step,
                            {'mse': mse_loss.item(), 'regularization_loss ': regularization_loss.item(), 'velocity_loss':velocity_loss.item()} , 'Muti_Loss')
        vis.plot_one(step, total_loss.item(), 'total_loss')
    if vald:
        _ade, _fde, _loss = evaluate(generator, epoch, 'validation')
        if _ade < constant_metrics['pretrain']['min_val_loss']:
            constant_metrics['pretrain']['min_val_loss'] = _ade
            constant_metrics['pretrain']['min_val_epoch'] = epoch
            save_model(generator, checkpoint_dir+'pretrain_best.pth')
        scheduler.step(_loss)
    # if lr_scheduler_on:
    # vis.plot_one(epoch+1, pretrain_optimizer.param_groups[0]['lr'], 'pretrain_lr', xlabel='epoch')
    # pretrain_scheduler.step()
spend1 = time.time() - since
with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)
save_model(generator, checkpoint_dir+'pretrain_final.pth')

