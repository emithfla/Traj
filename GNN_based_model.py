from models import generator
import sys
import config

# config
window_size = config.obs_seq_len
pred_len = config.pred_seq_len
in_features = config.in_features
out_features = config.out_features
lstm_features = config.lstm_features
disc_hidden = config.disc_hidden
out_size = config.output_size
max_node_num = config.max_node_num
disc_inpsize = config.disc_inpsize

# models
generators = {
    'cnn-gat': generator.CNN_GAT_Generator,
    'cnn-gcn': generator.CNN_GCN_Generator
    # 'lstm-gat': generator.LSTM_GAT_Generator,
    # 'lstm-gcn': generator.LSTM_GCN_Generator,
    # 'lstm-gat-multi': generator.LSTM_GAT_MULTI_Generator,
    # 'lstm-gcn-ne': generator.LSTM_GCN_NE_Generator,
    # 'cnn-gcn-ne': generator.CNN_GCN_NE_Generator
}
#
# discriminators = {
#     'gcn-gan': discriminator.GCN_GAN_Discriminator,
#     'cnn': discriminator.CNN_Discriminator,
#     'pool': discriminator.POOL_Discriminator,
#     'drnn': discriminator.DRNN_Discriminator,
#     'drnn1': discriminator.DRNN_Discriminator_1,
#     'sgan': discriminator.SGAN_Discriminator
# }
# print('generators:', generators.keys())
# print('discriminators:', discriminators.keys())

# args
generator_base_args = {
    'window_size': window_size,
    'n_pred': pred_len,
    'in_features': in_features,
    'out_features': out_features,
    'out_size': out_size,
    'embedding_dim': 64,
    'n_stgcnn': config.n_stgcnn,
    'n_txpcnn': config.n_txpcnn,
    'node_num': max_node_num,
    'lstm_features': lstm_features
}

# discriminator_base_args = {
#     'input_size': pred_len * disc_inpsize,
#     'hidden_size': disc_hidden
# }

generator_args = {
    'cnn-gat': {**generator_base_args},
    'cnn-gcn': {**generator_base_args}
    # 'lstm-gat': {**generator_base_args},
    # 'lstm-gcn': {**generator_base_args},
    # 'lstm-gat-multi': {**generator_base_args, 'n_head': 4},
    # 'lstm-gcn-ne': {**generator_base_args},
    # 'cnn-gcn-ne': {**generator_base_args}
}

# discriminator_args = {
#     'gcn-gan': {**discriminator_base_args, \
#                 'input_size': max_node_num * pred_len * disc_inpsize},
#     'cnn': {**discriminator_base_args, 'input_size': disc_inpsize},
#     'pool': {**discriminator_base_args},
#     'drnn': {**discriminator_base_args},
#     'drnn1': {**discriminator_base_args, 'input_size': disc_inpsize},
#     'sgan': {'input_size': disc_inpsize, 'embedding_size': 64, 'hidden_size': 64, 'mlp_size': 1024}
# }


def get_model(model_name, model_config=None):
    """
    :param model_name
    :param args
    """
    if model_config is None:
        model_config = config.model_config
    print(model_config)
    genn = model_config['gen']
    # discn = model_config['disc']
    if genn not in generators:
        print('Model "%s" does not exist !!!' % model_name, genn)
        sys.exit(1)
    else:
        generator_ = generators[genn](**generator_args[genn])
        print()
        # discriminator_ = discriminators[discn](**discriminator_args[discn])

    return generator_