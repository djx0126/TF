class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    keep_prob = 0.5
    batch_size = 50
    num_epochs = 5000
    beta = 0.001 # control the rate between result error and total_loss
    cnn_filters = [[1, 3, 32], [2, 5, 64]]  # for each item, [a,b,c] a num of convs in one layer, b conv size, c filter count
    fc_nodes = [128]
    padding_valid = True


class MidConfig(object):
    """Medium config."""
    init_scale = 0.05
    keep_prob = 1.0
    batch_size = 50
    num_epochs = 1000
    beta = 0.1
    cnn_filters = [[2, 3, 128], [3, 5, 256]]  # for each item, [a,b,c] a num of convs in one layer, b conv size, c filter count
    fc_nodes = [128, 128, 64]
    padding_valid = True


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    keep_prob = 1.0
    batch_size = 100
    num_epochs = 200000
    beta = 0.1
    cnn_filters = [[2, 3, 64], [3, 5, 128], [3, 11, 256], [3, 19, 256], [4, 31, 512]]  # for each item, [a,b,c] a num of convs in one layer, b conv size, c filter count
    fc_nodes = [128, 128, 64]
    padding_valid = True
