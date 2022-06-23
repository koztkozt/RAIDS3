# preprocess
class DataConfig(object):
    dataset_name = "udacity"
    data_path = "data/udacity/"
    data_name = "hsv_gray_diff_ch2"  # hsv_gray_diff_ch4
    img_height = 100
    img_width = 100
    num_channels = 2


# stage 1 training
class TrainConfig1(DataConfig):
    model_path = "comma_large_dropout"
    batch_size = 32
    num_epoch = 10
    X_train_mean_path = "X_train_gray_diff2_mean.npy"


# stage 2 training
class TrainConfig2(DataConfig):
    batch_size = 32
    num_epoch = 16


# RAIDS both stage 1 and 2
class RAIDSconfig(DataConfig):
    batch_size = 32
    num_epoch = 16


# optiU training
class optiUConfig(DataConfig):
    batch_size = 1
    target = 0.3


# advGAN training
class advGANConfig(DataConfig):
    batch_size = 1
    num_epoch = 32
    target = 10


# attacks
class attacksconfig(DataConfig):
    batch_size = 1
    num_epoch = 16
    target = 0.3


# defences
class defencesconfig(DataConfig):
    batch_size = 1
    # threshold = 0.8
    target = 0.3


class distillconfig(DataConfig):
    batch_size = 32
    num_epoch = 16


# class TestConfig(TrainConfig2):
#     batch_size = 32
#     num_epoch = 15
#     model_path = "models/weights_hsv_gray_diff_ch4_comma_large_dropout-01-0.00540.hdf5"
#     angle_train_mean = -0.004179079
