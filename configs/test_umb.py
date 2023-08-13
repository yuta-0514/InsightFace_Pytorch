from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 10
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "/mnt/umd_face"
config.num_classes = 8277
config.num_image = 367888
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

#SCOPS
config.num_parts = 3
config.learning_rate_w = 0.01
config.ref_net = "vgg19"
config.ref_layer = "relu5_2,relu5_4"
config.ref_norm = False
config.input_size = [112, 112]
config.learning_rate = 1e-5
config.lambda_con = 1e-1
config.lambda_eqv = 1e1
config.lambda_lmeqv = 1e1
config.lambda_sc = 1e2
config.lambda_orthonormal = 1e-1
config.tps_sigma = 0.01
config.tps_mode = "affine"
config.random_scale_low = 0.7
config.random_scale_high = 2.0
config.gpu = 0