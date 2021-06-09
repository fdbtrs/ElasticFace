from easydict import EasyDict as edict

config = edict()
config.dataset = "webface"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.weight_decay_last = 5e-3

config.batch_size = 128
config.lr = 0.1  # batch size is 512
config.output = "output"
config.s=64.0
config.m=0.35


config.local_rank=0
#net paramerters Default iresnet100
config.net_name="iresnet50"
if (config.net_name=="mobilefacenet"):
    config.embedding_size = 128

config.loss="ArcFace"
if (config.loss=="ElasticCosFace" or config.loss=="CosFace"):
    config.s = 64.0
    config.m = 0.35
elif (config.loss=="ElasticArcFace" or config.loss=="ArcFace"):
    config.s = 64.0
    config.m = 0.5
elif (config.loss=="Softmax"):
    config.s = 64.0
    config.m = 0.0

# Resume training, load the model of iteration global_step, at start_epoch
config.global_step=667640
config.resume=1 # set to 1 to load the model
config.start_epoch=0# set the start epoch



if config.dataset == "emore":
    config.rec = "/data/fboutros/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  26
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30" ,"calfw","cplfw"]
    config.eval_step=5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "ms1m-retinaface-t2":
    config.rec = "/train_tmp/ms1m-retinaface-t2"
    config.num_classes = 91180
    config.num_epoch = 25
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [11, 17, 22] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "glint360k":
    config.rec = "/data/fboutros/glink/glint360k"
    config.num_classes = 360232
    config.num_image = 17091657
    config.num_epoch = 22 #20
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
    config.eval_step= 16691 #33350  16691
    config.lr = 0.00001  # batch size is 512
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [8, 12, 15, 18] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "/data/fboutros/faces_webface_112x112"
    config.global_step=18202
    config.num_classes = 10572
    config.num_image = 490623
    config.num_epoch = 50#34
    config.eval_step= 958 #33350
    config.lr = 0.1 #[20, 28, 32]  [28,38,46]

    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [28, 38, 46] if m - 1 <= epoch])
    config.lr_func = lr_step_func

