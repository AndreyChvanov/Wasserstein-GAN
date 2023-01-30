from easydict import EasyDict



cfg = EasyDict()
cfg.datapath = '../data/MNIST/raw'

# Number of workers for dataloader

# Batch size during training
cfg.batch_size = 8

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
cfg.image_size = 64

# Number of channels in the training images. For color images this is 3
cfg.nc = 1

# Size of z latent vector (i.e. size of generator input)
cfg.nz = 100

# Size of feature maps in generator
cfg.ngf = 64

# Size of feature maps in discriminator
cfg.ndf = 64

# Number of training epochs
cfg.steps = 500

# Learning rate for optimizers
cfg.lr = 0.0001

# Beta1 hyperparam for Adam optimizers
cfg.beta1 = 0.0

cfg.ngpu = 1

cfg.discriminator_steps = 5
cfg.penalty_lambda = 10