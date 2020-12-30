try:
    from google.colab import drive

    drive.mount('/content/drive', force_remount=True)
    COLAB = True
    print("Note: using Google CoLab")
#     %tensorflow_version 2.x
except:
    print("Note: not using Google CoLab")
    COLAB = False

if COLAB:
    PROJECT_ROOT = "/content/drive/My Drive/Colab Notebooks/Generative Deep Learning - kuboko"
else:
    PROJECT_ROOT = "../"

LIB_PATH = PROJECT_ROOT

import sys

if not LIB_PATH in sys.path:
    sys.path.append(LIB_PATH)
    print(LIB_PATH + ' has been added to sys.path')

import os
import matplotlib.pyplot as plt

from models.GAN import GAN
from utils import io_utils as io
import yaml

with open(os.path.join(PROJECT_ROOT, "config.yml"), "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

mode = 'build'  # 'load' #

import os

"""## data"""

exec = cfg['exec']
io.cfg = cfg['io']
io.project_root = PROJECT_ROOT
RUN_FOLDER = io.prepare_run_folders()
(x_train, y_train) = io.load_camel_data()

# x_train.shape

# plt.imshow(x_train[0,:,:,0], cmap = 'gray')

"""## architecture"""

gan = GAN(input_dim=(28, 28, 1)
          , discriminator_conv_filters=[64, 64, 128, 128]
          , discriminator_conv_kernel_size=[5, 5, 5, 5]
          , discriminator_conv_strides=[2, 2, 2, 1]
          , discriminator_batch_norm_momentum=None
          , discriminator_activation='relu'
          , discriminator_dropout_rate=0.4
          , discriminator_learning_rate=0.0008
          , generator_initial_dense_layer_size=(7, 7, 64)
          , generator_upsample=[2, 2, 1, 1]
          , generator_conv_filters=[128, 64, 64, 1]
          , generator_conv_kernel_size=[5, 5, 5, 5]
          , generator_conv_strides=[1, 1, 1, 1]
          , generator_batch_norm_momentum=0.9
          , generator_activation='relu'
          , generator_dropout_rate=None
          , generator_learning_rate=0.0004
          , optimiser='rmsprop'
          , z_dim=100
          )

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

gan.discriminator.summary()

gan.generator.summary()

"""## training"""

BATCH_SIZE = 100
EPOCHS = 100
PRINT_EVERY_N_BATCHES = 10

gan.train(
    x_train
    , batch_size=BATCH_SIZE
    , epochs=EPOCHS
    , run_folder=RUN_FOLDER
    , print_every_n_batches=PRINT_EVERY_N_BATCHES
)

fig = plt.figure()
plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

plt.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25)
plt.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25)
plt.plot([x[0] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.xlim(0, 2000)
plt.ylim(0, 2)

plt.show()

fig = plt.figure()
plt.plot([x[3] for x in gan.d_losses], color='black', linewidth=0.25)
plt.plot([x[4] for x in gan.d_losses], color='green', linewidth=0.25)
plt.plot([x[5] for x in gan.d_losses], color='red', linewidth=0.25)
plt.plot([x[1] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('accuracy', fontsize=16)

plt.xlim(0, 2000)

plt.show()
