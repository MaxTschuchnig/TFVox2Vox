# Inspired and adapted from https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import keras as K

import dataloader
import parser
import models

arguments = parser.parse()

# image information
pathA, pathB = arguments.pathA, arguments.pathB
extensionA, extensionB = arguments.extensionA , arguments.extensionB
splitterA, splitterB = arguments.splitterA, arguments.splitterB
img_rows, img_cols, img_height = arguments.dimensions[0], arguments.dimensions[1], arguments.dimensions[2]
padding=arguments.padding

# network setup
f=arguments.filterN
filterShape=arguments.filterSize

# learning setup
epochs = arguments.epochs
batch_size = arguments.batchSize
rng_threshold = arguments.augRngThreshold
learningRate = arguments.learnRate
dropoutRate = arguments.dropout
l2regularizer = arguments.l2Regularization

# further setup
debugLevel = arguments.debug
downscale = False
if arguments.downscale:
    downscale = True
name = arguments.name

# Getting discPatch
patch_r = int(img_rows / 2**4)
patch_c = int(img_cols / 2**4)
patch_h = int(img_height / 2**4)
disc_patch = (patch_r, patch_c, patch_h, 1)

# Getting shape
inputShape = (img_rows, img_cols, img_height, 1)

trainGenEach = arguments.trainGenEach
trivialThreshold = arguments.trivialThreshold

# dataloader
dataloader = dataloader.DataLoader(pathA=pathA,
                                   extensionA=extensionA,
                                   splitterA=splitterA,
                                   pathB=pathB,
                                   extensionB=extensionB,
                                   splitterB=splitterB,
                                   rngThreshold=rng_threshold,
                                   downscale=downscale)

# Compile discriminator
discriminator = models.getDiscriminator(inputShape=inputShape, f=f, filterShape=filterShape, dropout_rate=dropoutRate,
                                        batchnorm=True, l2regularizer=l2regularizer)
discriminator.compile(loss='bce', optimizer=Adam(learning_rate=learningRate, beta_1=0.5), metrics=['accuracy'])
generator = models.getGenerator(inputShape=inputShape, f=f, filterShape=filterShape, dropout_rate=dropoutRate,
                                batchnorm=True, l2regularizer=l2regularizer)
generator.compile(loss=['mae'], optimizer=Adam(learning_rate=learningRate, beta_1=0.5))
combined = models.getVox3Vox(generator=generator, discriminator=discriminator)
combined.compile(loss=['bce', 'mae'], loss_weights=[1, 100], optimizer=Adam(learning_rate=learningRate, beta_1=0.5))

if debugLevel > 1:
    print(discriminator.summary())
    print(generator.summary())
    print(combined.summary())

if debugLevel > 3:
    print(f'Starting training for {epochs} epochs with a batch size of {batch_size} '
          f'and learning rate of {learningRate}')

def train(epochs, batch_size=1, sample_interval=50):
    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)

    dLosses = []
    gLoss1 = []
    gLoss2 = []

    for epoch in range(epochs):
        d_loss = [0,0]
        for batch_i, (imgs_A, imgs_B, _, _, index) in enumerate(
                dataloader.load_batch(
                    batch_size, padding=[(padding[0], 0), (padding[1], 0), (padding[2], 0), (padding[3], 0)],
                    augment=True, shuffle=True)):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            fake_A = generator.predict(imgs_B)

            # Train the discriminators (original images = real / generated = Fake)
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators, only each steps and if the discriminator is not solving a trivial solution
            discriminator.trainable = False
            if batch_i % trainGenEach == 0 and abs(d_loss_real[1] - d_loss_fake[1]) < trivialThreshold:
                g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            dLosses.append(d_loss)
            gLoss1.append(g_loss[0])
            gLoss2.append(g_loss[1])
            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            if debugLevel > 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {batch_i}/{dataloader.n_batches}] [D loss: {d_loss[0]}, "
                      f"real acc: {100*d_loss_real[1]:.2f}, fake acc: {100*d_loss_fake[1]:.2f}, "
                      f"acc: {100*d_loss[1]:.2f}] [G loss 1: {g_loss[0]:.2f}, G loss 2: {g_loss[1]:.2f}] "
                      f"time: {elapsed_time}")

            # If at save interval => save generated image samples
            if batch_i % sample_interval == 0 and debugLevel > 0:
                sampleImages(epoch=epoch, batch_i=batch_i)

    generatorName = f'{name}.keras'
    print(f'training done, saving trained generator as {name}')

    plotLosses(dLosses=dLosses, gLoss1=gLoss1, gLoss2=gLoss2)
    generator.save(generatorName)


def sampleImages(epoch, batch_i):
    imagesA, imagesB, indices = dataloader.load_data(
        batch_size=1, padding=[(padding[0], 0), (padding[1], 0), (padding[2], 0), (padding[3], 0)], augment=False)
    fakeA = generator.predict(imagesB)
    images = [imagesA, fakeA, imagesB]

    titles = ['Condition', 'Generated', 'Original']
    fig, axs = plt.subplots(1, 3)
    for i in range(3):
        axs[i].imshow(images[i][0, :, 100, :, 0])
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    fig.savefig(f'image({indices[0]})_{epoch}_{batch_i}.png')
    plt.close()


def plotLosses(dLosses, gLoss1, gLoss2):
    plt.close('all')
    plt.plot(np.array(dLosses)[:, 0], label='Discriminator Loss', color='blue')
    plt.plot(gLoss1, label='Generator Loss 1', color='green')
    plt.plot(gLoss2, label='Generator Loss 2', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Score')
    plt.title('Training Progress')
    plt.savefig(f'training.png')
    plt.close('all')


train(epochs=epochs, batch_size=batch_size)
print('done', flush=True)
