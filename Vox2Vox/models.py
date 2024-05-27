from keras import regularizers
from keras.models import Model
from keras.layers import Input, Conv2D, Conv3D, Conv3DTranspose, Dropout, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation, Flatten, Dense, LeakyReLU, Attention, Multiply, Add


def getGenerator(inputShape, f, filterShape, dropout_rate, batchnorm=True, l2regularizer=0.0):
    inputs = Input(shape=inputShape, dtype='float32')

    # Encoder
    dcd1 = getDoubleConvolutionBlock(inputs=inputs, filters=f[0], filterShape=filterShape, batchnorm=batchnorm,
                                     dropout_rate=dropout_rate, l2_strength=l2regularizer)
    p1 = MaxPooling3D((2, 2, 2))(dcd1)

    dcd2 = getDoubleConvolutionBlock(inputs=p1, filters=f[1], filterShape=filterShape, batchnorm=batchnorm,
                                     dropout_rate=dropout_rate, l2_strength=l2regularizer)
    p2 = MaxPooling3D((2, 2, 2))(dcd2)

    dcd3 = getDoubleConvolutionBlock(inputs=p2, filters=f[2], filterShape=filterShape, batchnorm=batchnorm,
                                     dropout_rate=dropout_rate, l2_strength=l2regularizer)
    p3 = MaxPooling3D((2, 2, 2))(dcd3)

    # Encoded
    dcb = getDoubleConvolutionBlock(inputs=p3, filters=f[2], filterShape=filterShape, batchnorm=batchnorm,
                                    dropout_rate=dropout_rate, l2_strength=l2regularizer)

    # Decoder
    u3 = Conv3DTranspose(f[2], filterShape, strides=(2, 2, 2), padding='same',
                         kernel_regularizer=regularizers.l2(l2regularizer))(dcb)
    concat3 = concatenate([u3, dcd3])

    dcu3 = getDoubleConvolutionBlock(inputs=concat3, filters=f[2], filterShape=filterShape, batchnorm=batchnorm,
                                     dropout_rate=dropout_rate, l2_strength=l2regularizer)
    u2 = Conv3DTranspose(f[1], filterShape, strides=(2, 2, 2), padding='same',
                         kernel_regularizer=regularizers.l2(l2regularizer))(dcu3)
    concat2 = concatenate([u2, dcd2])

    dcu2 = getDoubleConvolutionBlock(inputs=concat2, filters=f[1], filterShape=filterShape, batchnorm=batchnorm,
                                     dropout_rate=dropout_rate, l2_strength=l2regularizer)
    u1 = Conv3DTranspose(f[0], filterShape, strides=(2, 2, 2), padding='same',
                         kernel_regularizer=regularizers.l2(l2regularizer))(dcu2)
    concat1 = concatenate([u1, dcd1])

    dcu1 = getDoubleConvolutionBlock(inputs=concat1, filters=f[0], filterShape=filterShape, batchnorm=batchnorm,
                                     dropout_rate=dropout_rate, l2_strength=l2regularizer)

    decoded = Conv3D(1, 1, activation='tanh', padding='same', dtype='float32')(dcu1)

    return Model(inputs, decoded, name='Generator')


def getDiscriminator(inputShape, f, filterShape, dropout_rate, batchnorm=True, l2regularizer=0.0):
    ImageA = Input(shape=inputShape, dtype='float32')
    ImageB = Input(shape=inputShape, dtype='float32')

    combined_imgs = concatenate([ImageA, ImageB],axis=-1)

    # Discriminator
    d1 = getConvolutionBlock(inputs=combined_imgs, filters=f[0], filterShape=filterShape, batchnorm=batchnorm,
                             dropout_rate=dropout_rate, l2_strength=l2regularizer)
    pd1 = MaxPooling3D((2, 2, 2))(d1)

    d2 = getConvolutionBlock(inputs=pd1, filters=f[1], filterShape=filterShape, batchnorm=batchnorm,
                             dropout_rate=dropout_rate, l2_strength=l2regularizer)
    pd2 = MaxPooling3D((2, 2, 2))(d2)

    d3 = getConvolutionBlock(inputs=pd2, filters=f[2], filterShape=filterShape, batchnorm=batchnorm,
                             dropout_rate=dropout_rate, l2_strength=l2regularizer)
    pd3 = MaxPooling3D((2, 2, 2))(d3)

    d4 = getConvolutionBlock(inputs=pd3, filters=f[3], filterShape=filterShape, batchnorm=batchnorm,
                             dropout_rate=dropout_rate, l2_strength=l2regularizer)
    pd4 = MaxPooling3D((2, 2, 2))(d4)

    # Inspired and adapted from https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
    validity = Conv3D(1, kernel_size=4, strides=1, padding='same')(pd4)

    return Model([ImageA, ImageB], validity, name='Discriminator')


def getVox3Vox(generator, discriminator):
    input_shape = generator.input_shape[1:]

    imgA, imgB = Input(shape=input_shape), Input(shape=input_shape)  # In this case, both same
    fakeA = generator(imgB)  # fake_A = self.generator(img_B)

    discriminator.trainable = False
    valid = discriminator([fakeA, imgB])

    gan = Model(inputs=[imgA, imgB], outputs=[valid, fakeA], name='Vox3Vox')
    return gan


def getConvolutionBlock(inputs, filters, filterShape, dropout_rate, batchnorm=True, l2_strength=0.0):
    x = Conv3D(filters, filterShape, padding='same', kernel_regularizer=regularizers.l2(l2_strength))(inputs)
    if batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x


def getDoubleConvolutionBlock(inputs, filters, filterShape, dropout_rate, batchnorm=True, l2_strength=0.0):
    x = inputs
    for _ in range(2):  # Perform two times
        x = Conv3D(filters, filterShape, padding='same', kernel_regularizer=regularizers.l2(l2_strength))(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
    return x

