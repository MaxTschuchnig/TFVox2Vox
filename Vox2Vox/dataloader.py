# Inspired and adapted from https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
import SimpleITK as sitk
import numpy as np
import torchio as tio
import copy

from glob import glob

class DataLoader():
    def __init__(self, pathA, extensionA, splitterA, pathB, extensionB, splitterB, rngThreshold, downscale=False):
        self.pathA = pathA
        self.pathB = pathB

        self.extensionB = extensionB
        self.extensionA = extensionA

        self.splitterA = splitterA
        self.splitterB = splitterB

        self.rngThreshold = rngThreshold
        self.downscale = downscale

        # set train/val datapaths
        self.pathsA = glob(f'{self.pathA}/*.{self.extensionA}')
        self.pathsB = glob(f'{self.pathB}/*.{self.extensionB}')
        self.pathsA, self.pathsB = \
            sorted(self.pathsA, key=lambda x: self.splitIndexA(x), reverse=False), \
            sorted(self.pathsB, key=lambda x: self.splitIndexB(x), reverse=False)


    def load_data(self, batch_size=1, padding=[(0,0),(0,0),(0,0),(0,0)], augment=False):
        randomIndices = np.random.choice(len(self.pathsA), size=batch_size, replace=False)

        imgs_A = []
        imgs_B = []
        for index in randomIndices:
            A = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(self.pathsA[index])), axis=-1)
            B = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(self.pathsB[index])), axis=-1)

            A = self.process(image=A, padding=padding)
            B = self.process(image=B, padding=padding)

            if augment and np.random.rand() < self.rngThreshold:
                A, B = self.augment(A, B)

            imgs_A.append(A)
            imgs_B.append(B)

        return np.array(imgs_A), np.array(imgs_B), randomIndices

    def load_batch(self, batch_size=1, padding=[(0,0),(0,0),(0,0),(0,0)], augment=False, shuffle=False):
        self.n_batches = int(len(self.pathsA) / batch_size)
        print(self.n_batches)

        CPathsA = copy.deepcopy(self.pathsA)
        CPathsB = copy.deepcopy(self.pathsB)

        indices = np.arange(len(CPathsA))
        if shuffle:
            np.random.shuffle(indices)
            CPathsA = [CPathsA[i] for i in indices]
            CPathsB = [CPathsB[i] for i in indices]

        for i in range(self.n_batches):
            batchA = CPathsA[i*batch_size:(i+1)*batch_size]
            batchB = CPathsB[i*batch_size:(i+1)*batch_size]
            cIndices = indices[i*batch_size:(i+1)*batch_size]
            imgsA, imgsB = [], []
            sitkA, sitkB = [], []
            for (imageA, imageB) in zip(batchA, batchB):
                A = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(imageA)), axis=-1)
                B = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(imageB)), axis=-1)

                A = self.process(image=A, padding=padding)
                B = self.process(image=B, padding=padding)

                if augment and np.random.rand() < self.rngThreshold:
                    A, B = self.augment(A, B)

                imgsA.append(A)
                imgsB.append(B)
                sitkA.append(sitk.ReadImage(imageA))
                sitkB.append(sitk.ReadImage(imageB))

            yield np.array(imgsA), np.array(imgsB), sitkA, sitkB, cIndices

    def process(self, image, padding):
        image = np.pad(image, padding, mode='constant', constant_values=0)
        image = self.normalize(data=image)

        if self.downscale:
            image = image[::2, ::2, ::2]  # downsample by factor 2
        return image

    def splitIndexA(self, path):
        return int(path.split(self.splitterA)[1].split('.' + self.extensionA)[0])

    def splitIndexB(self, path):
        return int(path.split(self.splitterB)[1].split('.' + self.extensionB)[0])

    def augment(self, cbct, ct):
        allTransformsDict = [
            tio.RandomFlip(axes=('Anterior',),flip_probability=1),
            tio.RandomFlip(axes=('Posterior',),flip_probability=1),
            tio.RandomFlip(axes=('Inferior',),flip_probability=1),
            tio.RandomFlip(axes=('Superior',),flip_probability=1)
        ]

        fromTransformsDict = [
            tio.RandomBiasField(),
            tio.RandomBlur(),
            tio.RandomNoise(),
            tio.RandomGamma()
        ]

        # Create TorchIO subjects
        subject = tio.Subject(
            cbct=tio.Image(tensor=cbct),
            ct=tio.Image(tensor=ct)
        )

        # Apply transforms to both CBCT and CT
        for transform in allTransformsDict:
            if np.random.rand() > self.rngThreshold:
                subject = transform(subject)

        # Retrieve the augmented image tensors
        augmented_cbct_tensor = subject['cbct'].data
        augmented_ct_tensor = subject['ct'].data

        # Convert tensors to NumPy arrays
        augmented_cbct = augmented_cbct_tensor.numpy()
        augmented_ct = augmented_ct_tensor.numpy()

        subject = tio.Subject(
            cbct=tio.Image(tensor=augmented_cbct)
        )
        for transform in fromTransformsDict:
            if np.random.rand() > self.rngThreshold:
                subject = transform(subject)
        augmented_cbct_tensor = subject['cbct'].data
        augmented_cbct = augmented_cbct_tensor.numpy()

        return augmented_cbct, augmented_ct

    def normalize(self, data, min_std=1e-7):
        mean = data.mean()
        std = data.std()

        # Check for small standard deviation which can happen in very noisy data
        if abs(std) > min_std:
            data = (data - mean) / std
        else:
            data = (data - mean)

        return data

