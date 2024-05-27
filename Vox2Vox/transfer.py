import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import dataloader
import math

import argparse

def parse():
    parser = argparse.ArgumentParser(description="3d pix2pix version")
    parser.add_argument("-pathB", action="store", type=str, default="/data/volumesDomainB",
                        help="Path to the data folder of domain B (condition)")
    parser.add_argument("-extensionB", action="store", type=str, default="nii",
                        help="Extention of domain B (condition) data (nii default)")
    parser.add_argument("-splitterB", action="store", type=str, default="B-",
                        help="Splitter to obtain volume id. Assumed to be int, linking domains A and B. "
                             "If paths of volumes (domain B) are e.g. path/B-1, path/B-2, ... splitter should be 'B-'")

    parser.add_argument("-pathGenerator", action="store", type=str, required=True,
                        help="Path to and name of generator. E.g.: -pathGenerator /home/generator")

    parser.add_argument("-pathOutput", action="store", type=str, required=True,
                        help="Path to output folder. E.g.: -pathOutput /data/transferOut")

    parser.add_argument("-downscale", action="store_true", required=False,
                        help="Downscale once. Sometimes needed to fit data to GPU.")
    parser.add_argument("-padding", nargs='+', action="store", type=int, default=[0, 0, 0, 0],
                        help="Padding to apply to images to fit halving 3 times for 3 times downsampling in Encoder. "
                             "Needs to be 4 values E.g.: if image is shaped 180 118 182 padding should be 4 2 2 0")

    parser.add_argument('-batchSize', action='store', type=int, default=1,
                        help='Batch size. Default 1')
    return parser.parse_args()
arguments = parse()

pathB, extensionB, splitterB = arguments.pathB, arguments.extensionB, arguments.splitterB
pathGenerator = arguments.pathGenerator
outputFolder = arguments.pathOutput
downscale = arguments.downscale
batchSize = arguments.batchSize
padding = arguments.padding

dataloader = dataloader.DataLoader(pathA=pathB,
                                   extensionA=extensionB,
                                   splitterA=splitterB,
                                   pathB=pathB,
                                   extensionB=extensionB,
                                   splitterB=splitterB,
                                   rngThreshold=0,
                                   downscale=downscale)

generatorPath = f'{pathGenerator}.keras'
generator = tf.keras.models.load_model(generatorPath)

for batchI, (imgsA, imgsB, sitkA, sitkB, indices) in enumerate(
        dataloader.load_batch(
            batchSize, padding=[(padding[0], 0), (padding[1], 0), (padding[2], 0), (padding[3], 0)],
            augment=False, shuffle=False)):

    # Generate and save synthesized images
    synthesizedImages = generator.predict(imgsB)

    for i, img in enumerate(synthesizedImages):
        savePath = f'{outputFolder}/{indices[i]}.nii'

        # re-upscale
        if downscale:
            img = np.repeat(np.repeat(np.repeat(img, 2, axis=0), 2, axis=1), 2, axis=2)
        # Convert to original shape
        img = img[math.floor(padding[0]):,
              math.floor(padding[1]):,
              math.floor(padding[2]):]

        # Copying metadata
        synthesizedImage = sitk.GetImageFromArray(img)
        synthesizedImage.CopyInformation(sitkB[i])
        sitk.WriteImage(synthesizedImage, savePath)

        print(f"Saved: {savePath}")

print('done')