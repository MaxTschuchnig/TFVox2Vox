# TFVox2Vox
Simple and adaptable 3d implementation of Pix2Pix in keras-tensorflow

---

Install python packages using 
- pip install -r requirements.txt

---

To use with your 3d dataset, first train the model using:
- python folderToProject/train.py -pathA=/meddata/3dData/train-A/ -pathB=/meddata/3dData/train-B/ -splitterA=A- -splitterB=B- -dimensions 184 120 184 -name generator

After training successfully, your generator can be used by the transform script:
- python folderToProject/transfer.py -pathB=/meddata/3dData/test-B/ -splitterB=B- -pathGenerator=~/generator.keras -pathOutput=/meddata/3dData/transformed/

---

Parameters for **train.py** include:
- pathA (str, default: "/data/train/volumesDomainA"): Path to the train data folder of domain A (source).
- pathB (str, default: "/data/train/volumesDomainB"): Path to the train data folder of domain B (condition).
- extensionA (str, default: "nii"): Extension of domain A (source) data.
- extensionB (str, default: "nii"): Extension of domain B (condition) data.
- splitterA (str, default: "A-"): Splitter to obtain volume id for domain A.
- splitterB (str, default: "B-"): Splitter to obtain volume id for domain B.
- dimensions (list of ints, required): Input shape of the model in rows, cols, height (e.g., -dimensions 256 256 64).
- filterN (list of ints, default: [64, 128, 256, 512]): Amounts of filters per layer.
- filterSize (int, default: 3): Size of 3D convolutional filters.
- downscale (bool, default: False): Downscale once. Sometimes needed to fit data to GPU.
- learnRate (float, default: 0.0002): Learning rate for training of model(s).
- l2Regularization (float, default: 0.2): L2 based regularization for training of model(s).
- dropout (float, default: 0.0): Dropout rate for training of discriminator.
- epochs (int, default: 25): Amount of training epochs.
- batchSize (int, default: 1): Batch size.
- trainGenEach (int, default: 5): How often the Discriminator is trained for each Generator update.
- trivialThreshold (float, default: 0.9): Threshold to keep Discriminator from fitting a simple solution (all fake/real).
- augRngThreshold (float, default: 0.5): Threshold to apply random data augmentation.
- debug (int, default: 3): Debug level. The higher, the more debugging info.
- padding (list of ints, default: [0, 0, 0, 0]): Padding to apply to images to fit halving 3 times for 3 times downsampling in Generator Encoder and 4 times in Discriminator.
- name (str, default: 'generator'): Name for saved generator, useful for automation scripts. .keras gets added automatically.

---

Parameters for **transfer.py** include:
- pathB (str, default: "/data/volumesDomainB"): Path to the data folder of domain B (condition).
- extensionB (str, default: "nii"): Extension of domain B (condition) data.
- splitterB (str, default: "B-"): Splitter to obtain volume id for domain B.
- pathGenerator (str, required): Path to and name of generator (e.g., -pathGenerator /home/generator). .keras gets added automatically.
- pathOutput (str, required): Path to output folder (e.g., -pathOutput /data/transferOut).
- downscale (bool, default: False): Downscale once. Sometimes needed to fit data to GPU.
- padding (list of ints, default: [0, 0, 0, 0]): Padding to apply to images to fit halving 3 times for 3 times downsampling in Encoder.
- batchSize (int, default: 1): Batch size.

# Demo CBCTLiTS
We used this repo to perform style transfer from the CBCTLiTS datasets (128 projections) CBCT into CT.
We used the unlabelled, paired CBCT/CT test dataset to train the model using
- ~/folderToProject/train.py -pathA=/CBCTLiTS/TESTCTAlignedToCBCT/ -pathB=/CBCTLiTS/TESTCBCTSimulated/128 -splitterA=test-volume- -splitterB=REC- -dimensions 184 120 184 -downscale -padding 4 2 2 0 -name generator128 -epochs 50 -learnRate 0.0002 -trainGenEach=2 -trivialThreshold=0.75

and then applied our trained Vox2Vox model to translate the labelled, paired CBCT/CT train data from CBCT to CT using
- python ~/folderToProject/transfer.py -pathB=/CBCTLiTS/TRAINCBCTSimulated/128 -extensionB=nii.gz -splitterB=REC- -pathGenerator=./generator128 -pathOutput=/CBCTLiTS/transformed/128/ -downscale -padding 4 2 2 0 

CBCTLiTS test data link: https://www.kaggle.com/datasets/maximiliantschuchnig/cbct-liver-and-liver-tumor-segmentation-test-data
CBCTLiTS train data link: https://www.kaggle.com/datasets/maximiliantschuchnig/cbct-liver-and-liver-tumor-segmentation-train-data

This resulted in the following results:
TODO: Add results from paper

## License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

**Non-Commercial Use Only**: While this project is licensed under the GPL-3.0, we kindly request that users respect our wish for the software to be used for non-commercial purposes only. This request is not legally enforceable under the terms of the GPL-3.0.
