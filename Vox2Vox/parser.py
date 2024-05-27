import argparse

def parse():
    parser = argparse.ArgumentParser(description="3d pix2pix version")
    parser.add_argument("-pathA", action="store", type=str, default="/data/train/volumesDomainA",
                        help="Path to the train data folder of domain A (source)")
    parser.add_argument("-pathB", action="store", type=str, default="/data/train/volumesDomainB",
                        help="Path to the train data folder of domain B (condition)")
    parser.add_argument("-extensionA", action="store", type=str, default="nii",
                        help="Extention of domain A (source) data (nii default)")
    parser.add_argument("-extensionB", action="store", type=str, default="nii",
                        help="Extention of domain B (condition) data (nii default)")
    parser.add_argument("-splitterA", action="store", type=str, default="A-",
                        help="Splitter to obtain volume id. Assumed to be int, linking domains A and B. "
                             "If paths of volumes (domain A) are e.g. path/A-1, path/A-2, ... splitter should be 'A-'")
    parser.add_argument("-splitterB", action="store", type=str, default="B-",
                        help="Splitter to obtain volume id. Assumed to be int, linking domains A and B. "
                             "If paths of volumes (domain B) are e.g. path/B-1, path/B-2, ... splitter should be 'B-'")

    parser.add_argument("-dimensions", nargs='+', action="store", type=int, required=True,
                        help="Input shape of the model in rows, cols, height. E.g.: -dimensions 256 256 64")
    parser.add_argument("-filterN", nargs='+', action="store", type=int, default=[64, 128, 256, 512],
                        help="Amounts of filters per layer. Needs to be 4 values E.g.: -filterN 32 64 128 256")
    parser.add_argument('-filterSize', action='store', type=int, default=3,
                        help='Size of (3d) convolutional filters. 3 on default.')
    parser.add_argument("-downscale", action="store_true", required=False,
                        help="Downscale once. Sometimes needed to fit data to GPU (A6000 gets you pretty far).")

    parser.add_argument('-learnRate', action='store', type=float, default=0.0002,
                        help='Learning rate for training of model(s). Default 0.0002')
    parser.add_argument('-l2Regularization', action='store', type=float, default=0.2,
                        help='L2 based regularization for training of model(s). Default 0.2')
    parser.add_argument('-dropout', action='store', type=float, default=0.0,
                        help='Dropout rate for training of discriminator. Default 0.2')
    parser.add_argument('-epochs', action='store', type=int, default=25,
                        help='Amount of training epochs. Default 25')
    parser.add_argument('-batchSize', action='store', type=int, default=1,
                        help='Batch size. Default 1')
    parser.add_argument('-trainGenEach', action='store', type=int, default=5,
                        help='How often the Discriminator is trained for each Generator training. Default 5')
    parser.add_argument('-trivialThreshold', action='store', type=float, default=0.9,
                        help='Threshold to keep Disrcriminator from fitting a simple solution (all fake/reak). '
                             'Default 0.9')
    parser.add_argument('-augRngThreshold', action='store', type=float, default=0.5,
                        help='Threshold to apply random data augmentation. Default 0.5')

    parser.add_argument('-debug', action='store', type=int, default=3,
                        help='Debug level. The higher, the more debugging info. Default 3')
    parser.add_argument("-padding", nargs='+', action="store", type=int, default=[0, 0, 0, 0],
                        help="Padding to apply to images to fit halving 3 times for 3 times downsampling in Encoder. "
                             "Needs to be 4 values E.g.: if image is shaped 180 118 182 padding should be 4 2 2 0")
    parser.add_argument("-name", action="store", type=str, default='generator',
                        help="name for saved generator. useful for automation scripts")
    return parser.parse_args()