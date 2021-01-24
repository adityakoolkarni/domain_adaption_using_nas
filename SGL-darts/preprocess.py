import pickle
import argparse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--address', type=str, default='../data/cifar-100-python/train')

args = parser.parse_args()


def unpickle(args):
    with open(args.address, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
	data = unpickle(args)
	import ipdb; ipdb.set_trace()
