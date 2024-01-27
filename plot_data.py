from matplotlib import pyplot as plt

from create_dataset import MitDBDataset


def plot():
    mit_dataset = MitDBDataset('./data/mitdb/1.0.0/', load=True)
    X, y = mit_dataset.load_dataset(feature='MLII')
    plt.plot(X[1])
    plt.show()


if __name__ == '__main__':
    plot()
