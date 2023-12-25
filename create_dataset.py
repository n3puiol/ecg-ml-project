from wfdb import rdrecord, rdann
from sklearn import preprocessing
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import h5py


class MitDBDataset:
    def __init__(self, path: str, load: bool = False):
        self.NUMS = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114',
                     '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205',
                     '207', '208', '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223',
                     '228', '230', '231', '232', '233', '234']
        # self.NUMS = ['100', '101', '102']
        self.FEATURES = ['MLII', 'V1', 'V2', 'V4', 'V5']
        self.TEST_SET = ['101', '105', '114', '118', '124', '201', '210', '217']
        self.INPUT_SIZE = 256
        self.TRAIN_SET = [x for x in self.NUMS if x not in self.TEST_SET]
        self.path = path
        if load:
            self.load_dataset()
        else:
            self.create_dataset()

    def create_dataset(self):
        classes = ['N', 'V', '/', 'A', 'F', '~']  # ,'L','R',f','j','E','a']#,'J','Q','e','S']
        datadict, datalabel = dict(), dict()

        for feature in self.FEATURES:
            datadict[feature] = list()
            datalabel[feature] = list()

        for num in self.NUMS:
            record = rdrecord(self.path + num, smooth_frames=True)

            signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 0])).tolist()
            signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 1])).tolist()

            peaks, _ = find_peaks(signals0, distance=150)

            feature0, feature1 = record.sig_name[0], record.sig_name[1]

            # skip a first peak to have enough range of the sample
            for peak in tqdm(peaks[1:-1]):
                start, end = peak - self.INPUT_SIZE // 2, peak + self.INPUT_SIZE // 2
                ann = rdann(self.path + num, extension='atr', sampfrom=start, sampto=end,
                            return_label_elements=['symbol'])

                if len(ann.symbol) == 1 and (ann.symbol[0] in classes) and (
                        ann.symbol[0] != "N" or np.random.random() < 0.15):
                    y = [0] * len(classes)
                    y[classes.index(ann.symbol[0])] = 1
                    datalabel[feature0].append(y)
                    datalabel[feature1].append(y)
                    datadict[feature0].append(signals0[start:end])
                    datadict[feature1].append(signals1[start:end])
        self.save_dataset(datalabel, datadict)

    def save_dataset(self, datalabel, datadict):
        hf = h5py.File(self.path + 'dataset.hdf5', 'w')
        for feature in self.FEATURES:
            hf.create_dataset(feature + '_data', data=datadict[feature])
            hf.create_dataset(feature + '_label', data=datalabel[feature])

    def load_dataset(self):
        hf = h5py.File(self.path + 'dataset.hdf5', 'r')
        data = dict()
        label = dict()
        for feature in self.FEATURES:
            data[feature] = hf[feature + '_data'][:]
            label[feature] = hf[feature + '_label'][:]
        hf.close()
        return data, label


if __name__ == "__main__":
    dataset = MitDBDataset('./data/mitdb/1.0.0/', load=True)
