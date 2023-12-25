from wfdb import rdrecord, rdann
from sklearn import preprocessing
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm


class MitDBDataset:
    def __init__(self, path: str):
        self.NUMS = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114',
                     '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205',
                     '207', '208', '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223',
                     '228', '230', '231', '232', '233', '234']
        self.FEATURES = ['MLII', 'V1', 'V2', 'V4', 'V5']
        self.TEST_SET = ['101', '105', '114', '118', '124', '201', '210', '217']
        self.INPUT_SIZE = 256
        self.TRAIN_SET = [x for x in self.NUMS if x not in self.TEST_SET]
        self.path = path
        self.load_dataset()

    def load_dataset(self):
        for num in self.NUMS:
            record = rdrecord(self.path + num, smooth_frames=True)
            signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 0])).tolist()
            signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 1])).tolist()
            peaks, _ = find_peaks(signals0, distance=150)
            # skip a first peak to have enough range of the sample
            for peak in tqdm(peaks[1:-1]):
                start, end = peak - self.INPUT_SIZE // 2, peak + self.INPUT_SIZE // 2
                ann = rdann(self.path + num, extension='atr', sampfrom=start, sampto=end,
                            return_label_elements=['symbol'])


if __name__ == "__main__":
    dataset = MitDBDataset('./data/mitdb/1.0.0/')
