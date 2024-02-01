import argparse
import csv
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt

from patient import Patient, Patient207, Patient209
from annotation import Annotation
from ecgreading import ECGReading


class GeneratePatientDataset:

    def __init__(self, patient_id, path):

        if patient_id not in ['207', '209']:
            raise ValueError("Patient ID must be 207 or 209")

        self.patient_id = patient_id

        self.patient: Patient \
            = Patient209() if patient_id == '209' else Patient207()

        self.path = path + patient_id

        self.readings: dict[str, ECGReading] = self._sample_number_to_reading_map()
        self.all_readings = list(self.readings.values())
        self.all_ml_ii = [int(reading.ml_ii) for reading in self.all_readings]
        self.all_v_1 = [int(reading.v_1) for reading in self.all_readings]
        self.annotations: dict[tuple, Annotation] = self._sample_number_to_annotation_map()

        self.mapping: dict[tuple, Annotation] = self._ecg_reading_to_annotation_map(self.readings.items())
        self.filtered_mapping: dict[tuple, Annotation] = self._butter_filtered_reading()

    def _sample_number_to_reading_map(self) -> dict[str, ECGReading]:
        reading_dict = dict()
        i = 0
        with open(self.path + f'/{self.patient_id}.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                sample_number, ml_ii, v_1 = row
                reading_dict[sample_number] = ECGReading(ml_ii, v_1)

        return reading_dict

    def _sample_number_to_annotation_map(self) -> dict[tuple, Annotation]:
        annotation_dict = dict()
        i = 0
        previous_sample_num = 0
        with open(self.path + f'/{self.patient_id}annotations.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                time, sample_number, annotation = row

                if annotation not in self.patient.annotations:
                    previous_sample_num = sample_number
                    continue

                annotation_dict[(str(int(previous_sample_num) + 1), sample_number)] = Annotation(sample_number, time, annotation)
                previous_sample_num = sample_number

        return annotation_dict

    @staticmethod
    def _butter_filter(sequence):
        fs = 360
        nyquist = 0.5 * fs
        low = 0.4 / nyquist
        high = 45 / nyquist

        b, a = butter(N=3, Wn=[low, high], btype='band')
        return filtfilt(b, a, sequence)

    def _butter_filtered_reading(self):
        ml_ii_filtered = self._butter_filter(self.all_ml_ii)
        v_1_filtered = self._butter_filter(self.all_v_1)

        zipped_ecg_reading = list(zip(ml_ii_filtered, v_1_filtered))
        zipped_sample_ecg_reading = list(zip(self.readings.keys(), zipped_ecg_reading))

        return self._ecg_reading_to_annotation_map(zipped_sample_ecg_reading)

    def _ecg_reading_to_annotation_map(self, readings) -> dict[tuple, Annotation]:
        mapping = dict()
        seq_range = [sample_number_range for sample_number_range in self.annotations.keys()]

        current_list = []

        for sample_number, ecg_reading in readings:
            if len(seq_range) == 0:
                break

            lower, upper = 0, 1
            if int(seq_range[0][lower]) <= int(sample_number) <= int(seq_range[0][upper]):
                current_list.append(ecg_reading)
            elif int(sample_number) > int(seq_range[0][upper]):
                if len(current_list) > 0:
                    mapping[tuple(current_list)] = self.annotations[seq_range[0]]
                    current_list = []
                seq_range.pop(0)
            else:
                continue

        return mapping

    # Debug

    def display_readings(self):
        for sample, reading in self.readings.items():
            print(sample, reading.ml_ii, reading.v_1)

    def display_annotations(self):
        for sample, annotation in self.annotations.items():
            print(sample, annotation.time, annotation.annotation)

    def display_mapping(self):
        for key, val in self.mapping.items():
            print(key)
            print(val)

            print("========")

    def plot_all_readings(self, up_to=None, save=False):
        if up_to is None:
            up_to = len(self.all_ml_ii)

        plt.plot(self.all_ml_ii[:up_to], color='r')
        plt.title(f'Patient {self.patient_id} MLII Readings (up to {up_to} readings)')
        plt.xlabel('Sample Number')
        plt.ylabel('MLII Reading (mV)')
        if save:
            plt.savefig(f'figs/patient_{self.patient_id}_ml_ii.png')
        plt.show()

        plt.plot(self.all_v_1[:up_to], color='c')
        plt.title(f'Patient {self.patient_id} V1 Readings (up to {up_to} readings)')
        plt.xlabel('Sample Number')
        plt.ylabel('V1 Reading (mV)')
        if save:
            plt.savefig(f'figs/patient_{self.patient_id}_v_1.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset for a patient')
    parser.add_argument('patient_id', type=str, help='The patient ID to generate a dataset for')
    parser.add_argument('path', type=str, help='The path to the data folder')
    parser.add_argument('--plot', action='store_true', help='Plot the readings')
    parser.add_argument('--save', action='store_true', help='Save the plots')
    args = parser.parse_args()

    data = GeneratePatientDataset(args.patient_id, args.path)
    if args.plot or args.save:
        data.plot_all_readings(3000, args.save)
