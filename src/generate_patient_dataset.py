import argparse
import csv
from scipy.signal import butter, filtfilt

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
        readings = list(self.readings.values())
        ml_ii = [int(reading.ml_ii) for reading in readings]
        v_1 = [int(reading.v_1) for reading in readings]

        ml_ii_filtered = self._butter_filter(ml_ii)
        v_1_filtered = self._butter_filter(v_1)

        zipped_ecg_reading = list(zip(ml_ii_filtered, v_1_filtered))
        # (ml_ii, v_1)
        zipped_sample_ecg_reading = list(zip(self.readings.keys(), zipped_ecg_reading))
        # (sample_number, (filtered_ml_ii, f_v_1))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset for a patient')
    parser.add_argument('patient_id', type=str, help='The patient ID to generate a dataset for')
    parser.add_argument('path', type=str, help='The path to the data folder')
    args = parser.parse_args()

    data = GeneratePatientDataset(args.patient_id, args.path)
    print(len(list(data.filtered_mapping.keys())[0]))
