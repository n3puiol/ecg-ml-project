import argparse
import csv

from src.patient import Patient, Patient207, Patient209
from src.annotation import Annotation
from src.ecgreading import ECGReading


class GeneratePatientDataset:

    def __init__(self, patient_id):

        if patient_id not in ['207', '209']:
            raise ValueError("Patient ID must be 207 or 209")

        self.patient_id = patient_id

        self.patient: Patient \
            = Patient209() if patient_id == '209' else Patient207()

        self.path = 'data/' + patient_id

        self.readings: dict[str, ECGReading] = self._sample_number_to_reading()
        self.annotations: dict[str, Annotation] = self._sample_number_to_annotation()

        self.mapping: dict[tuple, Annotation] = self._ecg_reading_to_annotation_map()

    def _sample_number_to_reading(self) -> dict[str, ECGReading]:
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

    def _sample_number_to_annotation(self) -> dict[str, Annotation]:
        annotation_dict = dict()
        i = 0
        with open(self.path + f'/{self.patient_id}annotations.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                time, sample_number, annotation = row

                if annotation not in self.patient.annotations:
                    continue

                annotation_dict[sample_number] = Annotation(sample_number, time, annotation)

        return annotation_dict

    def _ecg_reading_to_annotation_map(self) -> dict[tuple, Annotation]:
        mapping = dict()
        seq_numbers = [int(sample_number) for sample_number in self.annotations.keys()]

        current_list = []

        for sample_number, ecg_reading in self.readings.items():
            if len(seq_numbers) == 0:
                break

            if int(sample_number) <= seq_numbers[0]:
                current_list.append(ecg_reading)
                continue

            seq_numbers.pop(0)
            mapping[tuple(current_list)] = self.annotations[str(int(sample_number) - 1)]
            current_list = []

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
    args = parser.parse_args()

    GeneratePatientDataset(args.patient_id)
