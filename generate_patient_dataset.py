import argparse
import csv


class ECGReading:
    def __init__(self, ml_ii: str, v_1: str):
        self.ml_ii = ml_ii
        self.v_1 = v_1


class Annotation:
    def __init__(self, sample_number: str, time: str, annotation: str):
        self.sample_number = sample_number
        self.time = time
        self.annotation = annotation


class GeneratePatientDataset:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.path = 'data/' + patient_id

        self.FEATURES = ['MLII', 'V1']

        self.readings: dict[str, ECGReading] = self.sample_number_to_reading()
        self.annotations: dict[str, Annotation] = self.sample_number_to_annotation()

        # self.display_readings()
        self.display_annotations()

    def sample_number_to_reading(self) -> dict[str, ECGReading]:
        reading_dict = dict()

        with open(self.path + f'/{self.patient_id}.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                sample_number, ml_ii, v_1 = row
                reading_dict[sample_number] = ECGReading(ml_ii, v_1)

        return reading_dict

    def sample_number_to_annotation(self) -> dict[str, Annotation]:
        annotation_dict = dict()

        with open(self.path + f'/{self.patient_id}annotations.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                time, sample_number, annotation = row
                annotation_dict[sample_number] = Annotation(sample_number, time, annotation)

        return annotation_dict

    def display_readings(self):
        for sample, reading in self.readings.items():
            print(sample, reading.ml_ii, reading.v_1)

    def display_annotations(self):
        for sample, annotation in self.annotations.items():
            print(sample, annotation.time, annotation.annotation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset for a patient')
    parser.add_argument('patient_id', type=str, help='The patient ID to generate a dataset for')
    args = parser.parse_args()

    GeneratePatientDataset(args.patient_id)
