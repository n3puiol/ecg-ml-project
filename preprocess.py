from generate_patient_dataset import GeneratePatientDataset

test = GeneratePatientDataset("207")

readings = test.readings
annotations = test.annotations


for sample, reading in readings.items():
    print(sample, reading.ml_ii, reading.v_1)