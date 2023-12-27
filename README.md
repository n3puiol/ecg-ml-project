# ecg-ml-project

## Prerequisites

### Python

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data

```bash
mkdir data
cd data
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
```

Make sure that the data is structured as follows:

```bash
data/mitdb/
├── 100.atr
├── 100.dat
├── 100.hea
...
├── 234.atr
├── 234.dat
└── 234.hea
```

## Usage

### Prepare data

In order to train the model, you first need to prepare the data. 
This can be done by running the following command (this takes a while):

```bash
```bash
python create_dataset.py
```

### Train model

In order to train the model, you can simply run the following command:
```bash
python train.py
```

Right now the model used is an implementation of an existing model (see baseline model for more info).

Also when training model, it will be saved every ten steps in the folder `models/`.



