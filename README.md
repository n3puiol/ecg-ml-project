# Classifying pathological heartbeats from ECG signals

I strongly recommend understanding the dataset. An explanation can be found [here](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm).

## Prerequisites

### Python

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data

Request access: https://drive.google.com/file/d/1q3IirCowDNIfSs9Yzb156yPVBjR8-C6B/view?usp=sharing

## Usage

### Prepare data

In order to train the model, you first need to prepare the data. 
This can be done by running the following command (this takes a while):

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



