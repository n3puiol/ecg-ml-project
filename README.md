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