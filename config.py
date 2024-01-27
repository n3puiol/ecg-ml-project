class Config:
    def __init__(self):
        self.input_size = 256
        self.feature = "MLII"
        self.filter_length = 32
        self.kernel_size = 16
        self.drop_rate = 0.2
        self.epochs = 10
        self.batch = 256
        self.patience = 10
        self.classes = ['N', 'V', '/', 'A', 'F', '~']