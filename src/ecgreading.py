class ECGReading:
    def __init__(self, ml_ii: str, v_1: str):
        self.ml_ii = ml_ii
        self.v_1 = v_1

    def __str__(self):
        return f"(MLII: {self.ml_ii}, V1: {self.v_1})"

    def __repr__(self):
        return self.__str__()