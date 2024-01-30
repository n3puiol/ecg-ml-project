class Annotation:
    def __init__(self, sample_number: str, time: str, annotation: str):
        self.sample_number = sample_number
        self.time = time
        self.annotation = annotation

    def __str__(self):
        return f"Sample #: {self.sample_number}, Time: {self.time}, Annotation: {self.annotation}"

    def __repr__(self):
        return self.__str__()