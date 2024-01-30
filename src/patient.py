class Patient:
    def __init__(self, annotations=None):
        if annotations is None:
            annotations = []
        self.annotations = annotations


class Patient207(Patient):
    def __init__(self):
        super().__init__(['L', 'A', 'V', '!', 'E'])


class Patient209:
    def __init__(self):
        super().__init__(['A'])
