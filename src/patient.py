class Patient:
    def __init__(self, annotations):
        self.annotations = annotations


class Patient207(Patient):
    def __init__(self):
        super().__init__(['L', 'A', 'V', '!', 'E'])


class Patient209(Patient):
    def __init__(self):
        super().__init__(['L', 'A', 'V', '!', 'E'])
