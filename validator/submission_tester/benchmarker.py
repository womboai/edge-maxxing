class Benchmarker:
    submissions: list[tuple[str, str]]
    scores: dict[str, ]
    done: bool

    def __init__(self):
        self.submissions = []
        self.scores = {}
        self.done = True

    def start_benchmarking(self, submissions: list[tuple[str, str]]):
        self.submissions = submissions
        self.scores = {}
        self.done = False
