from modules.mathwrapper import RandomInitializer as rint


class LayerUnit:
    def __init__(self, input):
        self.input = input
        self.weight = rint.generate_from_input(x, )