from framework.ops import linear


class Encoder:
    def __init__(self, width=4):
        self.validate_config(width)
        self.width = width

    def validate_config(self, width):
        assert width > 0

    def forward(self, values, layers):
        result = list(values)
        for _ in range(layers):
            result = linear(result)
        return result


def validate_long_sequence(sequence):
    assert sequence >= 1024


def validate_checkpoint_shape(layers):
    assert layers == 4
