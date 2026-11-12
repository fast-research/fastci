def unobserved_activation(value):
    return max(0, value)


def activation_wrapper(value):
    return unobserved_activation(value)


def linear(values):
    return [activation_wrapper(value * 2) for value in values]


def never_executed_hook(value):
    return value
