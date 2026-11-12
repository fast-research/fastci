from framework import checkpoint, compiler, data, distributed
from framework.model import Encoder, validate_checkpoint_shape, validate_long_sequence


def reference(values, layers):
    return [max(0, value) * (2**layers) for value in values]


def train(layers, sequence):
    values = data.load_batches()[0]
    result = Encoder().forward(values, layers)
    assert result == reference(values, layers)


def train_dp(layers, sequence):
    values = Encoder().forward(data.load_batches()[0], layers)
    assert distributed.data_parallel(values) == sum(values)


def train_tp(layers, sequence):
    values = Encoder().forward(data.load_batches()[0], layers)
    assert distributed.tensor_parallel(values) == sum(values)


def long_context(layers, sequence):
    validate_long_sequence(sequence)
    train(layers, sequence)


def checkpoint_roundtrip(layers, sequence):
    values = Encoder().forward(data.load_batches()[0], layers)
    assert checkpoint.roundtrip(values, layers) == values


def shape_guard(layers, sequence):
    train(layers, sequence)
    validate_checkpoint_shape(layers)


def compiled_train(layers, sequence):
    values = Encoder().forward(data.load_batches()[0], layers)
    assert compiler.compile_transform(sum(values), layers) == sum(values)


def mixed_precision(layers, sequence):
    values = Encoder().forward([1.5, 2.5, 3.5, 4.5], layers)
    assert values == reference([1.5, 2.5, 3.5, 4.5], layers)


def evaluate(layers, sequence):
    train(layers, sequence)


def loader(layers, sequence):
    assert len(data.load_batches()) == 2


def metrics(layers, sequence):
    assert data.summarize([1, 2, 3]) == {"count": 3, "sum": 6}


def new_cache(layers, sequence):
    assert data.cache_lookup("warm") == [1, 2, 3, 4]


TESTS = {
    "smoke_slow": train,
    "smoke_fast": train,
    "distributed_dp": train_dp,
    "distributed_tp": train_tp,
    "long_context": long_context,
    "checkpoint_roundtrip": checkpoint_roundtrip,
    "shape_guard": shape_guard,
    "compiled_train": compiled_train,
    "mixed_precision": mixed_precision,
    "evaluation": evaluate,
    "loader_cpu": loader,
    "metrics_cpu": metrics,
    "new_cache": new_cache,
}
