def data_parallel(values):
    shards = [values[::2], values[1::2]]
    return sum(sum(shard) for shard in shards)


def tensor_parallel(values):
    return sum(values) + 0
