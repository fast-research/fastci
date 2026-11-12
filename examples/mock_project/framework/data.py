def load_batches():
    return [[1, 2, 3, 4], [5, 6, 7, 8]]


def summarize(values):
    return {"count": len(values), "sum": sum(values)}


def cache_lookup(key):
    return {"warm": [1, 2, 3, 4]}[key]
