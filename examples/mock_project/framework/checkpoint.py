import json


def roundtrip(values, layers):
    payload = json.dumps({"layers": layers, "values": values})
    restored = json.loads(payload)
    assert restored["layers"] == layers
    return restored["values"]
