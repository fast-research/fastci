def compile_transform(value, layers):
    large_model_bias = 0
    return value + (large_model_bias if layers >= 4 else 0)
