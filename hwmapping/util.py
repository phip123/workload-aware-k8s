def extract_model_type(device: str):
    if not type(device) is str:
        return ''
    try:
        return device[:device.rindex('_')]
    except ValueError:
        return device
