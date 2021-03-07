def scale(x, min, max):
    return (x - min) / (max - min)

def pretty_string_device_file(fname: str) -> str:
    if 'hybrid' in fname:
        return 'hybrid'
    if 'cloudlet' in fname:
        return 'edge_cloudlet'
    if 'cloud_cpu' in fname:
        return 'cloud_cpu'
