import torch

def format_duration(start, end, device):
    duration = end - start
    msI = str(duration).find('.')
    milliseconds = int(str(duration)[msI+1:msI+4])
    duration = int(duration)
    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    formatted = f"{days:02d}d:{hours:02d}h:{minutes:02d}m:{seconds:02d}s:{milliseconds}ms "

    if device == 'cpu':
        device_specs = 'CPU'
    else:
        device_specs = f'GPU "{torch.cuda.get_device_name(0)}"'

    formatted += f'using the {device_specs}'
    return formatted.strip()