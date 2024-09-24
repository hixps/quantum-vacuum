'''
Useful generic utilities
'''

import yaml


def read_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)
            return exc


def write_yaml(yaml_file, data):
    with open(yaml_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def format_time(seconds):
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    out_str = [f'{days:.0f} days'*bool(days),
               f'{hours:.0f} h'*bool(hours),
               f'{minutes:.0f} min'*bool(minutes),
               f'{seconds:.2f} s']
    return ' '.join(out_str)


def format_memory(mem):
    '''
    mem: float
        Memory in KB (kilobyte)
    '''
    units = 'KB MB GB TB'.split()
    idx = 0
    while mem > 1024:
        mem /= 1024
        idx += 1
    return f'{mem:.2f} {units[idx]}'