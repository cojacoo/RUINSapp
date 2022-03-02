import os
from ruins import __version__

def increment(which='patch'):
    """
    Increment the version number. 
    """
    parts = __version__.split('.')
    if which == 'patch':
        parts[2] = str(int(parts[2]) + 1)
    elif which == 'minor':
        parts[1] = str(int(parts[1]) + 1)
        parts[2] = '0'
    elif which == 'major':
        parts[0] = str(int(parts[0]) + 1)
        parts[1] = '0'
        parts[2] = '0'
    else:
        raise ValueError("Invalid version increment.")
    return '.'.join(parts)


def replace(which='patch'):
    """
    Increment the version number for RUINS.

    """
    # find the file
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ruins', '__init__.py'))
    
    # read
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # replace the version
    for i, line in enumerate(lines):
        if '__version__' in line:
            new_version = increment(which)
            lines[i] = f"__version__ = '{new_version}'\n"
            break
    
    # overwrite
    with open(path, 'w') as f:
        f.writelines(lines)
    
    print(new_version)  


if __name__ == '__main__':
    import fire
    fire.Fire(replace)