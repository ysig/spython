import os, sys
from os import listdir
from os.path import join, abspath

def ls(sub):
    return 'Available tags: ' + str(list(listdir(sub)))

if __name__ == "__main__":
    sub = os.environ['SUB']
    if not os.path.isdir(sub):
        print(f'{sub} is not initialized.')
        sys.exit(0)

    if len(sys.argv) == 1:
        print(ls(sub))
        sys.exit(0)

    tag = sys.argv[1]
    name = 'log.txt'
    k = -1
    if len(sys.argv) > 2:
        if sys.argv[2] == '--script':
            name = 'script.txt'
        else:
            k = int(sys.argv[2])
 
    if len(sys.argv) > 3:
        assert sys.argv[3] == '--script' 
        name = 'script.txt'

    path = join(sub, tag)
    try:
        fp = sorted(listdir(path))[k]
    except FileNotFoundError:
        print('Tag not found...\n' + ls(sub))
        sys.exit(1)

    os.system('cat ' + abspath(join(path, fp, name)))
