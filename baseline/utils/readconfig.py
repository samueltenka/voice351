''' author: samuel tenka
    date: 2016-11-06
    descr: read from config file
    use:
'''

import terminal 
import os
print(os.listdir('.'))

try:
    with open('config.json') as f:
        json = eval(f.read())
except IOError: terminal.complain('accessing `config.json`!')
except SyntaxError: terminal.complain('parsing `config.json`!')

def get(fieldnm):
    return json[fieldnm] 
