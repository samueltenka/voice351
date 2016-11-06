''' author: sam tenka
    date: 2016-11-06
    descr: Making the terminal great again, via colors, 
           progress bars, and standardized user-input prompts.
    use:
'''

from __future__ import print_function
import os

def colorize(string):
    ''' Substitute ANSI escape codes for corresponding English commands.

        Example: If s=='Colorless {GREEN} ideas {BLUE} sleep {RED} furiously.',
        then print(f(s)) will produce the above famous sentence, where 'ideas'
        is in green type, 'sleep' is in blue type, and 'furiously.' and
        following text is is red type.

        Example: If s=='{UP}{UP}', then print(s) will raise the cursor two
        lines.
    '''
    for i, o in {'{RED}':'\033[31m',
                 '{YELLOW}':'\033[33m',
                 '{GREEN}':'\033[32m',
                 '{CYAN}':'\033[36m',
                 '{BLUE}':'\033[34m',
                 '{MAGENTA}':'\033[35m',
                 '{LEFT}':'\033[1D',
                 '{UP}':'\033[1A'}.items():
        string = string.replace(i, o)
    return string 

def set_color(color):
    ''' Set terminal color. See `colorize` for color choices.
    '''
    print(colorize('{%s}' % color), end='')

def complain(message):
    print(colorize('{RED}ERROR{YELLOW} %s' % message))

def user_input_iterator(prompt='> '):
    ''' Return string-generator of user input.
    '''
    colored_prompt = colorize('{GREEN}%s{BLUE}' % prompt)
    while True:
        ri = raw_input(colored_prompt)
        set_color('{RED}')
        if ri=='clear': os.system('clear')
        elif ri=='exit': break
        else: yield ri

