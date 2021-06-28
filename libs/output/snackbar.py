from termcolor import colored
from ..flags import Flags
import sys



def snackbar(message, type = 'main', log = 'main'):
    flags = Flags()
    if (log == 'main' or (log == 'log' and flags.log)):
        if (type == 'error'):
            print(colored('Error:\t', 'red'), message, file=sys.stderr)
            sys.exit(2)
        elif(type == 'info'):
            print(colored('Info:\t', 'cyan'), message)
        elif(type == 'success'):
            print(colored('Success:', 'green'), message)
        elif(type == 'main'):
            print(colored(message, 'yellow'))
        elif(type == ''):
            print(message)
