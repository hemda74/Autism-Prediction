import string
import random

def generateID(filename):
    rand_str = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))
    new_filename = rand_str + filename
    return new_filename
