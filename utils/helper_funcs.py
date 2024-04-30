def get_lp_size(max_lp_size):
    sizes = [4**i for i in range(1, 5)]
    return [size for size in sizes if size <= max_lp_size]

def before_pad(arg):
    arg_temp = []
    for atm in arg:
        if atm == 'PAD':
            break
        arg_temp.append(atm)
    return arg_temp