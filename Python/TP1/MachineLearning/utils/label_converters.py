def label_to_int(string_label):
    if string_label == 'rectangle': return 1
    if string_label == '5-point-star': return 2
    if string_label == 'triangle':
        return 3

    else:
        raise Exception('unkown class_label')

# como lo adapto para laburar con video?
def int_to_label(string_label):
    if string_label == 1: return 'rectangle'
    if string_label == 2: return '5-point-star'
    if string_label == 3:
        return 'triangle'
    else:
        raise Exception('unkown class_label')
