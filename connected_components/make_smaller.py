def make_smaller(x0x1y0y1, manner):
    assert manner in [0, 1, 2, 3]
    x0, x1, y0, y1 = x0x1y0y1
    failed = True
    new_x0x1y0y1 = None
    if manner == 0:
        if x0 < x1 - 1:
            new_x0x1y0y1 = (x0 + 1, x1, y0, y1)
            failed = False
        return new_x0x1y0y1, failed
    elif manner == 1:
        if x0 < x1 - 1:
            new_x0x1y0y1 = (x0, x1 - 1, y0, y1)
            failed = False
        return new_x0x1y0y1, failed
    elif manner == 2:
        if y0 < y1 - 1:
            new_x0x1y0y1 = (x0, x1, y0 + 1, y1)
            failed = False
        return new_x0x1y0y1, failed
    elif manner == 3:
        if y0 < y1 - 1:
            new_x0x1y0y1 = (x0, x1, y0, y1 - 1)
            failed = False
        return new_x0x1y0y1, failed
    else:
        raise Exception(f"{manner=}")
    
