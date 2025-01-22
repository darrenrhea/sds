def interpolate_keyframes(x):
    #  "pure 0" -20 "fractional01" -13 "pure 1"  -10 "fractional12 " -3 "pure2"  0 "frac23" 20 pure3
    keyframes_left_to_right = [4613, 4700, 4800, 4900]
    ramps = [
        dict(a=-1000, b=-19, k1=0, k2=0),  # pure0
        dict(a=-19, b=-10, k1=0, k2=1),  # fractional01
        dict(a=-10, b=-10, k1=1, k2=1),  # pure 1
        dict(a=-10, b=-3, k1=1, k2=2),  # fractional12
        dict(a=-3, b=-3, k1=2, k2=2),  # pure2
        dict(a=-3, b=20, k1=2, k2=3),  # fractional23
        dict(a=20, b=100000, k1=3, k2=3),  # pure3
    ]

    for dct in ramps:
        a = dct["a"]
        b = dct["b"]
        k1 = dct["k1"]
        k2 = dct["k2"]
        if a <= x and x <= b:
            break
        
    # solve this for t: x = (1-t)*a + t* b = a + t*(b-a)    
    t = (x - a) / (b - a)
    keyframe1 = keyframes_left_to_right[k1]
    keyframe2 = keyframes_left_to_right[k2]
    frac1 = 1-t
    frac2 = t

    if k1==k2:
        frac1 = 1
        frac2 = 0

    return dict(
        keyframe1=keyframe1,
        keyframe2=keyframe2,
        frac1=frac1,
        frac2=frac2
    )





if __name__ == "__main__":

    for x in range(-47, 47):
        

        
        ans = interpolate_keyframes(x)
        frac1 = ans["frac1"]
        frac2 = ans["frac2"]
        keyframe1 = ans["keyframe1"]
        keyframe2 = ans["keyframe2"]

        print(f"for x = {x}, lets do {frac1} of keyframe {keyframe1} plus {frac2} of keyframe {keyframe2}")