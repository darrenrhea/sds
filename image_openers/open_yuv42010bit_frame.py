"""

"""
import sys
import PIL.Image
import numpy as np
from print_image_in_iterm2 import print_image_in_iterm2
from pathlib import Path
import cv2

yuv_file_path = Path(
    "~/a/s3/awecomai-test-videos/nba/Mathieu/temp/slgame1_000001.yuv"
).expanduser()

file_bytes = yuv_file_path.read_bytes()
assert isinstance(file_bytes, bytes)
assert len(file_bytes) == 6220800

y_segment = file_bytes[:1920*1080*2]
u_segment = file_bytes[1920*1080*2:1920*1080*2 + 960*540*2]
v_segment = file_bytes[1920*1080*2 + 960*540*2:]

dt = np.dtype(np.uint16)
dt = dt.newbyteorder('little')  # makes sure it is little endian, but it was already, so not actually needed
y = np.frombuffer(y_segment, dtype=dt)
y = y.reshape(1080, 1920)

u = np.frombuffer(u_segment, dtype=dt)
u = u.reshape(540, 960)

v = np.frombuffer(v_segment, dtype=dt)
v = v.reshape(540, 960)

print(f"{np.max(y)=}, {np.min(y)=}")
print(f"{np.max(u)=}, {np.min(u)=}")
print(f"{np.max(v)=}, {np.min(v)=}")

PIL.Image.fromarray(y*64).save('y.png')
PIL.Image.fromarray(u*64).save('u.png')
PIL.Image.fromarray(v*64).save('v.png')

u_upscaled = u.repeat(2, axis=0).repeat(2, axis=1)
v_upscaled = v.repeat(2, axis=0).repeat(2, axis=1)

yuv = np.concatenate([y[:, :, None], u_upscaled[:, :, None], v_upscaled[:, :, None]], axis=2)
print(f"{yuv.shape=}, {yuv.dtype=}")
bgr = cv2.cvtColor(yuv*64, cv2.COLOR_YUV2BGR)
print(f"{bgr.shape=}, {bgr.dtype=}")
cv2.imwrite('converted.png', bgr)
b_min = np.min(bgr[:, :, 0])
g_min = np.min(bgr[:, :, 1])
r_min = np.min(bgr[:, :, 2])
print(f"{b_min=}, {g_min=}, {r_min=}")
sys.exit(1)


"""
r = 1.164 * y             + 1.596 * v
g = 1.164 * y - 0.392 * u - 0.813 * v
b = 1.164 * y + 2.017 * u
"""
M = np.array(
    [
        [1.164,      0,  1.596],
        [1.164, -0.392,  -0.813],
        [1.164,  2.017,       0]
    ]
)

print(np.linalg.inv(M))

"""
U1 Y1 V1 Y2
U3 Y3 V3 Y4
U5 Y5 V5 Y6 


a b c d
e f g h
i j

file size is 5529600 bytes
UYVY is 20 bits per pixel
1080 * 1920 * (10/8 + 10/8) = 5184000 bytes

5529600 / 5184000 = 1.0666666666666667

96/90 = 1.0666666666666667

https://wiki.multimedia.cx/index.php/V210

6 pixels are encoded in 16 bytes = 128 bits with 4*2 padding bits 

each 128-bit block is interpreted as twelve different value 10-bit values are stored like:
block 1, bits  0 -  9: U1
block 1, bits 10 - 19: Y0
block 1, bits 20 - 29: V1
block 2, bits  0 -  9: Y1
block 2, bits 10 - 19: U3
block 2, bits 20 - 29: Y2
block 3, bits  0 -  9: V3
block 3, bits 10 - 19: Y3
block 3, bits 20 - 29: U5
block 4, bits  0 -  9: Y4
block 4, bits 10 - 19: V5
block 4, bits 20 - 29: Y5

These twelve 10-bit values are stored in each 128-bit block,
U1 Y0 V1
Y1 U3 Y2
V3 Y3 U5
Y4 V5 Y5
that is 6-Y values and 3-U values and 3 V-values
In space they are like:
Y0 Y1 Y2 Y3 Y4 Y5
   U1    U3    U5
   V1    V3    V5
"""

import struct
# load the file as bytestring:

def convert_10bityuv444_to_10bitrgb444(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray
):
    """
    adapted from:
    http://adec.altervista.org/blog/yuv-422-v210-10-bit-packed-decoder-in-glsl/?doing_wp_cron=1700710014.6635560989379882812500

    """
    y = y.astype(np.float32)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    y = y - 64
    u = u - 512
    v = v - 512

    nr = 1.164 * y             + 1.596 * v
    ng = 1.164 * y - 0.392 * u - 0.813 * v
    nb = 1.164 * y + 2.017 * u

    nr = nr / 2
    ng = ng / 2
    nb = nb / 2
    
    
    r = nr.round().clip(0, 1023).astype(np.uint16)
    g = ng.round().clip(0, 1023).astype(np.uint16)
    b = nb.round().clip(0, 1023).astype(np.uint16)
    return r, g, b


def convert_10bitrgb444_to_10bityuv444(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray
):
    """
    Going backwards:
    adapted from:
    http://adec.altervista.org/blog/yuv-422-v210-10-bit-packed-decoder-in-glsl/?doing_wp_cron=1700710014.6635560989379882812500

    
    """
    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)

    r = r * 2
    g = g * 2
    b = b * 2

    
    y =  0.25686190 * r + 0.5042455 * g  + 0.09799913 * b
    u = -0.14823364 * r - 0.2909974 * g  + 0.43923104 * b
    v =  0.43923104 * r - 0.3677580 * g  - 0.07147305 * b

    y = y + 64
    u = u + 512
    v = v + 512

    y = y.round().clip(0, 1023).astype(np.uint16)
    u = u.round().clip(0, 1023).astype(np.uint16)
    v = v.round().clip(0, 1023).astype(np.uint16)
    return r, g, b


def printblock(n):
    s = f"{n:032b}"
    print("01234567890123456789012345678901")
    t = s[::-1]
    print(f"{t[0:9]} {t[10:20]} {t[20:30]} XX")
    print("")

def asbinary(number, num_bits):
    s = f"{number:0{num_bits}b}"
    return s[::-1]

assert asbinary(7, 10) == "1110000000"
assert asbinary(513, 10) == "1000000001"
assert asbinary(515, 10) == "1100000001"

bytes = open('frame.bin', 'rb').read()
print(f"{len(bytes)=}")
assert len(bytes) == 5529600

num_bytes = 5529600 

y_flat = np.zeros(
    shape=(1080, 1920),
    dtype=np.uint16
).flatten()

u_flat = np.zeros(
    shape=(1080, 1920),
    dtype=np.uint16
).flatten()

v_flat = np.zeros(
    shape=(1080, 1920),
    dtype=np.uint16
).flatten()



with open('recent/frame.bin', 'rb') as fp:

    for index_of_which_16byte_block in range(num_bytes // 16):
        print(f"{index_of_which_16byte_block=}")
        
        block1_4bytes = fp.read(4)
    
        block1 = struct.unpack("<I", block1_4bytes)[0]
        assert 0 <= block1 and block1 <= 2**32 - 1
    
        block2_4bytes = fp.read(4)
        block2 = struct.unpack("<I", block2_4bytes)[0]
        assert 0 <= block2 and block2 <= 2**32 - 1

        block3_4bytes = fp.read(4)
        block3 = struct.unpack("<I", block3_4bytes)[0]
        assert 0 <= block3 and block3 <= 2**32 - 1

        block4_4bytes = fp.read(4)
        block4 = struct.unpack("<I", block4_4bytes)[0]
        assert 0 <= block4 and block4 <= 2**32 - 1

        # printblock(block1)
        # printblock(block2)
        # printblock(block3)
        # printblock(block4)  


        # From block1 extract bits 0 through 9:
        # U1 Y0 V1
        U1 = block1 & 0b1111111111
        Y0 = (block1 >> 10) & 0b1111111111
        V1 = (block1 >> 20) & 0b1111111111

        # Y1 U3 Y2
        Y1 = block2 & 0b1111111111
        U3 = (block2 >> 10) & 0b1111111111
        Y2 = (block2 >> 20) & 0b1111111111
        

        # V3 Y3 U5
        V3 = block3 & 0b1111111111
        Y3 = (block3 >> 10) & 0b1111111111
        U5 = (block3 >> 20) & 0b1111111111

        # Y4 V5 Y5
        Y4 = block4 & 0b1111111111
        V5 = (block4 >> 10) & 0b1111111111
        Y5 = (block4 >> 20) & 0b1111111111

        V0 = V1
        U0 = U1
        U2 = U3
        V2 = V3
        U4 = U5
        V4 = V5

        y_flat[6*index_of_which_16byte_block:6*index_of_which_16byte_block+6] = [Y0, Y1, Y2, Y3, Y4, Y5]
        u_flat[6*index_of_which_16byte_block:6*index_of_which_16byte_block+6] = [U0, U1, U2, U3, U4, U5]
        v_flat[6*index_of_which_16byte_block:6*index_of_which_16byte_block+6] = [V0, V1, V2, V3, V4, V5]
        index_of_which_16byte_block += 1

    yuv_u16 = np.zeros(
        shape=(1080, 1920, 3),
        dtype=np.uint16
    )

    y = y_flat.reshape(1080, 1920).astype(np.uint16)
    u = u_flat.reshape(1080, 1920).astype(np.uint16)
    v = v_flat.reshape(1080, 1920).astype(np.uint16)

    print(f"{y.min(axis=0).min(axis=0)=}")
    print(f"{y.max(axis=0).max(axis=0)=}")

    print(f"{u.min(axis=0).min(axis=0)=}")
    print(f"{u.max(axis=0).max(axis=0)=}")

    print(f"{v.min(axis=0).min(axis=0)=}")
    print(f"{v.max(axis=0).max(axis=0)=}")
   

    r, g, b = convert_10bityuv444_to_10bitrgb444(
        y=y, u=u, v=v
    )

    recovered_u, recovered_y, recovered_v = convert_10bitrgb444_to_10bityuv444(
        r=r, g=g, b=b
    )



    

    rgb_u16 = np.zeros(
        shape=(1080, 1920, 3),
        dtype=np.uint16
    )

    rgb_u16[:, :, 0] = r
    rgb_u16[:, :, 1] = g
    rgb_u16[:, :, 2] = b


    rgb_u8 = (rgb_u16 // 4).astype(np.uint8)
    img = PIL.Image.fromarray(rgb_u8)
    img.save('temp.png')
    print_image_in_iterm2(image_path=Path('temp.png'))


print (f"{num_bytes // 16 * 6=}")