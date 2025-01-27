import numpy as np
import sys
from colorama import Fore, Style
import textwrap
from dedent_lines import dedent_lines



def random_string_x574873875():
    s = []
    for _ in range(10):
        s.append(np.random.choice(["a", " ", "\n"]))
    return "".join(s)

a='\na aa    \n'
def test_dedent_lines_1():

    actual = dedent_lines(a)
    should_be = textwrap.dedent(a)
    print(f"{a=}")
    print(f"{actual=}")
    print(f"{should_be=!r}")
    assert actual == should_be
            

def test_dedent_lines_2():
    for cntr in range(1000):
        a = random_string_x574873875()
        actual = dedent_lines(a)
        if not actual == textwrap.dedent(a):
            print(f"{a=}")
            print(f"{actual=}")
            print(f"{textwrap.dedent(a)=}")
            break
   

if __name__ == "__main__":
    test_dedent_lines_2()


