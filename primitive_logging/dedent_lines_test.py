import numpy as np
import textwrap
from dedent_lines import dedent_lines



def random_string_x574873875():
    s = []
    for _ in range(10):
        s.append(np.random.choice(["a", " ", "\n"]))
    return "".join(s)

def test_dedent_lines_1():
    a='\na aa    \n'
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
        d = textwrap.dedent(a)
        if not actual == d:
            print(f"s:\n{a=}")
            print("dedent_lines:")
            print(repr(actual))
            print("textwrap.dedent:")
            print(repr(d))
            break
   

if __name__ == "__main__":
    test_dedent_lines_2()


