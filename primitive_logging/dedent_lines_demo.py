import dedent_lines
from dedent_lines import (
     dedent_lines
)
import textwrap


def dedent_lines_demo():
    multiline_string = """\
        Hi

        Dog   
        

        cat




    """
    a = dedent_lines(multiline_string)
    b = textwrap.dedent(multiline_string)

    print(repr(a))
    print(repr(b))

    multiline_string = """\
  


        Hi
        Why


    """
    
    c = dedent_lines(multiline_string)
    for k in range(4):
        print(c)

    d = textwrap.dedent(multiline_string)
    for k in range(4):
        print(d)

    print(repr(c))
    print(repr(d))


if __name__ == "__main__":
    dedent_lines_demo()