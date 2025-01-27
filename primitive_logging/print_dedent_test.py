from print_dedent import (
     print_dedent
)
import textwrap

def test_print_dedent_1():
    x = 3
    y = 7 
    print_dedent(
        f"""\

        Yo: {x=} and yet {y=}.
        What is up with that?
            
            Here is an indented block.
            t should be 5
        Now we are back out here.

        
        After blank lines.
        """
    )

    print(textwrap.dedent(
        f"""\

        Yo: {x=} and yet {y=}.
        What is up with that?
            
            Here is an indented block.
            t should be 5
        Now we are back out here.

        
        After blank lines.
        """
    ))

if __name__ == "__main__":
    test_print_dedent_1()
