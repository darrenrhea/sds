import dedent_lines
from dedent_lines import (
     dedent_lines
)
import textwrap


def dedent_lines_demo():
    
    multiline_strings = [

        """\




        Many blank lines came before this line.
        Many blank lines came after this line.




        """,
        """\



        jpsosa_just_paste_scorebugs_onto_segmentation_annotations \\
        --in_dir ~/cleandata \\
        --out_dir ~/withscorebugs \\
        --print_in_iterm2

        """
    ]
    for multiline_string in multiline_strings:
        a = dedent_lines(multiline_string)
        b = textwrap.dedent(multiline_string)
        print(f"multiline_string:\n{multiline_string}")
        print(f"BEGIN dedent_lines:\n{a}ENDOF dedent_lines.\n")
        print(f"BEGIN textwrap.dedent:\n{b}ENDOF dedent_lines.\n")
        print(10*"\n")
    



if __name__ == "__main__":
    dedent_lines_demo()