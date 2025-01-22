from pygments import highlight, lexers, formatters

from pygments.styles import STYLE_MAP

# from pygments.styles import get_all_styles
# styles = list(get_all_styles())
# print(styles)
# # print(
# #     STYLE_MAP.keys()
# # )

def print_syntax_highlighted_python(source_code):
    """
    syntax highlights then prints Python source code
    """
    style = "monokai"
    colorful = highlight(
        code=source_code,
        lexer=lexers.Python3Lexer(),
        formatter=formatters.Terminal256Formatter(style=style)
    )
    print(colorful)

