from pygments import highlight, lexers, formatters


def print_syntax_highlighted_yaml(source_code):
    """
    syntax highlights then prints Python source code
    """
    style = "monokai"
    colorful = highlight(
        code=source_code,
        lexer=lexers.YamlLexer(),
        formatter=formatters.Terminal256Formatter(style=style)
    )
    print(colorful)

