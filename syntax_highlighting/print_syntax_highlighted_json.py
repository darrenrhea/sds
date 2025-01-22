from pygments import highlight, lexers, formatters


def print_syntax_highlighted_json(source_code):
    """
    syntax highlights then prints JSON source code.
    JSON with comments works, JSON5 does not.
    https://github.com/pygments/pygments/issues/1880
    """
    style = "monokai"
    colorful = highlight(
        code=source_code,
        lexer=lexers.data.JSONLexer(),
        formatter=formatters.Terminal256Formatter(style=style)
    )
    print(colorful)

