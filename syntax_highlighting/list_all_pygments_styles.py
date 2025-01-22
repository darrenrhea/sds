from pygments import highlight, lexers, formatters

from pygments.styles import STYLE_MAP

from pygments.styles import get_all_styles

styles = list(get_all_styles())
print(styles)
print(
    STYLE_MAP.keys()
)


