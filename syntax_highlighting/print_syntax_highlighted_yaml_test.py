from print_syntax_highlighted_yaml import (
     print_syntax_highlighted_yaml
)
from pathlib import Path

def test_print_syntax_highlighted_yaml_1():

    file_path = Path(
        "~/r/objects/5b/5b5437dcef44793e0b2b5fbd8010cfc5af8eaf63045da8b5a96fac9f3daffed5.yaml"
    ).expanduser()
    source_code = file_path.read_text()
    print_syntax_highlighted_yaml(source_code)

if __name__ == "__main__":
    test_print_syntax_highlighted_yaml_1()
