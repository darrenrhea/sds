from colorama import Fore, Style
from collections import OrderedDict
from typing import List, Tuple
from dataclasses import dataclass
from ast_to_concrete_syntax import Expr, expr_to_string
from pathlib import Path

def test_ast_to_concrete_syntax_1():
    expr = Expr(
        type="call",
        fn="setup",
        kwargs=OrderedDict(
            name="image_openers",
            version="0.1.0",
            py_modules=["amod", "bmod"],
            license="MIT",
            long_description=Expr(
                type="raw_source_code",
                text="open('README.md').read()",
            ),
            long_description_content_type='text/markdown',
            install_requires=[
                "numpy",
                "colorama",
            ],
            entry_points={
                "console_scripts": [
                    "a = amod_cli_tool:amod_cli_tool",
                    "b = bmod_cli_tool:bmod_cli_tool",
                ]
            }
        )
    )

    result = expr_to_string(expr)
    should_be = Path(
        "fixtures/ast_to_concrete_syntax_1_test.result"
    ).write_text(result)

    print(result)
    should_be = Path(
        "fixtures/ast_to_concrete_syntax_1_test.should_be"
    ).read_text()
    print("expected")

    # print(expected)
    assert result == should_be




def test_ast_to_concrete_syntax_2():
    call_to_setup = Expr(
        type="call",
        fn="setup",
        kwargs=OrderedDict(
            name="image_openers",
            version="0.1.0",
            py_modules=["amod", "bmod"],
            license="MIT",
            long_description=Expr(
                type="raw_source_code",
                text="open('README.md').read()",
            ),
            long_description_content_type='text/markdown',
            install_requires=[
                "numpy",
                "colorama",
            ],
            entry_points={
                "console_scripts": [
                    "a = amod_cli_tool:amod_cli_tool",
                    "b = bmod_cli_tool:bmod_cli_tool",
                ]
            }
        )
    )

    expr = Expr(
        type="block",
        lst=[
            call_to_setup,
            call_to_setup
        ],
    )

    result = expr_to_string(expr)
    should_be = Path(
        "fixtures/ast_to_concrete_syntax_2_test.result"
    ).write_text(result)

    print(result)
    should_be = Path(
        "fixtures/ast_to_concrete_syntax_2_test.should_be"
    ).read_text()
    print("expected")

    # print(expected)
    assert result == should_be





if __name__ == "__main__":
    test_ast_to_concrete_syntax_1()
    print(f"{Fore.GREEN}All tests passed!{Style.RESET_ALL}")