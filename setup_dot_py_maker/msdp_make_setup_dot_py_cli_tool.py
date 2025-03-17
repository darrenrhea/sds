import argparse
import textwrap
from colorama import Fore, Style
from collections import OrderedDict
from ast_to_concrete_syntax import Expr, expr_to_string
from pathlib import Path
from print_syntax_highlighted_python import print_syntax_highlighted_python

# although you can do this, we no longer like the idea
# of straight up installing short-named executables.
# instead eval $(set_abbreviations zsh darrenrhea)

short_to_long = dict(
    # e="edit_topic_book",
    # nu="nu_ie_store_url_into_my_system",
    # store_calibre="store_calibre_converted_pdf",
    # store="store_file",
    # v="vscode",
    # dhtstore="store_file_by_sha256",
    # fws256="get_file_path_of_sha256",
)

def make_setup_dot_py_for_this_dir(
        checkout_dir: Path,
        dry_run: bool = False
    ):
    """
    Given that you are inside a folder with python files,
    some of which end in ``_cli_tool.py``,
    this generates a setup.py to install all
    the ones that end in ``_cli_tool.py`` as CLI executables
    when someone does, say,
    
    .. code-block:: bash

        cd folder_containing_setup_dot_py
        pip install -e . --no-deps

    It will also generate a README.md file if you don't have one,
    and a .gitignore file if you don't have one.
    """
    assert checkout_dir.is_dir(), "checkout_dir must be a directory"
    name = checkout_dir.name

    all_python_files = list(checkout_dir.glob("**/*.py"))

    module_names = sorted([f.stem for f in all_python_files])

    # exlude test files and setup.py files.  We exclude cli_tools later.
    maybe_py_modules = [
        module_name for module_name in module_names
        if (
            (
                not module_name.startswith("test_")
                and 
                not module_name.endswith("_test")
                and
                not module_name.endswith("setup")  # there may already be a setup.py in the directory,
                and
                not module_name.endswith("generated_setup")  # we make this.
            )
        )
    ]
    
    cli_tools = [
        module_name for module_name in maybe_py_modules
        if (
            module_name.endswith("_cli_tool")
        )
    ]

    # seems like you don't have to list the cli_tools in py_modules for them to work?
    py_modules = [x for x in maybe_py_modules if x not in cli_tools]

    console_scripts = []
    for module_name in cli_tools:
        fully_qualified_executable_name = module_name[:-9]
        short_names = [key for key in short_to_long if short_to_long[key] == fully_qualified_executable_name]
        if len(short_names) == 0:
            short_names.append(fully_qualified_executable_name)
        for executable_name in short_names:
            console_scripts.append(
                f"{executable_name} = {module_name}:{module_name}"
            )
       

    install_requires = []


    call_to_setup = Expr(
        type="call",
        fn="setup",
        kwargs=OrderedDict(
            name=name,
            version="0.1.0",
            py_modules=py_modules,
            license="MIT",
            long_description=Expr(
                type="raw_source_code",
                text="open('README.md').read()",
            ),
            long_description_content_type='text/markdown',
            install_requires=install_requires,
            entry_points={
                "console_scripts": console_scripts
            }
        )
    )

    import_expr = Expr(
        type="import",
        module="setuptools",
        items=["setup"]
    )


    expr = Expr(
        type="block",
        lst=[
            import_expr,
            call_to_setup,
        ],
    )

    result = expr_to_string(expr)
    print(result)

    
    setup_dot_py_path = checkout_dir / "setup.py"

    if setup_dot_py_path.exists():
        if dry_run:
            generated_setup_dot_py_path = checkout_dir / "generated_setup.py"
            alternative_suggested = True
        else:
            generated_setup_dot_py_path = checkout_dir / "setup.py"
            alternative_suggested = False

    else:
        print(f"{Fore.GREEN}You don't have a setup.py, so we make you setup.py file:{Style.RESET_ALL}")
        generated_setup_dot_py_path = checkout_dir / "setup.py"
        alternative_suggested = False

    
    generated_setup_dot_py_path.write_text(result)

    dot_gitignore_file_path = checkout_dir / ".gitignore"
    if not dot_gitignore_file_path.exists():
        print(f"{Fore.YELLOW}Making you .gitignore file{Style.RESET_ALL}")
        dot_gitignore_file_path.write_text(
f"""
dist/
.DS_Store
.pytest_cache/
__pycache__/
.vscode/settings.json
*.pyc
{name}.egg-info/
"""
        )

    readme_file_path = checkout_dir / "README.md"
    if not readme_file_path.exists():
        print(f"{Fore.YELLOW}Since you do not have one, making you README.md file{Style.RESET_ALL}")
        readme_file_path.write_text(
            textwrap.dedent(
                f"""\
                # {name}

                This is a private Python library that you should install like this:
                    
                ```bash
                cd ~/r
                git clone git@github.com:darrenrhea/{name}
                cd ~/r/{name}
                pip install -e . --no-deps
                ```
                """
            )
        )
    
    if alternative_suggested:
        print(f"{Fore.MAGENTA}We suggest you do:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}\n    icdiff setup.py generated_setup.py{Style.RESET_ALL}")

        print(f"{Fore.MAGENTA}\nAnd if those changes seem reasonable, do:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}\n    mv generated_setup.py setup.py{Style.RESET_ALL}")

def msdp_make_setup_dot_py_cli_tool():
    argp = argparse.ArgumentParser()
    argp.add_argument("-n", "--dry_run", action="store_true")
    args = argp.parse_args()
    dry_run = args.dry_run
    checkout_dir = Path.cwd()

    make_setup_dot_py_for_this_dir(
        checkout_dir=checkout_dir,
        dry_run=dry_run,
    )


