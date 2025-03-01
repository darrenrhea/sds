import json
from colorama import Fore, Style
from collections import OrderedDict
from typing import List, Tuple
from dataclasses import dataclass


class Expr(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        legal_expr_types = [
            "call",
            "assignment",
            "block",
            "import",
            "raw_source_code"
        ]
        assert "type" in self.keys(), "All AST nodes must have a type"
        type_str = self["type"]
        assert isinstance(type_str, str), "the type of an AST node must be a string"
        assert type_str in legal_expr_types,  f"the type of an AST node must amongst {legal_expr_types} but you said {type_str}"


@dataclass
class State:
    previous_lines: List[Tuple[int, str]]
    current_indent: int
    current_line: str



def finish_line_into_state(s: str, state: State, delta_indent: int):
    """
    If you know the state, you can print into it.
    """
    assert isinstance(state, State)
    # unpack state:
    previous_lines = state.previous_lines
    current_line = state.current_line
    current_indent = state.current_indent

    finalized_line = state.current_line + s

    previous_lines.append(
        (current_indent, finalized_line)
    )
    current_indent = current_indent + delta_indent
    current_line = ""
    state = State(
        previous_lines=previous_lines,
        current_indent=current_indent,
        current_line=current_line
    )
    return state

def partial_print_into_state(s: str, state: State):
    """
    If you know the state, you can print into it.
    """
    assert isinstance(s, str), "s must be a string"
    # unpack state:
    previous_lines = state.previous_lines
    current_line = state.current_line
    current_indent = state.current_indent

    current_line = state.current_line + s

    state = State(
        previous_lines=previous_lines,
        current_indent=current_indent,
        current_line=current_line
    )
    return state

def dedent_state(state: State, delta_indent: int = -4):
    """
    If you know the state, you can print into it.
    """
    # unpack state:
    previous_lines = state.previous_lines
    current_line = state.current_line
    current_indent = state.current_indent
    assert current_line == "", "ERROR: very suspicious that the current line is not empty and you are dedenting"

    state = State(
        previous_lines=previous_lines,
        current_indent=current_indent + delta_indent,
        current_line=current_line
    )
    return state
    

def print_ast_into_state(expr, state: State):
    """
    """
    if isinstance(expr, Expr):
        type_ = expr["type"]
        if type_ == "call":
            keyword_to_value = expr["kwargs"]
            assert (
                isinstance(keyword_to_value, OrderedDict)
            ), "a call's kwargs must be an OrderedDict"
            fn = expr["fn"]
            assert isinstance(fn, str), "a call's fn must be a string"
            s = f"{fn}("
            state = finish_line_into_state(s=s, state=state, delta_indent=4)
            for keyword, value in keyword_to_value.items():
                s = f"{keyword}="
                state = partial_print_into_state(s=s, state=state)
                state = print_ast_into_state(expr=value, state=state)
                state = finish_line_into_state(s=",", state=state, delta_indent=0)
            state = dedent_state(state)
            s = ")"  # close the call
            state = partial_print_into_state(s=s, state=state)
        elif type_ == "block":
            lst = expr["lst"]
            assert isinstance(lst, list), "a block's lst must be a list"
            for value in lst:
                assert isinstance(value, Expr), "a block's lst must be a list of Exprs"
                state = print_ast_into_state(expr=value, state=state)
                state = finish_line_into_state(s="", state=state, delta_indent=0)
        elif type_ == "import":
            module = expr["module"]
            items = expr.get("items")
            assert items is None or isinstance(items, list), "a block's lst must be a list or None"
            if items is None:
                s = f"import {module}"
                state = finish_line_into_state(s=s, state=state)
            elif len(items) == 1:
                item = items[0]
                assert isinstance(item, str), "item to import must be a string"
                if module == item:  # import from yourself form
                    indent = 5
                    s = f"from {module} import ("
                    state = finish_line_into_state(s=s, state=state, delta_indent=indent)
                    state = finish_line_into_state(s=f"{item}", state=state, delta_indent=0)
                    state = dedent_state(state, delta_indent=-indent)
                    s = ")"  # close the import
                    state = finish_line_into_state(s=s, state=state, delta_indent=0)
                else:
                    indent = 4 
                    s = f"from {module} import {item}"
                    state = finish_line_into_state(s=s, state=state, delta_indent=0)
            else:
                s = f"from {module} import ("
                state = finish_line_into_state(s=s, state=state, delta_indent=4)
                for item in items:
                    assert isinstance(item, str), "items to import must be strings"
                    state = finish_line_into_state(s=f"{item},", state=state, delta_indent=0)
                state = dedent_state(state, delta_indent=-4)
                s = ")"  # close the import
                state = finish_line_into_state(s=s, state=state, delta_indent=0)
        elif type_ == "raw_source_code":  # this is weak.  We should have a better way to handle this.
            s = expr["text"]
            state = partial_print_into_state(s=s, state=state)
        else:
            raise ValueError(f"unknown Expr type {type_}")
    elif isinstance(expr, str):
        s = f'"{expr}"'
        state = partial_print_into_state(s=s, state=state)
    elif isinstance(expr, list) or isinstance(expr, tuple):
        if isinstance(expr, list):
            s = "["  # open the list
        else:
            s = "(" # open the tuple
        state = finish_line_into_state(s=s, state=state, delta_indent=4)

        for value in expr:
            state = print_ast_into_state(expr=value, state=state)
            state = finish_line_into_state(s=",", state=state, delta_indent=0)
        state = dedent_state(state)
        if isinstance(expr, list):
            s = "]"  # open the list
        else:
            s = ")" # open the tuple
        state = partial_print_into_state(s=s, state=state)   
    elif isinstance(expr, dict):
        keyword_to_value = expr
        for keyword, value in keyword_to_value.items():
            assert isinstance(keyword, str), "ERROR: currently we cannot handle dicts with non-string keys"
            s = "{"
            state = finish_line_into_state(s=s, state=state, delta_indent=4)
            for keyword, value in keyword_to_value.items():
                s = json.dumps(keyword) + ": "
                state = partial_print_into_state(s=s, state=state)
                state = print_ast_into_state(expr=value, state=state)
                state = finish_line_into_state(s=",", state=state, delta_indent=0)
            state = dedent_state(state)
            s = "}"  # close the dict
            state = partial_print_into_state(s=s, state=state)
    return state


def print_lines(state):
    for indent, line in state.previous_lines:
        print(" " * indent + line)
    print(" " * state.current_indent + state.current_line + "|")


def state_to_string(state):
    pieces = []
    for indent, line in state.previous_lines:
        pieces.append(" " * indent + line + "\n")
    pieces.append(" " * state.current_indent + state.current_line)
    return "".join(pieces)


def expr_to_string(expr: Expr):
    state = State(
        previous_lines=[],
        current_indent=0,
        current_line=""
    )
    state = print_ast_into_state(expr=expr, state=state)
    return state_to_string(state)



