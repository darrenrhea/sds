import base64
import os
import sys

def print_partial(msg):
    print(msg, end='', file=sys.stderr)

# tmux requires unrecognized OSC sequences to be wrapped with DCS tmux;
# <sequence> ST, and for all ESCs in <sequence> to be replaced with ESC ESC. It
# only accepts ESC backslash for ST.
def print_osc(terminal):
    if terminal.startswith('screen'):
        print_partial("\033Ptmux;\033\033]")
    else:
        print_partial("\033]")


# More of the tmux workaround described above.
def print_st(terminal):
    if terminal.startswith('screen'):
        print_partial("\a\033\\")
    else:
        print_partial("\a")


def print_data_image_in_iterm2(
    data: bytes,
    title: str
):
    """
    This understands how iterm2 wants to receive images.
    """
    assert isinstance(data, bytes), f"data must type bytes but it is of type {type(data)}"
    assert isinstance(title, str), f"title must be a str but it is of type {type(title)}"
    
    b64_data = base64.b64encode(data).decode('ascii')

    terminal = os.environ.get('TERM')
    print_osc(terminal)
    print_partial('1337;File=')
    args = []
    b64_file_name = base64.b64encode(title.encode('utf-8')).decode('ascii')
    args.append('name=' + b64_file_name)
    args.append('size=' + str(len(b64_data)))
    args.append("inline=1")
    print_partial(';'.join(args))
    print_partial(":")
    print_partial(b64_data)
    print_st(terminal)
    print("", file=sys.stderr)
