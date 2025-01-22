import uuid
import pyperclip

def uuid_cli_tool():
    s = str(uuid.uuid4())
    print(s)
    pyperclip.copy(s)


if __name__ == "__main__":
    uuid_cli_tool()