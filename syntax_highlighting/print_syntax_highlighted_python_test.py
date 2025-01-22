from print_syntax_highlighted_python import print_syntax_highlighted_python

def print_syntax_highlighted_python_test():
    source_code = """
    def myfunc(x: int, y: int) -> int:
        return x + y

    for i in range(10):
        print(i)  # a comment
    """

    print_syntax_highlighted_python(source_code)

if __name__ == "__main__":
    print_syntax_highlighted_python_test()