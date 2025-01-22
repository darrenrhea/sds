from pathlib import Path
from colorama import Fore, Style, init
import io
from typing import List
from get_lines_without_line_endings import get_lines_without_line_endings

def get_lines_without_line_endings_test():
    question_answer_pairs = [
        ("a\nb\nc\n", ["a", "b", "c"]),
        ("a\r\nb\nc", ["a", "b", "c"]),
        ("a\nb\nc\n", ["a", "b", "c"]),
        ("a\nb\nc", ["a", "b", "c"]),
    ]

    for content, answer in question_answer_pairs:
        file_path = Path("test.txt")
        file_path.write_text(content)
        lines_without_line_endings = get_lines_without_line_endings(file_path)
        
        assert (
            isinstance(lines_without_line_endings, List)
        ), f"ERROR: lines_without_line_endings is not a list???"

        assert lines_without_line_endings == answer, lines_without_line_endings
    
    print(f"{Fore.GREEN}get_lines_without_line_endings_test passed {len(question_answer_pairs)} tests{Style.RESET_ALL}")

if __name__ == "__main__":
    get_lines_without_line_endings_test()