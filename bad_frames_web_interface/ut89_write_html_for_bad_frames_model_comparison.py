from print_yellow import (
     print_yellow
)
from typing import Dict
from print_green import (
     print_green
)
from pathlib import Path




def ut89_write_html_for_bad_frames_model_comparison(
    list_of_lists: list[list[Dict[str, str]]],
    folder_that_web_paths_are_relative_to: Path,
):
    """
    This function writes the HTML file for the bad frames model comparison.
    It uses the list_of_lists generated above to create the HTML content.
    """
    assert folder_that_web_paths_are_relative_to.is_dir(), f"{folder_that_web_paths_are_relative_to} is not a directory"
    for lst in list_of_lists:
        for dct in lst:
            assert "original" in dct, f"{dct=} does not have 'original' key"
            assert "mask" in dct, f"{dct=} does not have 'mask' key"
            assert "name" in dct, f"{dct=} does not have 'name' key"
            original = dct["original"]
            mask = dct["mask"]
            name = dct["name"]
            original_file_path = folder_that_web_paths_are_relative_to / original
            assert original_file_path.exists(), f"{original_file_path} does not exist"
            mask_file_path = folder_that_web_paths_are_relative_to / mask
            assert mask_file_path.exists(), f"{mask_file_path} does not exist"
            
    print_green("Writing HTML for bad frames model comparison...")
        
    prepath = Path("/shared/www/pre.html").resolve()
    pre_content = prepath.read_text()
    postpath = Path("/shared/www/post.html").resolve()
    post_content = postpath.read_text()

    with open("/shared/www/index.html", "w") as f:
        f.write(pre_content)
        print("    const images = [", file=f)
        for lst in list_of_lists:
            print("        [", file=f)
            for x in lst:
                print(f"            {x},", file=f)
            print("        ],", file=f)
        print("    ];", file=f)
        f.write(post_content)
    
    print_yellow("Suggest that in a tmux called server you do:")
    print_green("cd /shared/www")
    print_green("python -m http.server 42857")
    print_yellow("Then you can visit the site at:")
    print_green("http://72.177.16.107:42857/index.html")