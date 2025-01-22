from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


import shutil

from pathlib import Path


def amcoff_add_many_copies_of_fixup_frames():
    print("Hello from amcoff_add_many_copies_of_fixup_frames")
    shared_dir = get_the_large_capacity_shared_directory()


    dst_dir = shared_dir / "fake_nba" / "fixups_multiplied"
    dst_dir.mkdir(exist_ok=True)

    # list all the folders that fixups are in:
    folder_strs = [
        "~/r/bos-gsw-2022-06-08-C01-30M_led/darren",
        "~/r/bos-cle-2024-05-09-youtube_led/anna",
    ]

    folders = [
        Path(folder_str).expanduser()
        for folder_str in folder_strs
    ]



    src_file_paths = []
    for folder in folders:
        print(folder)
        originals_in_folder = [
            p for p in folder.glob("*_original.jpg")
        ]
        masks_in_folder = [
            p for p in folder.glob("*_nonfloor.png")
        ]
        relevance_masks_in_folder = [
            p for p in folder.glob("*_relevance.png")
        ]
      
        assert len(originals_in_folder) == len(masks_in_folder)
        assert len(masks_in_folder) == len(relevance_masks_in_folder)
        src_file_paths.extend(
            originals_in_folder
        )
        src_file_paths.extend(
            masks_in_folder
        )
        src_file_paths.extend(
            relevance_masks_in_folder
        )
    
    for src_file_path in src_file_paths:
        for copy_index in range(20):
            variant_name = f"fixupsyo-{copy_index:02d}-{src_file_path.name}"
            print(f"{variant_name=}")
            dst_file_path = dst_dir / variant_name
            print(f"Copying\n{src_file_path}\nto\n{dst_file_path}\n")
            shutil.copy(
                src=src_file_path,
                dst=dst_file_path,
            )

    print(f"ls {dst_dir}")
          
if __name__ == "__main__":
    amcoff_add_many_copies_of_fixup_frames()
  


