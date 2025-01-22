from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
import argparse
import shutil
import textwrap


def mci_migrate_clip_id_cli_tool():
    """
    When you want to migrate a clip_id, use this.
    """
   
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            This tool is for migrating a clip_id from an old value to a new value.
            
            Example usage:
            
            mci_migrate_clip_id_cli_tool --old 123 --new 456
            """
        )
    )

    parser.add_argument(
        "--old",
        required=True,
        help="The old clip_id, what it is currently called",
    )

    parser.add_argument(
        "--new",
        required=True,
        help="The new clip_id to migrate to",
    )



    args = parser.parse_args()
    
    shared_dir = get_the_large_capacity_shared_directory()
    assert shared_dir.is_dir(), f"ERROR: {shared_dir} does not exist!"
    
    clips_dir = shared_dir / "clips"
    assert clips_dir.is_dir(), f"ERROR: {clips_dir} does not exist!"
    
    old_clip_id = args.old
    new_clip_id = args.new

    print(f"Migrating the old clip_id: {old_clip_id}")
    print(f"To the new clip_id: {new_clip_id}")

    old_clip_dir = shared_dir / "clips" / old_clip_id
    new_clip_dir = shared_dir / "clips" / new_clip_id
    old_frames_dir = old_clip_dir / "frames"
    new_frames_dir = new_clip_dir / "frames"


    assert old_clip_dir.exists(), f"ERROR: {new_clip_dir} does not exist!"
    assert old_frames_dir.exists(), f"ERROR: {new_frames_dir} does not exist!"

    assert not new_clip_dir.exists(), f"ERROR: {new_clip_dir} already exists!"


    for frame_path in old_frames_dir.iterdir():
        assert frame_path.name.startswith(old_clip_id), f"ERROR: {frame_path} does not start with {old_clip_id}"

    shutil.move(
        src=old_clip_dir,
        dst=new_clip_dir
    )


    for frame_path in new_frames_dir.iterdir():
        assert frame_path.name.startswith(old_clip_id), f"ERROR: {frame_path} does not start with {old_clip_id}"
        
        the_tail = frame_path.name[len(old_clip_id):]
        new_name = new_clip_id + the_tail
        new_frame_path = frame_path.with_name(new_name)
        print(f"Renaming {frame_path} to {new_frame_path}")
        frame_path.rename(new_frame_path)
    

    # rename any inferences:
    inferences_dir = shared_dir / "inferences"
    for mask_path in inferences_dir.iterdir():
        if not mask_path.is_file():
            continue
        if not mask_path.name.startswith(old_clip_id):
            continue
        assert mask_path.name.startswith(old_clip_id), f"ERROR: {mask_path} does not start with {old_clip_id}"

        the_tail = mask_path.name[len(old_clip_id):]
        new_name = new_clip_id + the_tail
        new_mask_path = mask_path.with_name(new_name)
        print(f"Renaming {mask_path} to {new_mask_path}")
        mask_path.rename(new_mask_path)
