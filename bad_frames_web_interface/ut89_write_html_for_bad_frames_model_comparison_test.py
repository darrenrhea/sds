from print_red import (
     print_red
)
from ut89_write_html_for_bad_frames_model_comparison import (
     ut89_write_html_for_bad_frames_model_comparison
)
from color_print_json import (
     color_print_json
)
from print_yellow import (
     print_yellow
)
from typing import Dict
from print_green import (
     print_green
)
from pathlib import Path



def test_ut89_write_html_for_bad_frames_model_comparison_1():
    final_model_ids = [
        "summerleague2025floorrev1epoch10",
        "summerleague2025floorrev2epoch23",
    ]
    
    clip_id_frame_index_pairs = [
        ("slday3game1", 75),
        ("slday3game1", 2756),
        ("slday10game1", 0),
        ("slday10game1", 2808),
    ]
    
    def clip_id_frame_index_model_id_to_original_path_str(
        clip_id: str,
        frame_index: int,
        final_model_id: str,
    ) -> Path:
        """
        ln -s /shared/preannotations/fixups /shared/www/fixups
        """
        src_path = Path(
            f"/shared/preannotations/fixups/{clip_id}/{final_model_id}/{clip_id}_{frame_index:06d}_original.jpg"
        )
        staging_folder = Path("/shared/www")
        assert src_path.exists(), f"{src_path} does not exist"
        relative_path = src_path.relative_to("/shared/preannotations")
        relative_path_str = str(relative_path)
        reconstituted = staging_folder / relative_path
        assert reconstituted.exists(), f"{reconstituted} does not exist"
        return relative_path_str
    
    def clip_id_frame_index_model_id_to_mask_path_str(
        clip_id: str,
        frame_index: int,
        final_model_id: str,
    ) -> Path:
        """
        ln -s /shared/preannotations/fixups /shared/www/fixups
        """
        staging_folder = Path("/shared/www")
        src_path = Path(
            f"/shared/preannotations/fixups/{clip_id}/{final_model_id}/{clip_id}_{frame_index:06d}_nonfloor.png"
        )
        assert src_path.exists(), f"{src_path} does not exist"
        relative_path = src_path.relative_to("/shared/preannotations")
        relative_path_str = str(relative_path)                                      
        return relative_path_str
                                      
    list_of_lists = [
        [
            {
                "original": clip_id_frame_index_model_id_to_original_path_str(
                    clip_id=clip_id,
                    frame_index=frame_index,
                    final_model_id=final_model_id,
                ),
                "mask": clip_id_frame_index_model_id_to_mask_path_str(
                    clip_id=clip_id,
                    frame_index=frame_index,
                    final_model_id=final_model_id,
                ),
                "name": clip_id_frame_index_model_id_to_original_path_str(
                    clip_id=clip_id,
                    frame_index=frame_index,
                    final_model_id=final_model_id,
                )
            }
            for final_model_id in final_model_ids
        ]
        for clip_id, frame_index in clip_id_frame_index_pairs
    ]

    color_print_json(list_of_lists)
    folder_that_web_paths_are_relative_to = Path("/shared/www")
    
    ut89_write_html_for_bad_frames_model_comparison(
        list_of_lists=list_of_lists,
        folder_that_web_paths_are_relative_to=folder_that_web_paths_are_relative_to,
    )
  
if __name__ == "__main__":
    test_ut89_write_html_for_bad_frames_model_comparison_1()
    print_green("test_ut89_write_html_for_bad_frames_model_comparison_1 completed successfully.")