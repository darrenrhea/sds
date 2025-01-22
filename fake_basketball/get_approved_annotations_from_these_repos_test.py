from get_approved_annotations_from_these_repos import (
     get_approved_annotations_from_these_repos
)

def test_get_approved_annotations_from_these_repos_1():

    repo_ids_to_use = [
        "bay-czv-2024-03-01_led",
        "bay-efs-2023-12-20_led",
        "bay-mta-2024-03-22-mxf_led",
        "bay-mta-2024-03-22-part1-srt_led",
        "bay-zal-2024-03-15-yt_led",
        "maccabi1080i_led",
        "munich1080i_led",
        "maccabi_fine_tuning",
    ]

    actual_annotations = get_approved_annotations_from_these_repos(
        repo_ids_to_use=repo_ids_to_use
    )
    
    for actual_annotation in actual_annotations:
        frame_index = actual_annotation["frame_index"]
        assert isinstance(frame_index, int)
        clip_id = actual_annotation["clip_id"]
        assert isinstance(clip_id, str)
        annotator = actual_annotation["annotator"]
        assert isinstance(annotator, str)
        annotation_id = actual_annotation["annotation_id"]
        assert isinstance(annotation_id, str)
        mask_file_path = actual_annotation["mask_file_path"]
        assert mask_file_path.is_file()
        original_file_path = actual_annotation["original_file_path"]
        assert original_file_path.is_file()
        repo_id = actual_annotation["repo_id"]
        assert isinstance(repo_id, str)

        print(f"{repo_id}=")
        print(f"{frame_index=}")
        print(f"{clip_id=}")
        print(f"{annotator=}")
        print(f"{annotation_id=}")
        print(f"{mask_file_path=}")
        print(f"{original_file_path=}")
        print("\n\n")

       
    print(f"total_annotations={len(actual_annotations)}")


    
if __name__ == "__main__":
    test_get_approved_annotations_from_these_repos_1()