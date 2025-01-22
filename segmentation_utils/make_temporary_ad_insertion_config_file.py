import better_json as bj
import tempfile


def make_temporary_ad_insertion_config_file(
    first_frame_index,
    last_frame_index,
    tracking_attempt_id,
    masking_attempt_id,
    insertion_attempt_id
):
    assert isinstance(first_frame_index, int)
    assert isinstance(last_frame_index, int)
    assert last_frame_index >= first_frame_index

    assert masking_attempt_id in ["final_bw", "temp"]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fp:
        out_json_tempfile_name = fp.name

    jsonable = {
        "clip_id": "gsw1",
        "tracking_attempt_id": tracking_attempt_id,
        "masking_attempt_id": masking_attempt_id,
        "first_frame": first_frame_index,
        "last_frame": last_frame_index,
        "insertion_attempt_id": insertion_attempt_id,
        "draw_wireframe": False,
        "save_frames": True,
        "filter_masks": 1,
        "ads": [
            {
                "png_path": "~/awecom/data/ads/awecom_r3.png",
                "x_center": 49,
                "y_center": 0,
                "width": 1.5,
                "height": 9.59090909090909
            },
            {
                "png_path": "~/awecom/data/ads/RSN_VS-3PT-A_Dr-Pepper_v03_fixed.png",
                "x_center": 42.5,
                "y_center": 0,
                "width": 8.5,
                "height": 7
            },
            {
                "png_path": "~/awecom/data/ads/ESPN_VS-3PT-A_TDBank_v01_fixed.png",
                "x_center": -43,
                "y_center": 0,
                "width": 5,
                "height": 5
            },
            {
                "png_path": "~/awecom/data/ads/NBCSB_VS-FSL_SamAdams_v01_fixed.png",
                "x_center": -37.58333,
                "y_center": 15,
                "width": 12.8333,
                "height": 7
            },
            {
                "png_path": "~/awecom/data/ads/VS-TBSL_Putnam_v02_fixed.png",
                "x_center": 37.58333,
                "y_center": -15,
                "width": 14,
                "height": 3.5
            },
            {
                "png_path": "~/awecom/data/ads/VS-TBSL_Chime_v01_fixed.png",
                "x_center": 37.58333,
                "y_center": 15,
                "width": 12.8333,
                "height": 5
            },
            {
                "png_path": "~/awecom/data/ads/NBCSB_VS-3PT-A_Arbella_v01_fixed.png",
                "x_center": -11,
                "y_center": -16,
                "width": 12.8333,
                "height": 6.5
            },
            {
                "png_path": "~/awecom/data/ads/FSSW_VS_FSL_Whataburger_v01_fixed.png",
                "x_center": -37.58333,
                "y_center": -15,
                "width": 14.8333,
                "height": 4
            }
        ]
    }

    bj.dump(out_json_tempfile_name, obj=jsonable)
    return out_json_tempfile_name