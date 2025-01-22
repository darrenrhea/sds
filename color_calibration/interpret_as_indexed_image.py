import numpy as np
import sys

from pathlib import Path

from print_image_in_iterm2 import print_image_in_iterm2
import PIL.Image


def interpret_as_paint_by_numbers(
    context_id,
    quantized_image_file_path,
    output_path
):
    """
    Give a the file path of a index-color/palette-based png,
    like those exported from GIMP when you go
    Image > Mode > Indexed
    """
    image_pil = PIL.Image.open(quantized_image_file_path)

    quantized_image_np = np.array(image_pil)
    print(quantized_image_np.shape)
    print(quantized_image_np.dtype)
    height, width = quantized_image_np.shape

   
    rgba_image_np = np.zeros(
        shape=(height, width, 4),
        dtype=np.uint8
    )
    

    # rgba_image_np[quantized_image_np == 0, 3] = 255

       
    if context_id == "22-23_CLE_CORE":

        num_colors = 10
        
        index_to_region_name = {
            0: "six_circle_and_rocket",
            1: "tp_circ_and_other_dark_maroon_lines_and_areas",
            2: "apron_paint_etc",
            3: "halfway_between_maroon_and_wood_region",
            4: "another_halfway_region",
            5: "gold_region",
            6: "measure_zero_region",
            7: "measure_zero_region_2",
            8: "wood_region",
            9: "white_region",
            10: "what_is",
        }



        region_name_to_color_name = dict(
            six_circle_and_rocket="six_circle_black",
            tp_circ_and_other_dark_maroon_lines_and_areas="cleveland_dark_maroon",
            apron_paint_etc="cleveland_light_maroon",
            halfway_between_maroon_and_wood_region="halfway_between_maroon_and_wood_color",
            another_halfway_region="wood_color",
            white_region="cleveland_white_color",
            gold_region="cleveland_gold",
            measure_zero_region="bright_blue",
            measure_zero_region_2="bright_blue",
            tp_circ_interior="tpcirc_interior_purple",
            lines_of_the_basketball="basketball_lines_color",
            apron="cleveland_light_maroon",
            wood_region="wood_color",
            what_is="bright_blue",
        )

        # get out digital color meter or something:
        color_name_to_rgba = dict(
            six_circle_black=[50, 50, 60, 255], #yes
            cleveland_dark_maroon=[114, 46, 71, 255], # yes
            cleveland_light_maroon=[112, 55, 84, 255], # yes 
            cleveland_gold=[170, 145, 104, 255], # yes
            cleveland_white_color=[220, 220, 220, 255], # yes
            halfway_between_maroon_and_wood_color=[0, 255, 0, 255],
            green=[255, 0, 0, 255],
            wood_color=[213, 188, 163, 255],
            bright_red=[255, 0, 0, 255],
            bright_green=[0, 255, 0, 255],
            bright_blue=[0, 0, 255, 255],
            lexus_black=[50, 34, 34, 255],
            paint_purple=[78, 46, 111, 255],
            tpcirc_interior_purple=[82, 53, 79, 255],
            basketball_lines_color=[212, 158, 58, 255],
            alt_white=[210, 184, 167, 255],
            basketball_gold=[212, 158, 58, 255],
            tpcirc_boundary_color=[115, 89, 101, 255],
            white_of_legal_lines=[214, 221, 213, 255],
            cyan=[0, 255, 255, 255],
        )


    elif context_id == "22-23_LAL_CORE":
        num_colors = 12

        index_to_region_name = {
            0: "weird_little_region",
            1: "lexus_font",
            2: "paint",
            3: "tp_circ_interior",
            4: "lines_of_the_basketball",
            5: "apron",
            6: "basketball_center_logo",
            7: "photographer_lines",
            8: "more_apron",
            9: "boundary_of_tpcirc",
            10: "most_wood",
            11: "legal_lines",
        }

        region_name_to_color_name = dict(
            weird_little_region="gold",
            lexus_font="lexus_black",
            paint="paint_purple",
            tp_circ_interior="tpcirc_interior_purple",
            lines_of_the_basketball="basketball_lines_color",
            apron="gold",
            basketball_center_logo="gold",
            photographer_lines="alt_white",
            more_apron="gold",
            boundary_of_tpcirc="tpcirc_boundary_color",
            most_wood="wood",
            legal_lines="white_of_legal_lines"
        )

        # get out digital color meter or something:
        color_name_to_rgba = dict(
            bright_red=[255, 0, 0, 255],
            lexus_black=[50, 34, 34, 255],
            paint_purple=[78, 46, 111, 255],
            tpcirc_interior_purple=[82, 53, 79, 255],
            basketball_lines_color=[212, 158, 58, 255],
            gold=[213, 161, 73, 255],
            alt_white=[210, 184, 167, 255],
            basketball_gold=[212, 158, 58, 255],
            tpcirc_boundary_color=[115, 89, 101, 255],
            wood=[210, 184, 167, 255],
            white_of_legal_lines=[254, 234, 243, 255],
            cyan=[0, 255, 255, 255],
        )
   
    elif context_id == "22-23_OKC_CORE":

        num_colors = 9
        
        index_to_region_name = {
            0: "transparent_region",
            1: "center_word_of_paycom",
            2: "paint",
            3: "orange_region",
            4: "green_region_of_paycom",
            5: "wood_screwed_up_by_backboards_shadow",
            6: "basketball_center_logo",
            7: "most_wood",
            8: "legal_lines",
           
        }

        region_name_to_color_name = dict(
            transparent_region="transparent",
            center_word_of_paycom="paycom_center_black",
            paint="blue_paint",
            orange_region="orange",
            green_region_of_paycom="green",
            tp_circ_interior="tpcirc_interior_purple",
            lines_of_the_basketball="basketball_lines_color",
            apron="gold",
            basketball_center_logo="gold",
            most_wood="wood_color",
            wood_screwed_up_by_backboards_shadow="wood_color",
            legal_lines="white_of_legal_lines"
        )

        # get out digital color meter or something:
        color_name_to_rgba = dict(
            transparent=[0, 0, 0, 0],
            paycom_center_black=[50, 50, 30, 255],
            blue_paint=[40, 93, 159, 255],
            orange=[176, 65, 64, 255],
            green=[29, 139, 71, 255],
            wood_color=[184, 171, 145, 255],
            bright_red=[255, 0, 0, 255],
            lexus_black=[50, 34, 34, 255],
            paint_purple=[78, 46, 111, 255],
            tpcirc_interior_purple=[82, 53, 79, 255],
            basketball_lines_color=[212, 158, 58, 255],
            gold=[213, 161, 73, 255],
            alt_white=[210, 184, 167, 255],
            basketball_gold=[212, 158, 58, 255],
            tpcirc_boundary_color=[115, 89, 101, 255],
            white_of_legal_lines=[214, 221, 213, 255],
            cyan=[0, 255, 255, 255],
        )
    
    elif context_id == "22-23_NYK_CITY":

        num_colors = 11
        
        index_to_region_name = {
            0: "transparent_region",
            1: "black_paint",
            2: "paint",
            3: "tp_circles",
            4: "specular_on_black",
            5: "more_black_stuff",
            6: "orange_region",
            7: "yet_more_specular_on_black",
            8: "also_wood",
            9: "most_wood",
            10: "out_of_bounds_lines",
        }

        region_name_to_color_name = dict(
            transparent_region="transparent_color",
            black_paint="nyk_black",
            paint="nyk_black",
            tp_circles="nyk_city_blue",
            specular_on_black="nyk_black",
            more_black_stuff="nyk_black",
            orange_region="nyk_city_orange",
            yet_more_specular_on_black="nyk_black",
            more_black_paint="nyk_black",
            legal_lines="nyk_wood_color",
            also_wood="nyk_wood_color",
            most_wood="nyk_wood_color",
            out_of_bounds_lines="nyk_city_out_of_bounds_line_white",

        )
        
        # get out digital color meter or something:
        color_name_to_rgba = dict(
            transparent_color=[0, 0, 0, 0],
            chase_blue = [42, 47, 93, 255],
            nyk_city_blue = [ 79, 90, 129, 255],
            nyk_city_out_of_bounds_line_white = [ 180, 180, 180, 255],  # correct for NYK
            nyk_city_orange=[227, 103, 48, 255], # correct for NYK
            nyk_wood_color=[212, 176, 118, 255], # correct for NYK
            nyk_black=[41, 41, 41, 255],  # correct for NYK
            paycom_center_black=[50, 50, 30, 255],
            bright_red=[255, 0, 0, 255],
            bright_yellow=[255, 255, 0, 255],
            lexus_black=[50, 34, 34, 255],
            paint_purple=[78, 46, 111, 255],
            tpcirc_interior_purple=[82, 53, 79, 255],
            basketball_lines_color=[212, 158, 58, 255],
            gold=[213, 161, 73, 255],
            alt_white=[210, 184, 167, 255],
            basketball_gold=[212, 158, 58, 255],
            tpcirc_boundary_color=[115, 89, 101, 255],
            cyan=[0, 255, 255, 255],
        )

    # generic stuff from now on:
    index_to_rgba = {
        index : color_name_to_rgba[region_name_to_color_name[index_to_region_name[index]]]
        for index in range(num_colors)
    }

    for index in range(num_colors):
        rgba_image_np[quantized_image_np == index, :] = index_to_rgba[index]

    output_pil = PIL.Image.fromarray(rgba_image_np)

    output_pil.save(str(output_path))

    print(f"See {output_path}")
   
    


def main():
    

    context_id = sys.argv[1]

    assert context_id in [
        "22-23_LAL_CORE",
        "22-23_OKC_CORE",
        "22-23_NYK_CITY",
        "22-23_CLE_CORE",
    ]
    
    # an image whose colors only take on a small number of values:
    quantized_image_file_path = Path(f"{context_id}_gimp_indexed_palette.png")
    output_path = Path(f"{context_id}_painted_by_numbers.png")

    interpret_as_paint_by_numbers(
        context_id=context_id,
        quantized_image_file_path=quantized_image_file_path,
        output_path=output_path
    )

    print(f"rsync -rP out.png lam:/mnt/nas/volume1/videos/temp/22-23_OKC_CORE_painted_by_numbers.png")
    print("Within the container:")

    print(
        f"cp /mnt/nas/volume1/videos/temp/22-23_OKC_CORE_painted_by_numbers.png /home/dev/tracker/registries/nba/floor_models/22-23_OKC_CORE_floortexture.png"
    )

   

if __name__ == "__main__":
    main()