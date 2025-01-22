from PIL import Image
from pathlib import Path

clip_name = "gsw1"
background_dir = Path(f"~/r/final_gsw1/scorebug_final_segmentation").expanduser()
foreground_dir = Path(f"~/r/gsw1/scorebug").expanduser()
out_dir = Path(f"~/r/final_gsw1/scorebug_final_segmentation").expanduser()

image_ids = [
    # "147568",
    # "150000",
    # "150291",
    # "150331",
    # "150348",
    # "150374",
    # "150893", #
    # "150900", #
    # "150953", #
    # "151004",
    # "151037",
    # "151158",
    # "151359",
    # "151382",
    # "151567",
    # "153700",
    # "153752"
    # "148785",
    # "150000",
    # "150291",
    # "150331",
    # "150348",
    # "150374",
    # "150900",
    # "150953",
    # "151004",
    # "151037",
    # "151158",
    # "151359",
    # "151382",
    # "151567",
    # "153700",
    # "153752",
    # "154000",
    # "159064",
    # "159128",
    # "160352",
    # "160800",
    # "161000",
    # "163266",
    # "163576",
    # "163768",
    # "163833",
    # "163851",
    # "163973",
    # "164044",
    # "164244",
    # "164740",
    # "165013",
    # "165719",
    # "165723",
    # "167200",
    # "168178",
    # "169348",
    # "169370",
    # "170501",
    # "170506",
    # "172946",
    # "174565",
    # "193372",
    # "193877",
    # "194127",
    # "194339",
    # "256429",
    # "296895",
    # "303177",
    # "305648",
    # "305706",
    # "328521",
    # "328554",
    # "328980",
    # "331545",
    # "331849",
    # "332931",
    # "333322",
    # "498907",
    # "498934",
    # "499171",
    # "499201",
    # "499227",
    # "499325",
    # "582534",
    # "582538",
    # "585197",
    "585978"
]

for image_id in image_ids:
    background = Image.open(f"{background_dir}/{clip_name}_{int(image_id):06d}_nonfloor.png").convert('RGBA')
    alpha = background.getchannel('A')

    foreground = Image.open(Path(f"{foreground_dir}/{clip_name}_{image_id}_color.png").expanduser()).convert('RGBA')

    # Do an alpha composite of foreground over background
    background.paste(foreground, (0, 0), foreground)
    background.putalpha(alpha)

    # Display the alpha composited image
    background.save(f"{background_dir}/{clip_name}_{int(image_id):06d}_nonfloor.png")