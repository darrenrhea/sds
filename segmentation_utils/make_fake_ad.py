from PIL import Image, ImageDraw, ImageFont

image = Image.new("RGB", (500, 100), "white")
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
draw = ImageDraw.Draw(image)
position = (10, 10)
# text = "PNEUSAGRICOLES"
text = "SPORTWETTEN"

left, top, right, bottom = draw.textbbox(position, text, font=font)
draw.rectangle((left-5, top-5, right+5, bottom+5), fill="green")
draw.text(position, text, font=font, fill="black")
# bbox = draw.textbbox(position, text, font=font)
# draw.rectangle(bbox, fill="red")
# draw.text(position, text, font=font, fill="black")

image.save("sportwetten_fake.png","PNG")