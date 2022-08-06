import os
from unicodedata import name as unicodename
from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__':
    font_name = "typefaces/liberation-mono/LiberationMono-Bold.ttf"
    font_size = 16
    text = "".join(map(chr, range(ord('a'), ord('z')))) + \
           "".join(map(chr, range(ord('A'), ord('Z')))) + \
           "".join(map(chr, range(ord('0'), ord('9')))) + "?,:{}-=_+.;|[]<>()/'!@#$%^&*`" + '"' "\\" + " "
    text_color = "black"
    background_color = "white"

    font = ImageFont.truetype(font_name, font_size)

    test_img = Image.new('RGB', (100, 100))
    test_draw = ImageDraw.Draw(test_img)
    bboxes = []
    for c in text:
        bb = test_draw.textbbox((0, 0), c,
                                font, anchor=None, spacing=4, align='left', direction=None,
                                features=None, language=None, stroke_width=0, embedded_color=False)
        bboxes.append(bb)
    del bb

    maximal_bb = list(zip(*bboxes))
    maximal_bb = [min(maximal_bb[0]), min(maximal_bb[1]), max(maximal_bb[2]), max(maximal_bb[3])]
    maximal_glyph_size = (maximal_bb[2] - maximal_bb[0], maximal_bb[3] - maximal_bb[1])

    typeface_cache_pathname = "typeface-cache"
    os.makedirs(typeface_cache_pathname, exist_ok=True)

    for c in text:
        img = Image.new('RGB', maximal_glyph_size, background_color)
        d = ImageDraw.Draw(img)
        d.text((0, 0), c, fill=text_color, font=font)

        img.save(os.path.join(typeface_cache_pathname, f"char_{unicodename(c)}.png"))
