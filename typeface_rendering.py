import os
import numpy as np
from unicodedata import name as unicodename
from PIL import Image, ImageDraw, ImageFont
from skimage import io as imgio


def get_glyphs(f_pathname, f_size, txt_color, bg_color, txt):

    font = ImageFont.truetype(f_pathname, f_size)

    test_img = Image.new('RGB', (100, 100))
    test_draw = ImageDraw.Draw(test_img)
    bboxes = []
    for c in txt:
        bb = test_draw.textbbox((0, 0), c,
                                font, anchor=None, spacing=4, align='left', direction=None,
                                features=None, language=None, stroke_width=0, embedded_color=False)
        bboxes.append(bb)
    del bb

    maximal_bb = list(zip(*bboxes))
    maximal_bb = [min(maximal_bb[0]), min(maximal_bb[1]), max(maximal_bb[2]), max(maximal_bb[3])]
    maximal_glyph_size = (maximal_bb[2] - maximal_bb[0], maximal_bb[3] - maximal_bb[1])

    _glyphs = dict()
    for c in txt:
        img = Image.new('RGB', maximal_glyph_size, bg_color)
        ImageDraw.Draw(img).text((0, 0), c, fill=txt_color, font=font)

        _glyphs[unicodename(c)] = np.array(img)

    return _glyphs


if __name__ == '__main__':

    font_pathname = "typefaces/liberation-mono/LiberationMono-Bold.ttf"
    font_size = 16
    text = "".join(map(chr, range(ord('a'), ord('z')))) + \
           "".join(map(chr, range(ord('A'), ord('Z')))) + \
           "".join(map(chr, range(ord('0'), ord('9')))) + "?,:{}-=_+.;|[]<>()/'!@#$%^&*`" + '"' "\\" + " "
    text_color = "black"
    background_color = "white"

    glyphs = get_glyphs(font_pathname, font_size, text_color, background_color, text)

    typeface_cache_pathname = "typeface-cache"
    os.makedirs(typeface_cache_pathname, exist_ok=True)
    for nm, im in glyphs.items():
        imgio.imsave(os.path.join(typeface_cache_pathname, f"char_{nm}.png"), im)

