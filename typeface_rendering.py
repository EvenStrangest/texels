import os
import numpy as np
from unicodedata import name as unicode_name
from PIL import Image, ImageDraw, ImageFont
from skimage import io as iio


class GlyphRenderer:
    def __init__(self, f_pathname, f_size, shape):
        self.f_pathname = f_pathname
        self.f_size = f_size
        self.shape = shape[1], shape[0]

        self.font = ImageFont.truetype(f_pathname, f_size)

    def render(self, txt_color, bg_color, _chr):
        _im = Image.new('RGB', self.shape, bg_color)
        ImageDraw.Draw(_im).text((0, 0), _chr, fill=txt_color, font=self.font)
        return np.array(_im)


def get_glyphs(f_pathname, f_size, txt_color, bg_color, txt):

    renderer = GlyphRenderer(f_pathname, f_size, shape=(None, None))

    test_img = Image.new('RGB', (100, 100))
    test_draw = ImageDraw.Draw(test_img)
    bboxes = []
    for _c in txt:
        bb = test_draw.textbbox((0, 0), _c,
                                renderer.font, anchor=None, spacing=0, align='left', direction=None,
                                features=None, language=None, stroke_width=0, embedded_color=False)
        bboxes.append(bb)
    del bb

    maximal_bb = list(zip(*bboxes))
    maximal_bb = [min(maximal_bb[0]), min(maximal_bb[1]), max(maximal_bb[2]), max(maximal_bb[3])]
    maximal_glyph_size = (maximal_bb[2] - maximal_bb[0], maximal_bb[3] - maximal_bb[1])

    renderer = GlyphRenderer(f_pathname, f_size, shape=maximal_glyph_size)

    _glyphs = dict()
    _glyph_names = dict()
    for _c in txt:
        img = renderer.render(txt_color, bg_color, _c)
        _glyphs[_c] = img
        _glyph_names[_c] = unicode_name(_c)

    return _glyphs, _glyph_names


if __name__ == '__main__':

    font_pathname = "typefaces/liberation-mono/LiberationMono-Bold.ttf"
    font_size = 16
    text = "".join(map(chr, range(ord('a'), ord('z')))) + \
           "".join(map(chr, range(ord('A'), ord('Z')))) + \
           "".join(map(chr, range(ord('0'), ord('9')))) + "?,:{}-=_+.;|[]<>()/'!@#$%^&*`" + '"' "\\" + " "
    text_color = "black"
    background_color = "white"

    glyphs, glyph_names = get_glyphs(font_pathname, font_size, text_color, background_color, text)

    typeface_cache_pathname = "typeface-cache"
    os.makedirs(typeface_cache_pathname, exist_ok=True)
    for c, im in glyphs.items():
        iio.imsave(os.path.join(typeface_cache_pathname, f"char_{glyph_names[c]}.png"), im)

