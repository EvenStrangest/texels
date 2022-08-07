import numpy as np
import imageio as imgio
from skimage import color

from typeface_rendering import get_glyphs
from vq_encode import crop_for_blocking, blockify_2d, deblockify_2d


if __name__ == '__main__':

    font_pathname = "typefaces/liberation-mono/LiberationMono-Bold.ttf"
    font_size = 16
    text = "".join(map(chr, range(ord('a'), ord('z')))) + \
           "".join(map(chr, range(ord('A'), ord('Z')))) + \
           "".join(map(chr, range(ord('0'), ord('9')))) + "?,:{}-=_+.;|[]<>()/'!@#$%^&*`" + '"' "\\" + " "
    text_color = "black"
    background_color = "white"

    img_pathname = "examples/tinylavi.png"

    glyphs = get_glyphs(font_pathname, font_size, text_color, background_color, text)

    img = imgio.imread(img_pathname).astype(float)  # TODO: handle deprecation warning

    img = np.expand_dims(color.rgb2gray(img), axis=-1)

    def arbitrary_from_dict(d):
        return d[next(iter(d))]
    block_w, block_h = arbitrary_from_dict(glyphs)[1].shape
    img = crop_for_blocking(img, block_w, block_h)
    blocks = blockify_2d(img, block_w, block_h)

    # TODO: do stuff

    img_rec = deblockify_2d(blocks)

