import os.path

import numpy as np
import imageio as iio
from skimage import color

from typeface_rendering import get_glyphs
from vq_encode import crop_for_blocking, blockify_2d, deblockify_2d, blocks_to_matrix


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

    img = iio.imread(img_pathname).astype(float)  # TODO: handle deprecation warning

    img = np.expand_dims(color.rgb2gray(img), axis=-1)

    def arbitrary_val_from_dict(d):
        return next(iter(d.values()))
    block_w, block_h, three = arbitrary_val_from_dict(glyphs).shape
    img = crop_for_blocking(img, block_w, block_h)
    blocks = blockify_2d(img, block_w, block_h)

    def grayscale_and_remove_mean(im):
        im = color.rgb2gray(im)
        im = im - np.mean(im.flatten())
        return im
    glyphs_as_matrix = blocks_to_matrix(list(map(grayscale_and_remove_mean, glyphs.values())))

    texels_shape = len(blocks), len(blocks[0])
    # glyph_selection_dtype = np.int8
    # assert len(glyphs) < np.iinfo(glyph_selection_dtype).max
    texel_glyph_selection = [[-1] * texels_shape[1] for i in range(texels_shape[0])]
    for i in range(texels_shape[0]):
        for j in range(texels_shape[1]):
            texel_glyph_selection[i][j] = int(np.argmax(np.matmul(glyphs_as_matrix, blocks[i][j].flatten())))

    def glyph_lookup(sel):
        return list(glyphs.values())[sel]
    texels = [list(map(glyph_lookup, sl)) for sl in texel_glyph_selection]

    img_rec = deblockify_2d(texels)

    iio.imsave(os.path.join("examples", "tinylavi-texels.png"), img_rec)

