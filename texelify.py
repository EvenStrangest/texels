import os.path

import numpy as np
import imageio as iio
from skimage import color
from skimage.filters import butterworth, gaussian

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

    img_name = "tinylavi"
    img_path = "examples"

    glyphs = get_glyphs(font_pathname, font_size, text_color, background_color, text)

    img_pathname = os.path.join(img_path, f"{img_name}.png")
    img = iio.imread(img_pathname).astype(float)  # TODO: handle deprecation warning

    cutoff_frequency_ratio = 0.01
    # img_lpf = butterworth(img, cutoff_frequency_ratio, channel_axis=2,
    #                       high_pass=False, order=40)  # TODO: why no squared_butterworth=True and npad=0 ?
    img_lpf = gaussian(img, sigma=10, mode='nearest', preserve_range=True, truncate=4.0, channel_axis=2)
    iio.imsave(os.path.join(img_path, f"{img_name}-lpf.png"), img_lpf)

    img_gl = np.expand_dims(color.rgb2gray(img), axis=-1)
    iio.imsave(os.path.join(img_path, f"{img_name}-gl.png"), img_gl)

    def arbitrary_val_from_dict(d):
        return next(iter(d.values()))
    block_w, block_h, three = arbitrary_val_from_dict(glyphs).shape
    img_gl = crop_for_blocking(img_gl, block_w, block_h)
    blocks = blockify_2d(img_gl, block_w, block_h)

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

    iio.imsave(os.path.join(img_path, f"{img_name}-texels.png"), img_rec)

