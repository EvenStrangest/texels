import os.path

import numpy as np
import imageio as iio
from skimage import color
from skimage.filters import gaussian

from typeface_rendering import get_glyphs, GlyphRenderer
from vq_encode import crop_for_blocking, blockify_2d, deblockify_2d, blocks_to_matrix


class TexelEncoder:
    def __init__(self, font_pn, font_sz, txt, fg_img, bg_img, contour_img):
        self.font_pathname = font_pn
        self.font_size = font_sz
        self.text = txt

        text_color, background_color = "black", "white"
        self.glyph_blocks, self.glyph_names = get_glyphs(self.font_pathname, self.font_size,
                                                         text_color, background_color, self.text)
        self.glyph_blocks = {k: self.grayscale_and_remove_mean(v) for k, v in self.glyph_blocks.items()}

        self.fg_img = crop_for_blocking(fg_img, self.glyph_shape)
        self.bg_img = crop_for_blocking(bg_img, self.glyph_shape)
        self.contour_img = crop_for_blocking(contour_img, self.glyph_shape)

        self.fg_blocks = blockify_2d(self.fg_img, self.glyph_shape)
        self.bg_blocks = blockify_2d(self.bg_img, self.glyph_shape)
        self.contour_blocks = blockify_2d(self.contour_img, self.glyph_shape)

        self.glyph_renderer = GlyphRenderer(self.font_pathname, self.font_size, shape=self.glyph_shape)

        self.__glyphs_as_matrix = blocks_to_matrix(self.glyph_blocks.values())

    def encode(self):
        _texels = [[None] * self.texels_shape[1] for _ in range(self.texels_shape[0])]
        for i in range(self.texels_shape[0]):
            for j in range(self.texels_shape[1]):
                _texels[i][j] = self.__encode_block(self.fg_blocks[i][j],
                                                    self.bg_blocks[i][j],
                                                    self.contour_blocks[i][j])
        return _texels

    def __encode_block(self, _fg_block, _bg_block, _contour_block):
        glyph_selection = int(np.argmax(np.matmul(self.__glyphs_as_matrix, _contour_block.flatten())))
        glyph_selection = list(self.glyph_blocks.keys())[glyph_selection]  # TODO: this is barbaric?!

        def mean_color(im):  # TODO: algorithmically broken - fix it!
            # clr = np.mean(np.mean(im, axis=0), axis=0)
            clr = im[0, 0, :]
            return tuple(clr.astype(np.int8))
        txt_color = mean_color(_fg_block)
        bg_color = mean_color(_bg_block)

        texel = self.glyph_renderer.render(txt_color, bg_color, glyph_selection)

        return texel

    @property
    def glyph_shape(self):
        def arbitrary_val_from_dict(d):
            return next(iter(d.values()))
        return arbitrary_val_from_dict(self.glyph_blocks).shape

    @property
    def texels_shape(self):
        return len(self.contour_blocks), len(self.contour_blocks[0])

    @staticmethod
    def grayscale_and_remove_mean(im):
        im = color.rgb2gray(im)
        im = im - np.mean(im.flatten())
        return im


if __name__ == '__main__':

    font_pathname = "typefaces/liberation-mono/LiberationMono-Bold.ttf"
    font_size = 16
    text = "".join(map(chr, range(ord('a'), ord('z')))) + \
           "".join(map(chr, range(ord('A'), ord('Z')))) + \
           "".join(map(chr, range(ord('0'), ord('9')))) + "?,:{}-=_+.;|[]<>()/'!@#$%^&*`" + '"' "\\" + " "

    img_name = "tinylavi"
    img_path = "examples"

    img_pathname = os.path.join(img_path, f"{img_name}.png")
    img = iio.imread(img_pathname).astype(float)  # TODO: handle deprecation warning

    img_lpf = gaussian(img, sigma=12, mode='nearest', preserve_range=True, truncate=4.0, channel_axis=2)
    iio.imsave(os.path.join(img_path, f"{img_name}-lpf.png"), img_lpf)

    img_gl = np.expand_dims(color.rgb2gray(img), axis=-1)  # TODO: move expand dims to where it is needed
    iio.imsave(os.path.join(img_path, f"{img_name}-gl.png"), img_gl)

    encoder = TexelEncoder(font_pathname, font_size, text, img, img_lpf, img_gl)

    texels = encoder.encode()

    img_rec = deblockify_2d(texels)

    iio.imsave(os.path.join(img_path, f"{img_name}-texels.png"), img_rec)

