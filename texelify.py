import os.path

import numpy as np
import imageio as iio
from skimage import color
from skimage.filters import gaussian

from typeface_rendering import get_glyphs, GlyphRenderer
from vq_encode import crop_for_blocking, blockify_2d, deblockify_2d, blocks_to_matrix


class TexelEncoder:
    def __init__(self, font_pn, font_sz, chars, fg_img, bg_img, contour_img):
        self.font_pathname = font_pn
        self.font_size = font_sz
        self.chars = chars

        text_color, background_color = "black", "white"
        self.glyph_blocks, self.glyph_names = get_glyphs(self.font_pathname, self.font_size,
                                                         text_color, background_color, self.chars)
        self.glyph_blocks = {k: self.grayscale_and_remove_mean(v) for k, v in self.glyph_blocks.items()}

        self.fg_img = crop_for_blocking(fg_img, self.glyph_shape)
        self.bg_img = crop_for_blocking(bg_img, self.glyph_shape)
        self.contour_img = crop_for_blocking(contour_img, self.glyph_shape)

        def normalize_to_range_0_255(im):
            _min, _max = np.min(im.flatten()), np.max(im.flatten())
            im = (im - _min) / _max * 255.0
            # assert np.all(im >= 0) and np.all(im <= 255)
            return im
        self.fg_img = normalize_to_range_0_255(self.fg_img)
        self.bg_img = normalize_to_range_0_255(self.bg_img)
        # TODO: consider moving grayscale_and_remove_mean of contour_img here from outside

        self.fg_blocks = blockify_2d(self.fg_img, self.glyph_shape)
        self.bg_blocks = blockify_2d(self.bg_img, self.glyph_shape)
        self.contour_blocks = blockify_2d(self.contour_img, self.glyph_shape)

        self.glyph_renderer = GlyphRenderer(self.font_pathname, self.font_size, shape=self.glyph_shape)

        self._glyphs_as_matrix = blocks_to_matrix(self.glyph_blocks.values())

    def __str__(self):
        _, font_name = os.path.split(self.font_pathname)
        return f"{font_name=}, {self.font_size=}, {self.chars=}"

    def encode(self):
        _texels = [[None] * self.texels_shape[1] for _ in range(self.texels_shape[0])]
        for i in range(self.texels_shape[0]):
            for j in range(self.texels_shape[1]):
                _texels[i][j] = self.__encode_block(self.fg_blocks[i][j],
                                                    self.bg_blocks[i][j],
                                                    self.contour_blocks[i][j])
        return _texels

    def __encode_block(self, _fg_block, _bg_block, _contour_block):
        glyph_selection = int(np.argmax(np.abs(np.matmul(self._glyphs_as_matrix, _contour_block.flatten()))))
        glyph_selection = list(self.glyph_blocks.keys())[glyph_selection]  # TODO: this is barbaric?!

        glyph_template = self.glyph_renderer.render(txt_color=(255, 255, 255), bg_color=(0, 0, 0), _chr=glyph_selection)
        glyph_template = glyph_template.astype(float)[:, :, 0] / 255.0

        def mean_color(im, mask=None):
            if mask is None or np.sum(mask.flatten()) == 0:
                mask = np.ones(shape=im.shape[0:2])
            masked_img = np.expand_dims(mask, axis=-1) * im
            clr = np.mean(np.mean(masked_img, axis=0), axis=0) / np.mean(mask.flatten())
            return tuple(clr.astype(np.uint8))
        txt_color = mean_color(_fg_block, glyph_template)
        bg_color = mean_color(_bg_block)
        # assert np.all(np.array(txt_color) >= 0) and np.all(np.array(txt_color) <= 255)
        # assert np.all(np.array(bg_color) >= 0) and np.all(np.array(bg_color) <= 255)

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

    @staticmethod
    def deblockify(blks):
        return deblockify_2d(blks)


if __name__ == '__main__':

    font_pathname = "typefaces/liberation-mono/LiberationMono-Bold.ttf"
    text = "".join(map(chr, range(ord('a'), ord('z')))) + \
           "".join(map(chr, range(ord('A'), ord('Z')))) + \
           "".join(map(chr, range(ord('0'), ord('9')))) + "?,:{}-=_+.;|[]<>()/'!@#$%^&*`" + '"' "\\" + " "

    img_name = "snapchatgold"
    img_path = "examples"

    gaussian_smoothing_sigma = 48
    font_size = 48

    parameters = '\n'.join([f"{gaussian_smoothing_sigma=}", f"{font_size=}", f"{text=}"])
    with open(os.path.join(img_path, f"{img_name}.txt"), 'w') as f:
        f.write(parameters)

    img_pathname = os.path.join(img_path, f"{img_name}.png")
    img = iio.imread(img_pathname).astype(float)  # TODO: handle deprecation warning

    # TODO: include parameters in filename
    # TODO: write parameters onto image body

    img_lpf = gaussian(img, sigma=gaussian_smoothing_sigma, mode='nearest', preserve_range=True, truncate=4.0, channel_axis=2)
    img_gl = TexelEncoder.grayscale_and_remove_mean(img - img_lpf)

    encoder = TexelEncoder(font_pathname, font_size, text, img, img_lpf, img_gl)

    iio.imsave(os.path.join(img_path, f"{img_name}-backg.png"), encoder.bg_img, compression=0)
    iio.imsave(os.path.join(img_path, f"{img_name}-foreg.png"), encoder.fg_img, compression=0)
    iio.imsave(os.path.join(img_path, f"{img_name}-contr.png"), encoder.contour_img, compression=0)

    texels = encoder.encode()
    img_rec = encoder.deblockify(texels)

    iio.imsave(os.path.join(img_path, f"{img_name}-texels.png"), img_rec, compression=0)

