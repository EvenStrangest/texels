from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio as iio
import os.path
from typing import NamedTuple, Union, Tuple, Optional

from texelify import TexelEncoder


class TexelifyDebugger(TexelEncoder):

    class DebugConfig(NamedTuple):
        title_font_pathname: str
        title_font_size: Optional[int] = None
        title_color: Optional[Union[str, Tuple[int, int, int]]] = None

        debug_notes: str = ""

    def __init__(self, im, config: TexelEncoder.Config, debug_config: DebugConfig):
        super().__init__(im, config)

        self.debug_conf = debug_config

        if self.debug_conf.title_font_size is None:
            self.debug_conf.title_font_size = self.fg_img.shape[0] // 10
        if self.debug_conf.title_color is None:
            self.debug_conf.title_color = "white"

        self.title_font = ImageFont.truetype(self.debug_conf.title_font_pathname, self.debug_conf.title_font_size)

        pass

    def place_title(self, im, title=None):
        if title is None:
            title = str(self) + "\n" + self.debug_conf.debug_notes  # TODO: do we want super() here?
        _im = Image.fromarray(im)
        ImageDraw.Draw(_im).text((0, 0), title, fill=self.debug_conf.title_color, font=self.title_font)
        return np.array(_im)


if __name__ == '__main__':

    img_name = "snapchatgold"
    img_path = "examples"

    img_pathname = os.path.join(img_path, f"{img_name}.png")
    im = iio.imread(img_pathname).astype(float)  # TODO: handle deprecation warning

    # TODO: include parameters in filename
    # TODO: write parameters onto image body

    text = "".join(map(chr, range(ord('a'), ord('z')))) + \
           "".join(map(chr, range(ord('A'), ord('Z')))) + \
           "".join(map(chr, range(ord('0'), ord('9')))) + "?,:{}-=_+.;|[]<>()/'!@#$%^&*`" + '"' "\\" + " "
    texel_config = TexelEncoder.Config(font_pathname="typefaces/liberation-mono/LiberationMono-Bold.ttf",
                                       font_size=48,
                                       chars=text,
                                       gaussian_smoothing_sigma=48)
    debugging_config = TexelifyDebugger.DebugConfig(title_font_pathname="typefaces/liberation-mono/LiberationMono-Regular.ttf")
    encoder = TexelifyDebugger(im, texel_config, debugging_config)

    with open(os.path.join(img_path, f"{img_name}.txt"), 'w') as f:
        f.write(str(encoder))

    iio.imsave(os.path.join(img_path, f"{img_name}-backg.png"), encoder.bg_img, compression=0)
    iio.imsave(os.path.join(img_path, f"{img_name}-foreg.png"), encoder.fg_img, compression=0)
    iio.imsave(os.path.join(img_path, f"{img_name}-contr.png"), encoder.contour_img, compression=0)

    texels = encoder.encode()
    img_rec = encoder.deblockify(texels)

    iio.imsave(os.path.join(img_path, f"{img_name}-texels.png"), img_rec, compression=0)

