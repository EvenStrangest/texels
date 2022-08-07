import numpy as np
from skimage import io as imgio
from skimage import color


def blockify_2d(im, bw, bh):
    x_count, y_count = im.shape[0] // bw, im.shape[1] // bh
    blcks = np.split(im, x_count, axis=0)
    blcks = list(map(lambda blk: np.split(blk, y_count, axis=1), blcks))
    return blcks


def deblockify_2d(blcks):
    im = list(map(lambda blk: np.concatenate(blk, axis=1), blcks))
    im = np.concatenate(im, axis=0)
    return im


def crop_for_blocking(im, bw, bh):
    new_w, new_h = (im.shape[0] // bw) * bw, (img.shape[1] // bh) * bh
    im = im[:new_w, :new_h, :]  # TODO: convert to center crop
    return im


img_pathname = "examples/tinylavi.png"
block_w, block_h = 100, 200

img = imgio.imread(img_pathname).astype(float)

img = np.expand_dims(color.rgb2gray(img), axis=-1)

img = crop_for_blocking(img, block_w, block_h)

blocks = blockify_2d(img, block_w, block_h)
img_rec = deblockify_2d(blocks)
assert np.allclose(img, img_rec)
del img_rec



moshe = 1

