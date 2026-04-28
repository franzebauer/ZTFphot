import sys
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np


def paint_psf(catalog, x, y, psf):
    X, Y = catalog.shape
    psf_x, psf_y = psf.shape
    psf_x, psf_xr = int(psf_x/2), psf_x % 2
    psf_y, psf_yr = int(psf_y/2), psf_y % 2

    for i in range(-psf_x, psf_x + psf_xr):
        for j in range(-psf_y, psf_y + psf_yr):
            if x + i >= 0 and x + i < X and y + j >= 0 and y + j < Y:
                # if not np.isnan(catalog[x+i, y+j]):
                catalog[x+i, y+j] += psf[psf_x + i, psf_y + j]

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def build_simulated_image(source_img, source_cat, save_name,
                          target_ra=None, target_dec=None):

    difimg = fits.open(source_img)
    catalog = fits.open(source_cat)

    fwhm = difimg[0].header["SEEING"]
    # fwhm = 1.5
    size = 25
    # size = (np.ceil(fwhm) // 2 * 4 + 1)

    _FIXED_FLUX = 1000  # all sources painted at the same flux; only positions matter

    wcs = WCS(difimg[0].header)
    catalog = Table(catalog[1].data)
    catalog = SkyCoord(ra=catalog['ALPHAWIN_J2000'], dec=catalog['DELTAWIN_J2000'], unit='deg')

    data = np.zeros(difimg[0].data.shape, dtype=np.float32)
    # preserve NaN mask from the diff image (bad pixels / edge regions)
    nan_mask = np.isnan(difimg[0].data)

    for object in catalog:
        x, y = wcs.world_to_pixel(object)
        x_, y_ = int(np.floor(x)), int(np.floor(y))
        psf = makeGaussian(size=size, fwhm=fwhm, center=(size//2 + (x - x_), size//2 + (y - y_))) * _FIXED_FLUX
        paint_psf(data, y_, x_, psf)

    if target_ra is not None and target_dec is not None:
        tgt = SkyCoord(ra=target_ra, dec=target_dec, unit='deg')
        if len(catalog) == 0 or tgt.separation(catalog).min().arcsec >= 3.0:
            x, y = wcs.world_to_pixel(tgt)
            x_, y_ = int(np.floor(x)), int(np.floor(y))
            nrows, ncols = data.shape
            if 0 <= x_ < ncols and 0 <= y_ < nrows:
                psf = makeGaussian(size=size, fwhm=fwhm,
                                   center=(size//2 + (x - x_), size//2 + (y - y_))) * _FIXED_FLUX
                paint_psf(data, y_, x_, psf)

    data = np.clip(data, 0, 32767).astype(np.int16)
    data[nan_mask] = 0  # replace NaNs with 0; int16 has no NaN
    difimg[0].data = data
    difimg.writeto(save_name, overwrite=True)


def main():
    # Builds a science image from a sexcat and a difference image downloaded from ztf
    # Arguments are given as aditional arguments when calling the script
    # First argument is the difimg downloaded from ZTF
    # Second argument is the Sexcat file downloaded from ZTF
    # Final argument is the save name, including path

    # Please create the required folders for the savepath.

    # Example
    # source_img = './scimrefdiffimg_1.fits'
    # source_cat = './sexcat_1.fits'
    # save_name = './sciimg_mocked_1.fits'

    source_img = sys.argv[1]
    source_cat = sys.argv[2]
    save_name = sys.argv[3]

    build_simulated_image(source_img=source_img, source_cat=source_cat, save_name=save_name)


if __name__ == '__main__':
    main()
