import sys
import os
import numpy as np
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table, vstack


def list_files(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):
            for file in files:
                r.append(os.path.join(subdir, file))

    return r


def list_folders(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):
            r.append(subdir)

    return r


# In []

def make_catalog(save_path, refcats, target_ra=None, target_dec=None):
    catalogs = [file for file in list_files(refcats) if "refsexcat.fits" in file]

    for catalog in catalogs:
        tmp = catalog.replace("refsexcat.fits", "refimg.fits")
        if os.path.exists(tmp):
            hdul = fits.open(tmp)
            for i, line in enumerate(hdul[0].header):
                if 'INFOBITS' in line:
                    infobits_ref = hdul[0].header[i]
                if 'MAGZPRMS' in line:
                    magzprms_ref = hdul[0].header[i]
                if 'MAGZP' in line:
                    magzp_ref = hdul[0].header[i]

            hdul.close()
        else:
            infobits_ref = np.uint32 = np.uint32.max
            magzp_ref = np.NAN
            magzprms_ref = np.NAN

        catname = catalog[catalog.rfind("ztf_")+4:catalog.rfind("_refsexcat")]
        hdul = fits.open(catalog)
        all_sources = Table(hdul[1].data)
        hdul.close()
        tmp = all_sources[all_sources['FLAGS'] == 0]

        # ── Force-inject target if it was removed by the FLAGS cut ────────────
        if target_ra is not None and target_dec is not None:
            cos_dec = np.cos(np.radians(target_dec))
            sep_arcsec = np.sqrt(
                ((all_sources['ALPHAWIN_J2000'] - target_ra) * cos_dec * 3600) ** 2 +
                ((all_sources['DELTAWIN_J2000'] - target_dec) * 3600) ** 2
            )
            nearest = int(np.argmin(sep_arcsec))
            if sep_arcsec[nearest] < 3.0:
                tgt_flags = int(all_sources['FLAGS'][nearest])
                if tgt_flags != 0:
                    tmp = vstack([tmp, all_sources[nearest:nearest + 1]])
                    print(f"  [catalog] Target injected: "
                          f"sep={sep_arcsec[nearest]:.2f}\"  FLAGS={tgt_flags}")
        tmp = tmp[tmp['MAG_BEST'] <-5.6]
        table = Table()
        table = tmp['CLASS_STAR', 'ALPHAWIN_J2000',
                    'DELTAWIN_J2000', 'MAG_AUTO', 'MAGERR_AUTO']
        table['FLAG_SE_REF'] = tmp['FLAGS']
        table['MAG_APER_3px'] = tmp['MAG_APER'][:, 1]
        table['MAG_APER_4px'] = tmp['MAG_APER'][:, 2]
        table['MAG_APER_6px'] = tmp['MAG_APER'][:, 3]
        table['MAG_APER_10px'] = tmp['MAG_APER'][:, 4]
        table['MAGERR_APER_3px'] = tmp['MAGERR_APER'][:, 1]
        table['MAGERR_APER_4px'] = tmp['MAGERR_APER'][:, 2]
        table['MAGERR_APER_6px'] = tmp['MAGERR_APER'][:, 3]
        table['MAGERR_APER_10px'] = tmp['MAGERR_APER'][:, 4]
        table['MAGZP_REF'] = magzp_ref,
        table['MAGZPRMS_REF'] = magzprms_ref,
        table['INFOBITS_REF'] = infobits_ref
        table['ID'] = tmp['CLASS_STAR'].astype(str)

        del tmp
        table.sort('MAG_APER_3px')
        for i in range(len(table)):
            table['ID'][i] = str(i) + "_" + catname

        catname = save_path + '/' + catname + '(REFERENCE)[OBJECTS].csv'
        ascii.write(table, catname, format='csv', overwrite=True)

def main():
    # Makes a catalog to use with other scripts
    # arguments are given as aditional arguments when calling the script
    # first argument is the save path location
    #second argument is the path to the reference catalogs downloaded from ZTF

    # Example
    # save_path = 'D:/Documents/Work/UV/Downloads/Objects/Divided'
    # refcats = 'D:/Documents/Work/UV/Downloads/Img/Reference'

    save_path = sys.argv[1]
    refcats = sys.argv[2]

    make_catalog(save_path, refcats)


if __name__ == '__main__':
    main()
