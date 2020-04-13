import requests
import os
import subprocess
import shutil
from glob import glob
from pathos.threading import ThreadPool
from pathos.util import print_exc_info

# !pip install beautifulsoup4
from bs4 import BeautifulSoup


class DecompileException(Exception):
    pass


def prep_dir_apk(out_dir, apk_fn):
    package = os.path.basename(apk_fn)[:-4]  # TODO: not ending in .apk
    app_dir = os.path.join(out_dir, package)
    if not os.path.exists(app_dir):
        os.mkdir(app_dir)
    return app_dir, package


def get_apk(app_url):
    """Scrape the apk file from APKPure

    :param app_url: url to app on APKPure
    :returns: APK in bytes
    """
    package = app_url.split('/')[-1]
    print(f"Downloading apk {package}...")
    app_url += '/download?from=details'
    try:
        download_page = requests.get(app_url)
        assert download_page.status_code == 200
        soup = BeautifulSoup(download_page.content, features="lxml")
    except:
        raise DecompileException(f'Link for {app_url} is not valid')
    try:
        apk_url = soup.select('#iframe_download')[0].attrs['src']
    except:
        raise DecompileException(f"Err: Download link not found in {package}")
    return requests.get(apk_url).content


def save_apk(raw_dir, package, apk):
    """Save apk bytecode to filesystem

    :param raw_dir: path to class raw directory
    :param apk: APK package bytes
    :returns: filename of APK
    """
    apk_fn = os.path.join(raw_dir, f'{package}.apk')
    apk_fp = open(apk_fn, 'wb')
    apk_fp.write(apk)
    return apk_fn


def apktool_decompile(apk_fn, app_dir):
    """Decompile the apk package to its directory using apktool

    :param apk_fn: filename for the APK
    :param app_dir: path to app directory for output
    :raises: Exception is something bad happened
    """
    print(f'Decompiling {os.path.basename(apk_fn)}...')

    command = subprocess.run([
        'apktool', 'd',     # decode
        apk_fn,             # apk filename
        '-o', app_dir,      # out dir path
        '-f'                # overwrite out path
    ], capture_output=True)

    print(command.stdout.decode(), end="")
    if command.stderr != b'':
        print(command.stderr.decode())
        raise DecompileException('apktool error')


def clean(app_dir):
    """Clean unwanted files and other folders (resources) from app directory

    :param app_dir: path to app directory
    """
    # os.remove(apk_fn)
    unwanted_subdirs = (
        set(glob(os.path.join(app_dir, '*/'))) -
        set(glob(os.path.join(app_dir, 'smali*/')))
    )
    for dir in unwanted_subdirs:
        shutil.rmtree(os.path.abspath(dir))


def remove(app_dir, package):
    """Remove anything that is from this package"""
    shutil.rmtree(app_dir)
    apk_fn = app_dir + '.apk'
    if os.path.exists(apk_fn):
        os.remove(apk_fn)


def validity_check(app_dir):
    """Check if decompiled app directory has smali files"""
    smali_fn_ls = sorted(glob(
        os.path.join(app_dir, 'smali*/**/*.smali'), recursive=True
    ))
    if len(smali_fn_ls) == 0:
        raise DecompileException('App has no smali files')


def download_apk(apkpure_url, out_dir):
    # apk = get_apk(apkpure_url)
    # package = apkpure_url.split('/')[-1]
    # apk_fn = save_apk(out_dir, package, apk)
    # return apk_fn
    try:
        apk = get_apk(apkpure_url)
        package = apkpure_url.split('/')[-1]
        apk_fn = save_apk(out_dir, package, apk)
        return apk_fn
    except DecompileException as e:
        print(e)
        return None


def mt_download_apk(urls, out_dir, nproc):
    with ThreadPool(nproc) as p:
        apk_fns = p.map(download_apk, urls, [out_dir] * len(urls))
    return apk_fns


def decompile_apk_dir(apps_dir, out_dir=None):
    """depr"""
    apk_ls = glob(os.path.join(apps_dir, '*.apk'))
    assert all(apk.endswith('.apk') for apk in apk_ls)

    # count = 0
    app_dir_ls = []
    for apk_fn in apk_ls:
        try:
            app_dir, package = prep_dir_apk(apps_dir, apk_fn)
            apktool_decompile(apk_fn, app_dir)
            clean(app_dir)
            validity_check(app_dir)
            app_dir_ls.append(app_dir)
            print()  # empty line

        except DecompileException as e:
            print("Unexpected error:", e)
            # raise
            clean(app_dir, package)
            print()
            continue

    return app_dir_ls


def decompile_one_apk(apk_fp, out_dir):
    try:
        app_dir, package = prep_dir_apk(out_dir, apk_fp)
        apktool_decompile(apk_fp, app_dir)
        clean(app_dir)
        validity_check(app_dir)
        print()
        return app_dir
    except DecompileException as e:
        print(print_exc_info())
        print()
        remove(app_dir, package)
        return None


def decompile_apks(apk_fpaths, out_dir):
    apk_dirs = []
    for apk_fp in apk_fpaths:
        apk_dirs.append(decompile_one_apk(apk_fp, out_dir))
    return apk_dirs


def mt_decompile_apks(apk_fpaths, out_dir, nproc):
    with ThreadPool(nproc) as p:
        apk_dirs = p.map(decompile_one_apk, apk_fpaths, [out_dir] * len(apk_fpaths))
    # apk_dirs = [i for i in apk_dirs if i is not None]
    return apk_dirs
