import pandas as pd
import requests
import gzip
from tqdm import tqdm
import io
import xml.etree.ElementTree as ET
import re
import os

from multiprocess import Pool


def xml_tree(url):
    """Parse a xml url request to a ElementTree root

    :param url: url string of the xml link, could be gzipped
    :returns: root -- root of the ElementTree
    """
    for _ in range(3):
        try:
            r = requests.get(url).content
            break
        except:
            pass
    else:
        print('Error getting', url)

    if url.endswith('.gz'):
        r = gzip.decompress(r)

    fp = io.StringIO(r.decode())
    root = ET.parse(fp).getroot()
    return root


def parse_main_xml(xml_url):
    """Parse the sitemap url from apkpure.com to get of list of apps

    :param xml_url: - string of url
    :returns: urls -- list of urls for the apps main page
    """
    root = xml_tree(xml_url)
    urls = [c[0].text for c in root]
    urls = ([
        u for u in urls
        if not ('default' in u or 'topics' in u or 'tag' in u or 'group' in u)
    ])
    return urls


def extract_apps(sitemap_url):
    """Extract a single sitemap.xml.gz and return a dataframe of apps

    >>> df = extract_apps('https://apkpure.com/sitemaps/beauty-2.xml.gz')
    >>> df.shape
    (1000, 4)

    :param sitemap_url: url in main sitemap.xml
    :returns: df -- pandas dataframe of all apps in the url
    """
    try:
        sitemap_root = xml_tree(sitemap_url)
    except XMLUrlInvalidError:
        return pd.DataFrame()

    apps = list(sitemap_root)
    apps = [app for app in apps if 'image' in app[4].tag]
    sitemap_category = re.search('\w+', sitemap_url.split('/')[-1]).group(0)

    df = pd.DataFrame({
        'url': [a[0].text for a in apps],
        'lastmod': [a[1].text for a in apps],
        'name': [a[4][1].text for a in apps],
    })
    df.lastmod = pd.to_datetime(df.lastmod)
    df['category'] = sitemap_category

    return df


def clean_and_process(metadata):
    """Clean inconsistencies and reduce complexity

    :param metadata: dataframe after aggregation
    :returns: metadata -- same dataframe reference
    """
    # remove this duplicated row
    metadata = metadata[~(
        (metadata['name'] == 'Vendetta Miami Police Simulator 2019') & 
        (metadata['category'] == 'comics')
    )]

    # extract more info in url
    url_info = metadata['url'].str.rsplit('/', n=2, expand=True) \
        .rename(columns=dict(zip(range(3), ['domain', 'name_slug', 'package'])))
    metadata = pd.concat([metadata, url_info], axis=1)

    # clean
    metadata = metadata.drop(columns=['domain', 'url'])
    metadata = metadata.reset_index(drop=True)
    metadata = metadata[['package', 'name', 'category', 'name_slug', 'lastmod']]
    return metadata


def run(data_fp, nproc):
    """Runs the first step of the data pipeline
    Gets the entire sitemap and stores it in data_fp

    :param data_fp: output filepath for metadata.parquet
    """
    sitemap_urls = parse_main_xml('https://apkpure.com/sitemap.xml')

    print('Getting app data...')
    with Pool(nproc) as p:
        df_list = list(tqdm(
            p.imap_unordered(extract_apps, sitemap_urls),
            total=len(sitemap_urls)
        ))
    metadata = pd.concat(df_list, ignore_index=True)  # aggregate

    metadata = clean_and_process(metadata)

    print(f'Saving to {data_fp} ...')
    metadata.to_parquet(data_fp, engine='pyarrow')


def load_data(data_dir, nproc, data_fp=None):
    """Check if parquet data exists. If not, proceed to download"""
    if not data_fp:
        data_fp = os.path.join(data_dir, 'metadata.parquet')
    if not os.path.exists(data_fp):
        print('Preload file does not exist. Downloading..')
        run(data_fp, nproc)

    print(f'Reading {data_fp}.. ')
    apps = pd.read_parquet(data_fp)
    print('done')
    return apps
