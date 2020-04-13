import random
import re
from collections import defaultdict

from src.data.preload import parse_main_xml, extract_apps


def df_random(apps):
    """Randomly sample apps from the dataframe.
    Returns an iterator yielding a url from apkpure.com

    :param apps: dataframe from metadata.parquet.
    :yields: url
    """
    history = []
    while True:
        sample = apps.sample(1).squeeze()
        app_index = sample.package
        if app_index in history:
            continue
        else:
            history.append(app_index)

        yield f"https://apkpure.com/{sample.name_slug}/{sample.package}"


def construct_categories():
    """Construct a dict of sitemaps by their app category"""
    sitemaps_ls = parse_main_xml('https://apkpure.com/sitemap.xml')
    sitemaps_by_category = defaultdict(list)
    for i in sitemaps_ls:
        category = re.findall('sitemaps\/([\w]+)', i)[0]
        sitemaps_by_category[category] += [i]

    return dict(sitemaps_by_category)


def dynamic_category(sitemaps_by_category, category):
    """Randomly sample apps from a specific category.
    Returns an iterator yielding a url from apkpure.com

    :param: sitemaps_by_category: dictionary of sitemaps by category
    :param: category: the category to sample from
    :yields: url
    """
    history = []
    while True:
        sampled_sitemap = random.sample(sitemaps_by_category[category], 1)[0]
        sampled_url = extract_apps(sampled_sitemap).sample(1).url.values.item()
        app_index = sampled_url.split('/')[-1]
        if app_index in history:
            continue
        else:
            history.append(app_index)

        yield sampled_url


def dynamic_random(sitemaps_by_category):
    """Randomly sample apps from any category.
    Returns an iterator yielding a url from apkpure.com

    :param: sitemaps_by_category: dictionary of sitemaps by category
    :yields: url
    """
    iters_dict = {
        category: dynamic_category(sitemaps_by_category, category)
        for category in sitemaps_by_category.keys()
    }
    while True:
        sampled_category = random.sample(sitemaps_by_category.keys(), 1)[0]
        yield next(iters_dict[sampled_category])
