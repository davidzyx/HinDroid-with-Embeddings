import os
from glob import glob
import random

import src.utils as utils
import src.data.sampling as sampling
import src.data.decompile as apktool


class IngestionException(Exception):
    pass


class IngestionPipeline():
    def __init__(self, n, out_dir, nproc):
        self.n_remaining = n
        self.out_dir = out_dir
        self.nproc = nproc
        self.final_dirs = []
        self.run()

    @classmethod
    def from_category(cls, category, n, out_dir, nproc):
        cls.url_iter = cls.init_sample_iter(cls, category)
        return cls(n, out_dir, nproc)

    @classmethod
    def from_urls(cls, urls, out_dir, nproc):
        cls.url_iter = iter(urls)
        return cls(len(urls), out_dir, nproc)

    @classmethod
    def from_apks(cls, fp_iter, n, out_dir):
        cls.n_remaining = n
        cls.out_dir = out_dir
        cls.final_dirs = []
        while cls.n_remaining > 0:  # modified run() with only decompile
            cls.n_failed = 0
            apk_fps = [next(fp_iter) for _ in range(cls.n_remaining)]
            smali_dirs = cls.step_decompile_apks(cls, apk_fps)
            cls.final_dirs += smali_dirs
            cls.n_remaining = cls.n_failed
        return cls.final_dirs

    def init_sample_iter(self, category):
        sitemaps_by_cat = sampling.construct_categories()
        if category == 'random':
            return sampling.dynamic_random(sitemaps_by_cat)
        else:
            return sampling.dynamic_category(sitemaps_by_cat, category)

    def step_download_apks(self, url_iter):
        """Pipeline block: Download apks from url iterators to out directory"""
        try:
            jobs = [next(url_iter) for _ in range(self.n_remaining)]
        except StopIteration as e:
            raise IngestionException("Not enough invalid apks from urls")
        apk_fps = apktool.mt_download_apk(jobs, self.out_dir, self.nproc)
        self.n_failed += sum(1 for i in apk_fps if i is None)
        return [i for i in apk_fps if i is not None]

    def step_decompile_apks(self, apk_fps):
        smali_dirs = apktool.mt_decompile_apks(apk_fps, self.out_dir, 2)
        self.n_failed += sum(1 for i in smali_dirs if i is None)
        smali_dirs = [i for i in smali_dirs if i is not None]
        return smali_dirs

    def run(self):
        while self.n_remaining > 0:
            self.n_failed = 0
            apk_fps = self.step_download_apks(self.url_iter)
            smali_dirs = self.step_decompile_apks(apk_fps)
            self.final_dirs += smali_dirs
            self.n_remaining = self.n_failed


def stage_apkpure(cls_i_cfg, out_dir, nproc):
    sampling_cfg = cls_i_cfg['sampling']
    smali_dirs = []

    if sampling_cfg['method'] == 'random':
        target, n = 'random', sampling_cfg['n']
        ppl = IngestionPipeline.from_category(target, n, out_dir, nproc)
        smali_dirs = ppl.final_dirs
    elif sampling_cfg['method'] == 'category':
        for target, n in sampling_cfg['category_targets'].items():
            ppl = IngestionPipeline.from_category(target, n, out_dir, nproc)
            smali_dirs += ppl.final_dirs
            del ppl
            print(f'Done ingesting {n} apps from {target}')
    else:
        raise NotImplementedError
    return smali_dirs


def stage_url(cls_i_cfg, out_dir, nproc):
    target = cls_i_cfg['sampling']['url_targets']
    ppl = IngestionPipeline.from_urls(target, out_dir, nproc)
    smali_dirs = ppl.final_dirs
    return smali_dirs


def stage_apk(cls_i_cfg, out_dir, nproc):
    sampling_cfg = cls_i_cfg['sampling']
    external_dir = cls_i_cfg['external_dir']

    if cls_i_cfg['external_structure'] == 'flat':
        apk_fps = glob(os.path.join(external_dir, '*.apk'))
        if sampling_cfg['method'] == 'random':
            n = sampling_cfg['n']
            assert len(apk_fps) >= n
            fp_iter = iter(random.sample(apk_fps, len(apk_fps)))
            smali_dirs = IngestionPipeline.from_apks(fp_iter, n, out_dir)
        elif sampling_cfg['method'] == 'all':
            assert len(apk_fps) > 0
            print(f'Ingesting {len(apk_fps)} apks')
            smali_dirs = apktool.mt_decompile_apks(apk_fps, out_dir, 2)
            smali_dirs = [i for i in smali_dirs if i is not None]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return smali_dirs


def stage_smali(cls_i_cfg, out_dir, nproc):
    external_dir = cls_i_cfg['external_dir']
    sampling_cfg = cls_i_cfg['sampling']

    if cls_i_cfg['external_structure'] == 'flat':
        smali_dirs = glob(os.path.join(external_dir, '*/'))
        assert len(smali_dirs) > 0, "external_dir has no app"
        assert sampling_cfg['method'] == 'random'
        smali_dirs = random.sample(smali_dirs, sampling_cfg['n'])
    elif cls_i_cfg['external_structure'] == 'by_category_variety':
        smali_dirs = glob(os.path.join(external_dir, '*', '*', '*/'))
        if sampling_cfg['method'] == 'random':
            smali_dirs = random.sample(smali_dirs, sampling_cfg['n'])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    links = [os.path.join(out_dir, os.path.basename(os.path.dirname(src))) for src in smali_dirs]
    [os.symlink(src, link, target_is_directory=True) for src, link in zip(smali_dirs, links)]
    return smali_dirs


def run_pipeline(data_cfg, nproc):
    final_dirs = {}
    for cls_i, cls_i_cfg in data_cfg.items():

        print(f"Ingesting {cls_i}")
        out_dir = utils.RAW_CLASSES_DIRS[cls_i]

        if cls_i_cfg['stage'] == "apkpure":
            smali_dirs = stage_apkpure(cls_i_cfg, out_dir, nproc)
        elif cls_i_cfg['stage'] == "url":
            smali_dirs = stage_url(cls_i_cfg, out_dir, nproc)
        elif cls_i_cfg['stage'] == "apk":
            smali_dirs = stage_apk(cls_i_cfg, out_dir, nproc)
        elif cls_i_cfg['stage'] == "smali":
            smali_dirs = stage_smali(cls_i_cfg, out_dir, nproc)
        else:
            raise NotImplementedError

        final_dirs[cls_i] = smali_dirs

    return final_dirs


def get_data(**config):
    """Main function of data ingestion. Runs according to config"""
    # Set number of process, default to 2
    nproc = config['nproc'] if 'nproc' in config.keys() else 2

    classes = run_pipeline(config['data_classes'], nproc)
    print({k: len(v) for k, v in classes.items()})
