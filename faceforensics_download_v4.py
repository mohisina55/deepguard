#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import urllib.request
import tempfile
import time
import sys
import json
from tqdm import tqdm
from os.path import join, isfile


# Constants
FILELIST_URL = 'misc/filelist.json'
DEEPFEAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5']

DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

ALL_DATASETS = list(DATASETS.keys())
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']


def parse_args():
    parser = argparse.ArgumentParser(description='Download FaceForensics++ data')
    parser.add_argument('output_path', type=str, help='Output directory')
    parser.add_argument('-d', '--dataset', type=str, default='all',
                        choices=ALL_DATASETS + ['all'], help='Dataset to download')
    parser.add_argument('-c', '--compression', type=str, default='raw', choices=COMPRESSION)
    parser.add_argument('-t', '--type', type=str, default='videos', choices=TYPE)
    parser.add_argument('-n', '--num_videos', type=int, default=None, help='Limit number of videos')
    parser.add_argument('--server', type=str, default='EU', choices=SERVERS)
    parser.add_argument('--dry_run', action='store_true', help="List number of videos without downloading")
    return parser.parse_args()


def get_server_url(server):
    return {
        'EU': 'http://canis.vc.in.tum.de:8100/',
        'EU2': 'http://kaldir.vc.in.tum.de/faceforensics/',
        'CA': 'http://falas.cmpt.sfu.ca:8100/',
    }.get(server)


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rProgress: {percent}%, {progress_size // (1024*1024)} MB, {speed} KB/s, {int(duration)}s elapsed")
    sys.stdout.flush()


def download_file(url, out_file, report_progress=False):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if isfile(out_file):
        tqdm.write(f"[SKIP] File already exists: {out_file}")
        return
    tmp_fd, tmp_path = tempfile.mkstemp()
    os.close(tmp_fd)
    try:
        if report_progress:
            urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)
        else:
            urllib.request.urlretrieve(url, tmp_path)
        os.rename(tmp_path, out_file)
    except Exception as e:
        tqdm.write(f"[ERROR] Failed to download {url}: {e}")
        os.remove(tmp_path)


def download_files(filenames, base_url, output_path, report_progress=True):
    os.makedirs(output_path, exist_ok=True)
    for filename in tqdm(filenames, desc="Downloading files"):
        full_url = base_url + filename
        out_path = join(output_path, filename)
        download_file(full_url, out_path, report_progress=report_progress)


def get_filelist(dataset_path, args):
    if 'DeepFakeDetection' in dataset_path or 'actors' in dataset_path:
        resp = urllib.request.urlopen(args.base_url + DEEPFEAKES_DETECTION_URL)
        filepaths = json.loads(resp.read().decode("utf-8"))
        return filepaths['actors'] if 'actors' in dataset_path else filepaths['DeepFakesDetection']
    else:
        resp = urllib.request.urlopen(args.base_url + FILELIST_URL)
        file_pairs = json.loads(resp.read().decode("utf-8"))
        if 'original' in dataset_path:
            return [item for pair in file_pairs for item in pair]
        else:
            flist = []
            for pair in file_pairs:
                flist.append('_'.join(pair))
                if args.type != 'models':
                    flist.append('_'.join(pair[::-1]))
            return flist


def main(args):
    args.base_url = get_server_url(args.server) + 'v3/'
    args.deepfakes_model_url = args.base_url + 'manipulated_sequences/Deepfakes/models/'

    print(f"Terms of Service: {get_server_url(args.server)}webpage/FaceForensics_TOS.pdf")
    input("Press Enter to continue or Ctrl+C to abort...")

    datasets_to_download = [args.dataset] if args.dataset != 'all' else ALL_DATASETS

    for dataset in datasets_to_download:
        dataset_path = DATASETS[dataset]

        if 'original_youtube_videos' in dataset:
            print(f"Skipping {dataset} (downloaded as .zip from misc)")
            continue

        print(f"\nProcessing dataset: {dataset}")
        filelist = get_filelist(dataset_path, args)
        total = len(filelist)
        print(f"Found {total} items")

        if args.dry_run:
            continue

        if args.num_videos:
            filelist = filelist[:args.num_videos]
            print(f"Downloading first {args.num_videos} videos")

        if args.type == 'videos':
            filelist = [f + '.mp4' for f in filelist]
            dataset_url = f"{args.base_url}{dataset_path}/{args.compression}/videos/"
            output_path = join(args.output_path, dataset_path, args.compression, "videos")
        elif args.type == 'masks':
            if 'original' in dataset:
                print("Masks not available for original dataset. Skipping.")
                continue
            if 'FaceShifter' in dataset:
                print("Masks not available for FaceShifter. Skipping.")
                continue
            filelist = [f + '.mp4' for f in filelist]
            dataset_url = f"{args.base_url}{dataset_path}/masks/videos/"
            output_path = join(args.output_path, dataset_path, "masks", "videos")
        elif args.type == 'models':
            if dataset != 'Deepfakes':
                print("Only Deepfakes has models. Skipping.")
                continue
            for folder in tqdm(filelist, desc="Downloading Deepfakes models"):
                folder_url = f"{args.deepfakes_model_url}{folder}/"
                out_path = join(args.output_path, dataset_path, "models", folder)
                download_files(DEEPFAKES_MODEL_NAMES, folder_url, out_path)
            continue
        else:
            print(f"Unknown type {args.type}. Skipping.")
            continue

        print(f"Downloading {len(filelist)} files to {output_path}")
        download_files(filelist, dataset_url, output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
