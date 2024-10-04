# import os
# import json
# import shutil
# from tqdm import tqdm
# import tarfile
# import argparse
# from urllib.error import HTTPError
# import urllib.request


# def main(args):
#   input_data = []
#   with open(args.input_path) as f:
#     for line in f:
#       input_data.append(json.loads(line))

#   # Download all PMC articles
#   print('Downloading PMC articles')
#   for idx, sample in enumerate(tqdm(input_data)):
#     try:
#       urllib.request.urlretrieve(sample['pmc_tar_url'], os.path.join(args.pmc_output_path, os.path.basename(sample['pmc_tar_url'])))
#     except HTTPError as e:
#       print('Error downloading PMC article: {}'.format(sample['pmc_tar_url']))
#       continue


#   # Untar all PMC articles
#   print('Untarring PMC articles')
#   for sample in tqdm(input_data):
#     fname = os.path.join(args.pmc_output_path, os.path.basename(os.path.join(sample['pmc_tar_url'])))
#     tar = tarfile.open(fname, "r:gz")
#     tar.extractall(args.pmc_output_path)
#     tar.close()
    
#   # Copy to images directory
#   print('Copying images')
#   for sample in tqdm(input_data):
#     src = os.path.join(args.pmc_output_path, sample['image_file_path'])
#     dst = os.path.join(args.images_output_path, sample['pair_id']+'.jpg')
#     shutil.copyfile(src, dst)
      

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_path', type=str, default='data/llava_med_image_urls.jsonl')
#     parser.add_argument('--pmc_output_path', type=str, default='data/pmc_articles/')
#     parser.add_argument('--images_output_path', type=str, default='data/images/')
#     args = parser.parse_args()
#     main(args)
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from tqdm import tqdm
import tarfile
import shutil

def download_and_extract(url, pmc_output_path):
    try:
        # Download
        output_path = os.path.join(pmc_output_path, os.path.basename(url))
        urllib.request.urlretrieve(url, output_path)
        # Extract
        with tarfile.open(output_path, "r:gz") as tar:
            tar.extractall(pmc_output_path)
        return True
    except Exception as e:
        print(f'Error processing {url}: {e}')
        return False

def copy_image(source, destination):
    try:
        shutil.copyfile(source, destination)
        return True
    except Exception as e:
        print(f'Error copying from {source} to {destination}: {e}')
        return False

def main(args):
    input_data = []
    with open(args.input_path) as f:
        for line in f:
            input_data.append(json.loads(line))
            
    input_data = input_data[:300000]
    
    # Download and extract files using multithreading
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = []
        for sample in input_data:
            url = sample['pmc_tar_url']
            futures.append(executor.submit(download_and_extract, url, args.pmc_output_path))

        # Wait for all downloads and extractions to complete
        for future in tqdm(futures):
            future.result()

    # Copy images using multithreading
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = []
        for sample in input_data:
            src = os.path.join(args.pmc_output_path, sample['image_file_path'])
            dst = os.path.join(args.images_output_path, sample['pair_id']+'.jpg')
            futures.append(executor.submit(copy_image, src, dst))

        # Wait for all copies to complete
        for future in tqdm(futures):
            future.result()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/data2.jsonl')
    parser.add_argument('--pmc_output_path', type=str, default='data/pmc/')
    parser.add_argument('--images_output_path', type=str, default='data/images_2/')
    args = parser.parse_args()
    main(args)
