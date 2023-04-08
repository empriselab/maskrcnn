import os 
from pathlib import Path
import glob
import argparse
import concurrent.futures

TRAINING_SET_VERSION = '1'
BASE_DIR = Path(__file__).absolute().parents[1]
print("BASE DIRECTORY :: {}".format(BASE_DIR))
TRAINING_DATA_DIR = str(BASE_DIR / 'data' / 'training' / 'v{}'.format(TRAINING_SET_VERSION))
MASKRCNN_DIR = str(BASE_DIR / 'data' / 'training' / 'maskrcnn')
IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, 'images')
MASKS_DIR = os.path.join(TRAINING_DATA_DIR, 'masks')

def format_from_raw(image_filename):
    pass

def format_from_maskrcnn(image_filename):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bag", default=None)    
    parser.add_argument("--from_raw", default=False)
    args = parser.parse_args()

    if args.bag is None:
        print('Scanning dir **')
        examples = glob.glob('{}/**/*_img.png'.format(TRAINING_DATA_DIR), recursive=True)
    else:
        print('Scanning dir bag_{}'.format(args.bag))
        examples = glob.glob('{}/bag_{}/*_img.png'.format(TRAINING_DATA_DIR, args.bag), recursive=True)
    
    if not args.from_raw:
        fn = format_from_maskrcnn
    else:
        fn = format_from_raw

    max_processes = os.cpu_count()
    N = len(examples)
    print('Number of examples :: {}'.format(N))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as ex:
        n_processed = 0
        for result in ex.map(fn, examples):
            n_processed += 1

            if n_processed % 100 == 0:
                print('Progress: {}/{}'.format(n_processed, N), end='\r')
