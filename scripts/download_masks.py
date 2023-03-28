import requests
from PIL import Image
import json
from io import BytesIO
import numpy as np
from segments import utils    # pip install segments-ai
from tqdm import tqdm


# Load the JSON file
with open('../data/json/emprise-feeding-infra-ground-truth.json', 'r') as f:
    js = json.load(f)['dataset']
    samples = js['samples']
    categories = js['tasks']['ground-truth']['attributes']['categories']

# store a mapping of (category ID) -> (food item) in a separate JSON 
id_to_food = {x['id']:x['name'] for x in categories}
with open('../data/json/id_to_food_mapping.json', 'w') as f:
    json.dump(id_to_food, f)

# Iterate through the entries in the ground truth JSON file
N = len(samples)
for sample in tqdm(samples):
    uuid = sample["uuid"]
    name = sample["name"]
    ground_truth_attr = sample['labels']['ground-truth']['attributes']
    url = ground_truth_attr['segmentation_bitmap']['url']
    annotations = ground_truth_attr['annotations']

    # Download the PNG file from the URL and save in raw dir
    response = requests.get(url)    # TODO: determine if this works after trial expires
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img.save(f'../data/segmentations/raw/{name}')

    # get mapping from local id to global category id
    id_to_category_id = {x['id']:x['category_id'] for x in annotations}
    id_to_category_id[0] = 0    # map 0 -> 0 since it isn't in the .json

    # get transform raw segmentation bitmap into labeling arr with shape (w, h) where value is the id 
    id_array = utils.load_label_bitmap_from_url(url)
    category_id_array = np.vectorize(id_to_category_id.get)(id_array)

    # save into the segmentations/processed dir
    processed_img = Image.fromarray(category_id_array.astype(np.uint8), mode='L')
    processed_img.save(f'../data/segmentations/processed/{name}')
