import os
import glob
import random

import numpy as np
from torchmetrics import JaccardIndex, Dice
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import load_model

VALIDATION_PATHS = ['../../data/training/sam_plus_maskrcnn/bag_{}'.format(i) for i in range(1,4)]
CKPTS = ['best_train_model.pth', 'best_val_model.pth', 'archive/best_train_model.pth', 'archive/best_val_model.pth']
NUM_EVAL_EXAMPLES = 50
MAKSRCNN_CONFIDENCE_THRESH = 0.4
NUM_CLASSES = 34

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)


def get_random_exmaples() -> list:
    """
    Load a random set of examples in VALIDATION_PATHS
    """
    all_example_filenames = []
    for directory in VALIDATION_PATHS:
        all_example_filenames.extend(glob.glob(directory + '/*'))

    sampled_example_filenames = random.sample(all_example_filenames, NUM_EVAL_EXAMPLES)
    return sampled_example_filenames

def load_models() -> dict:
    """
    Load a selection of models for comparison
    """
    models = {}
    for ckpt in CKPTS:
        models[ckpt] = load_model(ckpt, use_cuda=False)
    return models

def box_xywh_to_xyxy(boxes):
    """
    Convert (N x 4) tensor from (x,y,w,h) to (x1,y1,x2,y2)
    """
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:,1] = boxes[:,0]
    boxes_xyxy[:,0] = boxes[:,1]
    boxes_xyxy[:,3] = boxes[:,0] + boxes[:,2]
    boxes_xyxy[:,2] = boxes[:,1] + boxes[:,3]
    return boxes_xyxy

def evaluate(models:dict, examples:list) -> dict:
    """
    Iterate through examples, predicting on them with each model. Then with those 
    predictions, record metrics 
    """
    scores = {m:{'map_scores':[], 'mar_scores':[]} for m in models}
    
    for example in tqdm(examples):
        npz_data = np.load(example)
        image = (torch.tensor(npz_data['image']) / 255.).permute(2,1,0)
        masks = torch.tensor(npz_data['masks'], dtype=torch.int64)
        labels = torch.tensor(npz_data['labels'], dtype=torch.int64)
        boxes = torch.tensor(npz_data['boxes'],dtype=torch.float32)

        for model_name, model in models.items():
            model.eval()
            with torch.no_grad():
                prediction = model([image])

            target = [{
                'boxes':box_xywh_to_xyxy(boxes),
                'masks':masks,
                'labels':labels,
            }]

            mp = MeanAveragePrecision()
            mp.update(prediction, target)
            results = mp.compute()

            scores[model_name]['map_scores'].append(results['map'])
            scores[model_name]['mar_scores'].append(results['mar_10'])

    return scores

def parse_results(scores:dict) -> None:
    """
    Parse the MAP and MAR scores for each model
    """
    for model_name in scores:
        print('-'*30)
        print('MODEL :: {}'.format(model_name))
        print('MEAN MAP :: {}'.format(np.mean(scores[model_name]['map_scores'])))
        print('MEAN MAR :: {}'.format(np.mean(scores[model_name]['mar_scores'])))

    print('-'*30)



def main():

    # load examples from a data path
    examples = get_random_exmaples()

    # load MaskRCNN models from checkpoints
    models = load_models()

    # evaluate models on examples
    scores = evaluate(models, examples)

    # printout
    parse_results(scores)
    












if __name__ == '__main__':
    main()