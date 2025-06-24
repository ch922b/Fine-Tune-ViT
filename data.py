from datasets import load_dataset
from transformers import ViTImageProcessor
import torch

def get_dataset():
    return load_dataset('beans')

def get_processor(model_name_or_path='google/vit-base-patch16-224-in21k'):
    return ViTImageProcessor.from_pretrained(model_name_or_path)

def transform_fn(processor):
    def transform(example_batch):
        inputs = processor([x for x in example_batch['image']], return_tensors='pt')
        inputs['labels'] = example_batch['labels']
        return inputs
    return transform

def apply_transform(dataset, transform):
    return dataset.with_transform(transform)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
