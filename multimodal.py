#!/usr/bin/env python3
# -*- coding: utf8 -*-

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from PIL import Image
import requests
import torch
import numpy as np

def load_clip(model_name):
  model = CLIPModel.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  processor = CLIPProcessor.from_pretrained(model_name)
  return model, tokenizer, processor

def embed_text(text, tokenizer, model):
  inputs = tokenizer(text, return_tensors="pt")
  text_features = model.get_text_features(**inputs).squeeze(dim=0)
  return text_features

def embed_image(image, processor, model):
  inputs = processor(images=image, return_tensors="pt")
  image_features = model.get_image_features(**inputs).squeeze(dim=0)
  width, height = image.size
  image = image.resize((int(width/(height/250)), 250))
  display(image)
  return image_features

def embed_local_image(path, processor, model):
  image = Image.open(path).convert("RGB")
  image_features = embed_image(image, processor, model)
  return image_features

def embed_image_from_url(url, processor, model):
  image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
  image_features = embed_image(image, processor, model)
  return image_features
  