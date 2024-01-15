#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
Utility functions for reasoning demos
"""

from typing import List, Tuple, Dict
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


#########################################################
# Generate syllogisms
# follows terminology at https://en.wikipedia.org/wiki/Syllogism
def gen_syllogism(M, S, P, neg="not", types="aeio", figures="1234", existential_import=True):
    """
    Returns a generator that produces syllogisms. Each syllogism is a pair of
    its id (consisting of the figure and sentence types) and a tuple of three
    sentences (2 premises and a conclusion).
    Each syllogism also comes with an inference label that depends on the existential import flag
    """
    all_figures = {'1':('MP','SM'), '2':('PM','SM'), '3':('MP','MS'), '4':('PM','MS')}
    all_types = {'a': "All {} are {}",
                 'e': "No {} are {}",
                 'i': "Some {} are {}",
                 'o': f"Some {{}} are {neg} {{}}"
    }
    # filter the figures and types based on the input
    sel_figures = tuple( (f, v) for (f, v) in sorted(all_figures.items()) if f in figures )
    sel_types = tuple( (t, v) for (t, v) in sorted(all_types.items()) if t in types ) 

    # labeling the syllogisms
    if existential_import:
        entailment = set('f1-aaa f1-eae f1-aai f1-aii f1-eao f1-eio f2-aee f2-eae f2-aeo f2-aoo f2-eao f2-eio f3-aai f3-aii f3-iai f3-eao f3-eio f3-oao f4-aee f4-aai f4-iai f4-aeo f4-eao f4-eio'.split())
    else:
        entailment = set('f1-aaa f1-aii f1-eae f1-eio f2-aee f2-aoo f2-eae f2-eio f3-aii f3-eio f3-iai f3-oao f4-aee f4-eio f4-iai'.split())
    # contradictions can be deduced based on the negation of the conclusions
    neg_map = str.maketrans("aeio", "oiea")
    contradiction = set([ f"{e[:-1]}{e[-1].translate(neg_map)}" for e in entailment ])
    
    # Get label based on the name of a syllogism
    def get_label(name):
        if name in entailment: return 'entailment'
        if name in contradiction: return 'contradiction'
        return 'neutral'
    
    d = {'M':M, 'S':S, 'P':P}
    # generation by looping over figures
    for f, (p1, p2) in sel_figures:
        # make placeholders for f-string substitutions 
        p1 = [ f"{{{x}}}" for x in p1 ]
        p2 = [ f"{{{x}}}" for x in p2 ]
        p = [ '{S}', '{P}' ]
        # generate sentences by looping over sentence types
        for t, s in sel_types:
            for t1, s1 in sel_types:
                for t2, s2 in sel_types:
                    prem1 = s1.format(*p1)
                    prem2 = s2.format(*p2)
                    con = s.format(*p)
                    name = f"f{f}-{t1}{t2}{t}"
                    label = f"-{get_label(name)}" if  else "" 
                    yield f"{name}{label}", (prem1.format(**d), prem2.format(**d), con.format(**d))
                    
                    
#########################################################
# Classify NLI problems

def load_tok_model(hub_name):
    tokenizer = AutoTokenizer.from_pretrained(hub_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(hub_name)
    return tokenizer, model

def probs2prediction(probs, id2label):
    """ 
    Gets prob distribution and selects the max with its corresponding label.
    Returns dict of prediction details
    """
    lab_index = np.argmax(probs)
    return {"label_index": lab_index, "label": id2label[lab_index],
            "prob": probs[lab_index],
            "probs": {l:probs[i] for i, l in id2label.items()}}

def predict_nli(tokenizer, model, nli_prob, device=None):
    """ 
    nli_prob - list with two elements
    """
    encoded_prob = tokenizer(*nli_prob, truncation=True, padding=True, return_tensors="pt")
    encoded_prob = encoded_prob.to(device) if device else encoded_prob.to(model.device)
    output = model(**encoded_prob) #transformers.modeling_outputs.SequenceClassifierOutput
    probs = torch.softmax(output.logits, dim=1).tolist()[0]
    return probs2prediction(probs, model.config.id2label)
