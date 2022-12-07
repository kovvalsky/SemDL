#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
Utility functions for reasoning demos
"""

from typing import List, Tuple, Dict
import re


#########################################################
# follows terminology at https://en.wikipedia.org/wiki/Syllogism
def gen_syllogism(M, S, P, neg="not", types="aeio", figures="1234", annotated=False):
    """
    Returns a generator that produces syllogisms. Each syllogism is a pair of
    its id (consisting of the figure and sentence types) and a tuple of three
    sentences (2 premises and a conclusion).
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
    entailment = set('f1-aaa f1-aii f1-eae f1-eio f2-aee f2-aoo f2-eae f2-eio f3-aii f3-eio f3-iai f3-oao f4-aee f4-eio f4-iai'.split())
    # contradictions can be deduced based on the negation of the conclusions
    neg_map = str.maketrans("aeio", "oiea")
    contradiction = set([ f"{e[:-1]}{e[-1].translate(neg_map)}" for e in entailmen ])
    
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
                    label = f"-{get_label(name)}" if annotated else "" 
                    yield f"{name}{label}", (prem1.format(**d), prem2.format(**d), con.format(**d))
