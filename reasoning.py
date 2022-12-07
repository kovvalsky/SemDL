#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
Utility functions for reasoning demos
"""

from typing import List, Tuple, Dict
import re


#########################################################
# follows terminology at https://en.wikipedia.org/wiki/Syllogism
def gen_syllogism(M, S, P, neg="not", types="aeio", figures="1234"):
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
                    yield f"f{f}-{t1}{t2}{t}", (prem1.format(**d), prem2.format(**d), con.format(**d))
