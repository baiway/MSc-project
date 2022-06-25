#!/usr/bin/env python3

import sys
import numpy as np
import json

def evaluate_g1(F, L, a, D, d, E):
    I = np.pi * (D ** 4 - d ** 4) / 32.0
    return -F * ((a ** 2 * (L - a) ** 2) / (3.0 * E * L * I))

def evaluate_g2(F, L, a, D, d, E):
    b = L - a
    I = np.pi * (D ** 4 - d ** 4) / 32.0
    return -F * ((b * (L ** 2 - b ** 2)) / (6.0 * E * L * I))

def evaluate_g3(F, L, a, D, d, E):
    I = np.pi * (D ** 4 - d ** 4) / 32.0
    return F * ((a * (L ** 2 - a ** 2)) / (6.0 * E * L * I))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Usage: beam input_file")
    json_input = sys.argv[1]
    with open(json_input, 'r') as f:
        inputs = json.load(f)

    F = inputs['F']
    L = inputs['L']
    a = inputs['a']
    D = inputs['D']
    d = inputs['d']
    E = inputs['E']

    g1 = evaluate_g1(F, L, a, D, d, E)
    g2 = evaluate_g2(F, L, a, D, d, E)
    g3 = evaluate_g3(F, L, a, D, d, E)
    
    outfile = inputs['outfile']

    with open(outfile, 'w') as f:
        json.dump({'g1': g1, 'g2': g2, 'g3': g3}, f)