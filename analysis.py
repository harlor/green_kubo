#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="""
                            Plot the viscosity
                            """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', dest='i', help="Input files", type=str)
    args = parser.parse_args()

    fs = args.i.split(',')
    for f_name in fs:
        f = np.load(f_name)
        eta = f['eta']

        t = eta[:, 0]
        vis = eta[:, 1]

        plt.semilogx(t, vis, label=f_name)


    plt.legend()
    plt.show()


main()