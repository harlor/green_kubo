#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="""
                            Plot the viscosity
                            """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', dest='i', help="Input files", type=str)
    parser.add_argument('-t', '--titles', dest='t', help="Titles", type=str)
    args = parser.parse_args()

    fs = args.i.split(',')
    ts = args.t.split(',')
    for f_name, title in zip(fs, ts):
        f = np.load(f_name)
        eta = f['eta']

        t = eta[:, 0]
        vis = eta[:, 1]

        plt.semilogx(t, vis, label=title)

    plt.legend(fontsize=20)
    plt.xlabel('$\\tau$ [ps]', size=20)
    plt.ylabel('$\eta$ [mPas]', size=20)
    plt.tight_layout()
    plt.show()


main()