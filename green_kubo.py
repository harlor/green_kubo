#!/usr/bin/env python

import argparse
import numpy as np
from scipy.integrate import cumtrapz
import pandas as pd

from parser.xvg_parser import extract_dataframe


def c_fft(a):
    n = len(a)
    s = np.square(np.absolute(np.fft.fft(a, 2*n)))
    s2 = np.fft.ifft(s)[0:n]

    c = s2 / np.arange(n, 0, -1)
    return np.real(c)


def main():
    parser = argparse.ArgumentParser(description="""
                        Calculate the viscosity by using the Green-Kubu formula
                        """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', dest='i', help="Input file", type=str)
    parser.add_argument('-o', '--output', dest='o', help="Output file", type=str)
    parser.add_argument('-c', '--components', dest='c', help="Componenets", type=str, default='Pres-XY,Pres-XZ,Pres-YZ')
    parser.add_argument('-l', '--seglength', dest='l', help="Length of segments in ps", type=float, default=1000.0)
    parser.add_argument('-V', '--volume', dest='v', help="Volume in nm^3", type=float, default=1.0)
    parser.add_argument('-t', '--temperature', dest='t', help="Temperature in K", type=float, default=300.0)
    args = parser.parse_args()

    # Load the stress tensor
    print('Reading %s' % args.i)

    components = args.c.split(',')

    # initialize dt
    dt = None

    seglen = -1
    sum_acf = np.array([])

    segment = 0
    n = 0
    with open(args.i, 'r') as f:
        i = 0
        names = []
        rows = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            if "label" in line and "xaxis" in line:
                xaxis = line.split('"')[-2]

            if line.startswith("@ s") and "subtitle" not in line:
                name = line.split("legend ")[-1].replace('"', '').strip()
                names.append(name)

            # should catch non-numeric lines so we don't proceed in parsing
            # here
            if line.startswith(('#', '@')):
                continue

            # parse line as floats
            row = map(float, line.split())
            rows.append(row)

            # Init segment
            if i == 1 and n == 0:
                dt = list(rows[1])[0] - list(rows[0])[0]
                print('dt = %f' % dt)
                seglen = int(args.l / dt)

                # Seglen threshold
                if seglen < 2:
                    seglen = 2

                print('seglen = %d' % seglen)

                sum_acf = np.zeros(seglen - 1)

            if i == seglen - 1:
                i = -1
                segment += 1

                cols = [xaxis]
                cols.extend(names)

                p_ii = pd.DataFrame(rows, columns=cols)
                rows = []

                for c in components:
                    print('%s: %d' % (c, segment))
                    p = p_ii[c]
                    n += 1
                    sum_acf += c_fft(p)[1:]
            i += 1

    acf = sum_acf / float(n)
    t_acf = np.array(range(len(acf))) * dt


    # Prefactor Bar**2 * nm**3 / kB * ps * 1000
    visc_pref = (10 ** 5)**2 * (10 ** -9) ** 3 / 1.381e-23 * 10 ** -12 * 1000

    eta = cumtrapz(acf, dx=dt) * args.v / args.t * visc_pref
    np.savez(args.o, eta=np.transpose(np.array([t_acf[1:], eta])))

main()