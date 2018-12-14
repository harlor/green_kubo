#!/bin/env python

import argparse
import numpy as np
from scipy.integrate import cumtrapz

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
    parser.add_argument('-V', '--volume', dest='v', help="Volume in nm³", type=float, default=1.0)
    parser.add_argument('-t', '--temperature', dest='t', help="Temperature in K", type=float, default=300.0)
    args = parser.parse_args()

    # Load the stress tensor
    print('Reading %s' % args.i)
    p_ii = extract_dataframe(args.i)

    components = args.c.split(',')
    t = p_ii['Time (ps)']

    dt = t.values[1] - t.values[0]
    print('dt = %f' % dt)

    segments = int(t.values[-1] / args.l)
    seglen = int(args.l / dt)
    print('Number of segments: %d' % segments)

    acf = np.zeros(seglen - 1)
    for c in components:
        p = p_ii[c]
        for i in range(segments):
            print('%s: %d/%d' % (c, i+1, segments))
            ps = p.values[i*seglen:((i+1)*seglen)]
            acf += c_fft(ps)[1:] / float(len(components)) / float(segments)
    t_acf = np.array(range(len(acf))) * dt


    # Prefactor Bar**2 * nm**3 / kB * ps * 1000
    visc_pref = (10 ** 5)**2 * (10 ** -9) ** 3 / 1.381e-23 * 10 ** -12 * 1000

    eta = cumtrapz(acf, dx=dt) * args.v / args.t * visc_pref
    np.savez(args.o, eta=np.transpose(np.array([t_acf[1:], eta])))

main()