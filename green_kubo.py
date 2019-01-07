#!/usr/bin/env python

import argparse
import numpy as np
from scipy.integrate import cumtrapz
import pandas as pd
import tailer

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
    parser.add_argument('-c', '--components', dest='c', help="Independent non-diagonal components of the stress tensor", type=str, default='Pres-XY,Pres-XZ,Pres-YZ')
    parser.add_argument('-d', '--dcomponents', dest='d', help="Don't use the diagonal components of the stress", default=True, action='store_false')
    parser.add_argument('-l', '--seglength', dest='l', help="Length of segments in ps", type=float, default=1000.0)
    parser.add_argument('-V', '--volume', dest='v', help="Volume in nm^3", type=float, default=1.0)
    parser.add_argument('-t', '--temperature', dest='t', help="Temperature in K", type=float, default=300.0)
    args = parser.parse_args()

    # Set non diagonal components:
    if args.c == 'None':
        components = []
    else:
        components = args.c.split(',')

    # Set diagonal components
    dcomponents = ['Pres-XX', 'Pres-YY', 'Pres-ZZ']

    # initialize dt
    dt = None

    # Set seglen to -1 which means it has to be initialized
    seglen = -1

    # Set segment index to 0
    segment = 0

    # Set normalization to 0
    n = 0

    for ll in tailer.tail(open(args.i), 1):
        lt = float(list(filter(None, ll.split(' ')))[0])
        segments = int(lt / args.l)

    # Start reading from file
    print('Reading %s' % args.i)
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
            if i <= 1 and n == 0:
                if i == 1:
                    dt = float(line.split()[0]) - t0
                    print('dt = %f' % dt)
                    seglen = int(args.l / dt)

                    # Seglen threshold
                    if seglen < 2:
                        seglen = 2

                    print('seglen = %d' % seglen)

                    sum_acf = np.zeros(seglen - 1)
                # This means i == 0:
                else:
                    t0 = float(line.split()[0])

            if i == seglen - 1:
                i = -1
                segment += 1

                cols = [xaxis]
                cols.extend(names)

                p_ii = pd.DataFrame(rows, columns=cols)
                rows = []

                # Calculate auto correlation function for non diagonal components:
                for c in components:
                    print('%s: %d/%d' % (c, segment, segments))
                    p = p_ii[c]
                    n += 2
                    sum_acf += 2 * c_fft(p)[1:]
                # Calculate auto correlation function for diagonal components:
                if args.d:
                    # Calculate sum P_ii = P_xx + P_yy + P_zz
                    sp_ii = (p_ii[dcomponents[0]] + p_ii[dcomponents[1]] + p_ii[dcomponents[2]]) / 3.0
                    # Normalization += 4
                    n += 4
                    for c in dcomponents:
                        print('%s: %d/%d' % (c, segment, segments))
                        p = p_ii[c] - sp_ii
                        sum_acf += c_fft(p)[1:]
            i += 1

    acf = sum_acf / float(n)
    t_acf = np.array(range(len(acf))) * dt

    # Prefactor Bar**2 * nm**3 / kB * ps * 1000
    visc_pref = (10 ** 5)**2 * (10 ** -9) ** 3 / 1.381e-23 * 10 ** -12 * 1000

    eta = cumtrapz(acf, dx=dt) * args.v / args.t * visc_pref

    # Save eta:
    print('Write to %s' % args.o)
    np.savez(args.o, eta=np.transpose(np.array([t_acf[1:], eta])))

main()