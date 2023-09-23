#!env python3
from __future__ import print_function
import argparse
import sys
import re
import textwrap

digito = [simb for simb in '0123456789']
letra = [simb for simb in 'abcdefghijklmnopqrstuvwxyz']
romano = [simb for simb in 'IVXLC']
simbolo = digito + letra + romano +[simb for simb in '_']

usage_msg='''Examples of usage:
Consider the following transducer (without weights), written in our format
    0 1 digito =
    1 3 simbolo =
    1 4 [^a-eh-z23] =
it will be converted into
    0 1 a a
    ...
    0 1 u u
    1 3 a a
    ...
    1 3 4 4
    1 4 f f
    1 4 g g
    1 4 4 4

Consider the following transducer (with weights), written in our format
    0 2 consoante = 1.0
    1 3 [abc] = 0.8
it will be converted into
    0 2 a a 1.0
    0 2 b b 1.0
    0 2 c c 1.0
    ...
    0 2 z z 1.0
    1 3 a a 0.8
    1 3 b b 0.8
    1 3 c c 0.8

The symbol "=" indicates that the output will be the same as the input'''

def print_line(f, t, i, o, w=None):
    if o == "=": o = i
    if w == None:
        print("%s\t%s\t%s\t%s" % (f, t, i, o))
    else:
        print("%s\t%s\t%s\t%s\t%s" % (f, t, i, o, w))


def expand(line):
    cols = line.strip().split()
    if len(cols) == 1:
        print("%s" % cols[0])

    elif 4 <= len(cols) <= 5:
        weight = None
        if len(cols) == 5: weight = cols[4]

        if cols[2] == "digito":
            for s in digito:
                print_line(cols[0], cols[1], s, cols[3], weight)
        elif cols[2] == "letra":
            for s in letra:
                print_line(cols[0], cols[1], s, cols[3], weight)
        elif cols[2] == "romano":
            for s in romano:
                print_line(cols[0], cols[1], s, cols[3], weight)
        elif cols[2] == "simbolo":
            for s in simbolo:
                print_line(cols[0], cols[1], s, cols[3], weight)
        elif cols[2].startswith("["):  # Regular expression
            for s in simbolo:
                if re.search(cols[2], s) != None:
                    print_line(cols[0], cols[1], s, cols[3], weight)
        else:
            print_line(cols[0], cols[1], cols[2], cols[3], weight)
    else:
        print("Error, incorrect number of colunms:", cols, file=sys.stderr)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Converts an FST written in our compact notation into a nice FST that can be used by openfst",
        epilog=textwrap.dedent(usage_msg))
    PARSER.add_argument('file', help='input file')
    args = PARSER.parse_args()

    with open(args.file) as f:
        while True:
            line = f.readline()
            if line == '': break
            if not re.match("^\s*$",line): expand(line)
