#!env python3
import argparse
import csv
import re

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Converts a word into an FST")
    PARSER.add_argument("-s", dest="symbols", default="syms.txt", help="file containing the symbols")
    PARSER.add_argument('word', help='a word')
    args = PARSER.parse_args()

    if not args.symbols:
        # processes character by character
        for i,c in enumerate(args.word):
            print("%d %d %s %s" % (i, i+1, c, c) )
        print(i+1)
    else:
        with open(args.symbols) as f:
            symbols = [ row.split()[0] for row in f if row.split()[0] != "eps" ]
            symbols.sort(key = lambda s: len(s), reverse=True)
            exp = re.compile("|".join(symbols))

        word = args.word
        m = exp.match(word)
        i=0
        while ( len(word) > 0 ) and ( m is not None ):
            print("%d %d %s %s" % (i, i+1, m.group(), m.group()) )
            word = word[m.end():]
            m = exp.match(word)
            i += 1
        if len(word) > 0:
            print("unknown symbols in expression: ", word)
        else:
            print(i)
