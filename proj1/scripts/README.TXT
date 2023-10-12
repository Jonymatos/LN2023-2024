Incluem-se dois programas escritos em python que podem ser úteis para a elaboração do trabalho.


word2fst.py - dada uma palavra, permite gerar um transdutor correspondente a essa palavra. Exemplos:
./word2fst.py pedro
    0 1 p p
    1 2 e e
    2 3 d d
    3 4 r r
    4 5 o o
    5

./word2fst.py -s symbols.txt 2/set
    0 1 2 2
    1 2 / /
    2 3 set set
    3


compact2fst.py - permite gerar um transdutor a partir de outro, escrito com uma notação mais compacta.
Para designar um conjunto de símbolos pode usar as seguintes expressões:
    vogal, consoante, simbolo, ou expressões regulares do tipo [expr]

=================================
usage: compact2fst.py [-h] file

Converts an FST written in our compact notation into a nice FST that can be used by openfst

positional arguments:
  file        input file

optional arguments:
  -h, --help  show this help message and exit

Examples of usage:

Consider the following transducer (without weights), written in our format
    0 1 vogal =
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


Consider fthe following transducer (with weights), written in our format
    0 2 consoante = 1.0
    1 3 [abc] = 0.8

it will be converted into
    0 2 b b 1.0
    0 2 c c 1.0
    ...
    0 2 z z 1.0
    1 3 a a 0.8
    1 3 b b 0.8
    1 3 c c 0.8

The symbol "=" (mandatory for expressions) indicates that the output will be the same as the input
