#!/bin/bash

mkdir -p compiled images

rm -f ./compiled/*.fst ./images/*.pdf

# ############ Compile source transducers ############
for i in sources/*.txt tests/*.txt; do
	echo "Compiling: $i"
    fstcompile --isymbols=syms.txt --osymbols=syms.txt $i | fstarcsort > compiled/$(basename $i ".txt").fst
done

# ############ CORE OF THE PROJECT  ############

fstconcat compiled/mmm2mm.fst compiled/aux.fst > compiled/mix2numerical.fst
fstconcat compiled/pt2en_aux.fst compiled/aux.fst > compiled/pt2en.fst
fstinvert compiled/pt2en.fst > compiled/en2pt.fst

# datenum2text
fstconcat compiled/month.fst compiled/date2enum_aux1.fst > compiled/date2enum_aux3.fst
fstconcat compiled/date2enum_aux3.fst compiled/day.fst > compiled/date2enum_aux4.fst
fstconcat compiled/date2enum_aux4.fst compiled/date2enum_aux2.fst > compiled/date2enum_aux5.fst
fstconcat compiled/date2enum_aux5.fst compiled/year.fst > compiled/datenum2text.fst








# ############ generate PDFs  ############
echo "Starting to generate PDFs"
for i in compiled/*.fst; do
	echo "Creating image: images/$(basename $i '.fst').pdf"
   fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
done

fst2word() {
	awk '{if(NF>=3){printf("%s",$3)}}END{printf("\n")}'
}

#4 - tests the transducers with the students' 18th birthday
dateMembers=("DEC/02/2018" "MAY/08/2019")

#Testing with syms.txt as output
FSTs=(mix2numerical.fst en2pt.fst)
for i in "${FSTs[@]}"; do
    echo "Testing $i:"
    for w in $dateMembers; do
        res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
                           fstcompose - compiled/$i | fstshortestpath | fstproject --project_type=output |
                           fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=syms.txt | fst2word)
        echo "$w = $res"
    done
done

#TODO Testing with ./scripts/syms-out.txt as output
FSTs=(day.fst month.fst)
for i in "${FSTs[@]}"; do
    echo "Testing $i:"
    for w in "1" "02" "03" "12"; do
        res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
                           fstcompose - compiled/$i | fstshortestpath | fstproject --project_type=output |
                           fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./scripts/syms-out.txt | fst2word)
        echo "$w = $res"
    done
done

FSTs=(datenum2text.fst)
for i in "${FSTs[@]}"; do
    echo "Testing $i:"
    for w in "09/15/2055"; do
        res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
                           fstcompose - compiled/$i | fstshortestpath | fstproject --project_type=output |
                           fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./scripts/syms-out.txt | fst2word)
        echo "$w = $res"
    done
done

#TODO just to test date2text when it is ready
: '
#1 - generates files
echo "\n***********************************************************"
echo "Testing 4 (the output is a transducer: fst and pdf)"
echo "***********************************************************"
for w in compiled/t-*.fst; do
    fstcompose $w compiled/date2text.fst | fstshortestpath | fstproject --project_type=output |
                  fstrmepsilon | fsttopsort > compiled/$(basename $i ".fst")-out.fst
done
for i in compiled/t-*-out.fst; do
	echo "Creating image: images/$(basename $i '.fst').pdf"
   fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
done

'