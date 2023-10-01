#!/bin/bash

mkdir -p compiled images

rm -f ./compiled/*.fst ./images/*.pdf

# ############ Compile source transducers ############
for i in sources/*.txt tests/*.txt; do
	echo "Compiling: $i"
    fstcompile --isymbols=syms.txt --osymbols=syms.txt $i | fstarcsort > compiled/$(basename $i ".txt").fst
done

# ############ CORE OF THE PROJECT  ############

#mix2numerical
fstconcat compiled/mmm2mm.fst compiled/aux.fst > compiled/mix2numerical.fst

#pt2en
fstconcat compiled/pt2en_aux.fst compiled/aux.fst > compiled/pt2en.fst

#en2pt
fstinvert compiled/pt2en.fst > compiled/en2pt.fst

# datenum2text
fstconcat compiled/month.fst compiled/aux_slash.fst |
fstconcat - compiled/day.fst |
fstconcat - compiled/aux_comma.fst |
fstconcat - compiled/year.fst > compiled/datenum2text.fst

# mix2text
fstcompose compiled/mix2numerical.fst compiled/datenum2text.fst > compiled/mix2text.fst

# date2text
fstunion compiled/mix2text.fst compiled/datenum2text.fst > compiled/date2text.fst


# ############ generate PDFs  ############
echo "Starting to generate PDFs"
for i in compiled/*.fst; do
	echo "Creating image: images/$(basename $i '.fst').pdf"
   fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
done
echo ""

fst2word() {
	awk '{if(NF>=3){printf("%s",$3)}}END{printf("\n")}'
}

#4 - tests the transducers with the students' 18th birthday
dateMembersMixPT=("DEZ/02/2018" "MAI/08/2019" "DEZ/2/2018" "MAI/8/2019")
dateMembersMixEN=("DEC/02/2018" "MAY/08/2019" "DEC/2/2018" "MAY/8/2019")
dateMembersNumerical=("12/02/2018" "05/08/2019" "12/2/2018" "5/8/2019")

#Testing with syms.txt as output
echo "Testing mix2numerical.fst:"
for w in "${dateMembersMixPT[@]}" "${dateMembersMixEN[@]}"; do
    res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
                        fstcompose - compiled/mix2numerical.fst | fstshortestpath | fstproject --project_type=output |
                        fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=syms.txt | fst2word)
    echo "$w = $res"
done
echo ""

echo "Testing en2pt.fst:"
for w in "${dateMembersMixEN[@]}"; do
    res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
                        fstcompose - compiled/en2pt.fst | fstshortestpath | fstproject --project_type=output |
                        fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=syms.txt | fst2word)
    echo "$w = $res"
done
echo ""

echo "Testing datenum2text.fst:"
for w in "${dateMembersNumerical[@]}"; do
    res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
                        fstcompose - compiled/datenum2text.fst | fstshortestpath | fstproject --project_type=output |
                        fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./scripts/syms-out.txt | fst2word)
    echo "$w = $res"
done
echo ""

echo "Testing mix2text.fst:"
for w in "${dateMembersMixPT[@]}" "${dateMembersMixEN[@]}"; do
    res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
                        fstcompose - compiled/mix2text.fst | fstshortestpath | fstproject --project_type=output |
                        fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./scripts/syms-out.txt | fst2word)
    echo "$w = $res"
done
echo ""

echo "Testing date2text.fst:"
for w in "${dateMembersMixPT[@]}" "${dateMembersMixEN[@]}" "${dateMembersNumerical[@]}"; do
    res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
                        fstcompose - compiled/date2text.fst | fstshortestpath | fstproject --project_type=output |
                        fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./scripts/syms-out.txt | fst2word)
    echo "$w = $res"
done
echo ""

#1 - generates files
echo "Generating files"
for w in compiled/t-*.fst; do
    fstcompose $w compiled/date2text.fst | fstshortestpath | fstproject --project_type=output |
                  fstrmepsilon | fsttopsort > compiled/$(basename $w ".fst")-out.fst
done
for i in compiled/t-*-out.fst; do
    fstdraw --portrait --isymbols=syms.txt --osymbols=./scripts/syms-out.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
done
echo "Done!"
