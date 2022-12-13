#!/bin/bash

# Script by Salomon Kabongo, skabongo.github.io

# index_dir="index"
# papers_dir="papers"
arxiv_dir="arxiv_pdf"

# arxiv_dir="trash_tex"
# arxiv_dir="trash_1"

mkdir -p "${arxiv_dir}" 

# MINSIZE is 500 bytes.
MINSIZE=500

# MAXSIZE is 500_000_000 = 500 MB
MAXSIZE=50000000


python <<EOF
import pandas as pd
df = pd.read_csv('../annotations_nov042022/paper_links.tsv', sep="\t", names=["name", "link"])
df["link"] = df["link"].apply(lambda x: x.replace("abs", "pdf"))
df[["link"]].to_csv('../annotations_nov042022/final_link_paper.txt', header=False, index=False)
EOF


process_file () {
    file="$1"
    file_ID=${file##*/}
    echo "Processing file ${file_ID}"
    if [ -e "$arxiv_dir/$file_ID" ]
    then
        echo "File $arxiv_dir/$file_ID already exist"        
    else
        sleep 1.1
        # wget --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" "$file" -P temp/ 
        wget --user-agent="Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.12) Gecko/20101026 Firefox/3.6.12" "$file" -P temp/ 
        mv temp/"$file_ID" $arxiv_dir/"${file_ID}"
        
        # tar -xf temp/"${file_ID}.tar.gz" --directory=temp 
        
        # # Check all the file that end with .tex to get the main file
        # # this block of code assume that the biggest .tex file is the main file
        # biggerFile=0
        # final_file=""
        # for file in temp/*.tex ; do
        #     FILESIZE=$(stat -c%s $file)
        #     if (( FILESIZE > biggerFile)); then
        #         biggerFile=$FILESIZE
        #         final_file=$file
        #     else
        #         continue
        #     fi

        #     cp $final_file temp/main.tex
        # done
        
        # cd temp
        # echo "before perl"
        # FILESIZE=$(stat -c%s "main.tex")
        # echo "File size "$FILESIZE
        # if (( FILESIZE > MAXSIZE)); then
        #     echo "File ${file_ID}.tex too big"
        #     echo "File Size ${FILESIZE}"
        #     # continue

        # else
        #     perl ../latexpand main.tex > "${file_ID}.tex"  
        #     echo "after perl"
        #     cd ..
        #     mv temp/"${file_ID}.tex" "$arxiv_dir/"
        # fi

        

        # rm temp/* -r
    fi    
}

while read file
do
    process_file "${file}"
# done <../annotations_final/test_papers.txt
done <../annotations_nov042022/final_link_paper.txt