#!/bin/bash

# Script by Salomon Kabongo, skabongo.github.io
source "/nfs/home/kabenamualus/anaconda3/bin/activate"


arxiv_tex_dir="arxiv_tex"
arxiv_xml_dir="arxiv_xml"
arxiv_tex_edit_dir="trash_tex_edit"

# rm -r "${arxiv_xml_dir}"

mkdir -p "${arxiv_tex_dir}" "${arxiv_xml_dir}" "${arxiv_tex_edit_dir}"

# MINSIZE is 500 bytes.
MINSIZE=500

# MAXSIZE is 500_000_000 = 500 MB
MAXSIZE=50000000

for FILE in ${arxiv_tex_dir}/*; do 
    echo "${FILE}"; 
    FILESIZE=$(stat -c%s "${FILE}")
    file=${FILE##*/}
    file_ID=${file%.*}
    
    echo "Processing file $arxiv_xml_dir/$file_ID.xml ..."
    echo "Size ${FILESIZE}"

    if [ -e "$arxiv_xml_dir/$file_ID.xml" ]
    then
        echo "File $arxiv_xml_dir/$file_ID.xml already exist"
    else
        # Checkpoint
        if (( FILESIZE < MINSIZE)); then
            rm "${FILE}"
        fi
        if (( FILESIZE > MAXSIZE)); then
            echo "File ${file_ID}.tex too big"
            continue
        fi

        if [ -e "$arxiv_tex_edit_dir/$file_ID.tex" ]
        then
            echo "File $arxiv_tex_edit_dir/$file_ID.tex already exist"
            # continue
        else
            python sample.py -pInputFile "$arxiv_tex_dir/${file_ID}.tex" -pOutput "$arxiv_tex_edit_dir/${file_ID}.tex"
        fi

        

        { # try

            # Kill the script if it runs for more than 10 minutes
            timeout 5m pandoc "$arxiv_tex_edit_dir/${file_ID}.tex" -f latex -t tei --template=default.tei -o "$arxiv_xml_dir/${file_ID}.xml"
            #save your output
            

        } || { # catch
            # save log for exception 
            echo "$arxiv_tex_edit_dir/${file_ID}.tex to .xml failed"
            
            rm "$arxiv_xml_dir/${file_ID}.xml"
            continue
        }

        
        echo "File $arxiv_xml_dir/$file_ID.xml completed"  

    fi
           
done