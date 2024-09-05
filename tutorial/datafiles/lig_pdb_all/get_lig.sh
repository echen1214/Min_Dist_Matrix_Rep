#!/bin/bash
# implemented temporary working directory so that google drive cloud storage wouldn't
# have to sync all of the files
WORK_DIR=`mktemp -d `
echo ${WORK_DIR}

if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

function cleanup {      
  rm -rf "$WORK_DIR"
  echo "Deleted temp working directory $WORK_DIR"
}

pwd=$(pwd)/
raw_dir="/Users/ericchen/Google Drive/My Drive/test_dist_analy/cdk_230706/datafiles/raw_pdb/"
cd ${WORK_DIR}
for path in "${raw_dir}"*.pdb 
#for path in "${raw_dir}"1H01.pdb 
do 
    pdb=$( basename "$path")
    fn="${pdb%.*}" 
    if ! ls "${pwd}"${fn}* 1> /dev/null 2>&1
    then
        echo here "${path}"
        pdb_selhetatm "${path}" | sed -e "/HOH/d" > ${fn}_tmp.pdb
        pdb_splitchain ${fn}_tmp.pdb
        rm ${fn}_tmp.pdb 
        for path1 in `ls *tmp*.pdb`
        do
            python "${pwd}"sep_by_resname.py ${path1}
            rm ${path1}
        done
	mv *.pdb "${pwd}"
    fi
done

#for path in `ls *tmp*.pdb`
#do
#    python sep_by_resname.py ${path}
#    rm ${path}
#done

trap cleanup EXIT
