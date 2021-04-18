#!/bin/bash

archive_name="glove.6B.zip"
root_dir="GloVe"

if [ "$#" -gt 1 ]; then
	echo "$# parameters given. Only 1 expected. Use -h to view command format"
	exit 1
elif [ "$#" -gt 1 ]; then
	archive_name=$1
fi

if [ "$1" == "-h" ]; then
	echo "Usage: `basename $0` [archive name to download from nlp.stanford.edu]"
	echo "Default archive: $archive_name"
	exit 1
fi

mkdir -p $root_dir
wget http://nlp.stanford.edu/data/$archive_name -P $root_dir
unzip $archive_name -d $root_dir
rm $archive_name