#!/usr/bin/env bash

##### run compare.py 
a = $1
b = $2

echo "input-> $a"
echo "output-> $b"

echo "start text compare..."

python ali_wx_wiki_vec_compare.py $a $b

echo "text compare end!!"
