#!/bin/bash
export MACA_PATH=/opt/maca/
w=${1:-5};t=${2:-1000};e=${3:-0}
g=$(mx-smi -L 2>/dev/null|grep -cE '^GPU#[0-9]+'||echo 8)
for((i=0;i<g;i++));do exec 9>/tmp/mg${i}.lock;if flock -n 9;then export MACA_VISIBLE_DEVICES=$i;break;fi;exec 9>&-;done
make clean;make test_maca
./test_maca $w $t $e
