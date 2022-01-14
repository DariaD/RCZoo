#!/usr/bin/env bash

datasetnames=(

#    amazonqa
#    amazonyesno
    arc
#    babi
#    boolq
#    #cbt
#    #clicr
#    #cnn
#    coqa
#    cosmosqa
#    #dailymail
#    dream
#    drop
#    duorc
#    emrqa
#    hotpotqa
#    mcscript
#    mcscript2.0
#    mctest160
#    mctest500
#    medhop
#    movieqa
#    #msmarco
#    multirc
#    narrativeqa
#    newsqa
#    pubmedqa
#    quac
#    quasars
#    quasart
#    race
#    racec
#    recipeqa
#    reclor
#    searchqa
#    sharc
#    squad
#    squad2.0
#    turkqa
#    tydi
#    wdw
#    wikihop
#    wikimovies

)

for i in "${datasetnames[@]}";
do
    #echo $i
	python3 run_datasets.py --task_name="$i" --debug_flag=True
done