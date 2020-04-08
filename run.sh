datasetnames=(
    boolq
    #pubmedqa
    #squad
    #squad2.0
    wikimovies
    #narrativeqa
    #sharc
    turkqa
    hotpotqa
    medhop
    wikihop
    recipeqa
    quasars
    quasart
    amazonqa
    duorc
    drop
    clicr
    babi
    #wikiqa
)

for i in "${datasetnames[@]}";
do
    #echo $i
	python3 rc_zoo_run.py --task_name="$i"
done