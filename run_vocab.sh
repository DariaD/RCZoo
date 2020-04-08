datasetnames=(
    quac
    multirc
    coqa
    cosmosqa
    movieqa
    mcscript
    mcscript2.0
    amazonyesno
    mctest160
    mctest500
    lambada
)

for i in "${datasetnames[@]}";
do
    #echo $i
	python3 vocab_run.py --task_name="$i"
done