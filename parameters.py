# File to set up dataset names and main paths.
# Update: Jan 2022
# Author: Daria D.


#TODO:
# Set up your path to the datasets and where put the results.
data_folder = "/home/pinecone/Data/"
result_folder = data_folder + "_RC_ZOO_RESULT"

save_data_path = "/home/pinecone/Data/RC_Zoo_ANALYSIS/"



common_NE_tags = ["PERSON", "ORG", "GPE", "LOC"]
additional_NE_tags = ["NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]
date_time_NE_tags = ["DATE", "TIME", "PERCENT", "MONEY"]
other_NE_tags = ["QUANTITY", "ORDINAL", "CARDINAL"]

dataset_names = {
    # 1
    "boolq": "BoolQ",
    # 2
    "pubmedqa": "PubMedQA",
    # 3
    "squad": "SQuAD",
    # 4
    "squad2.0": "SQuAD2.0",
    # 5
    "wikimovies": "WikiMovies",
    # 6
    "narrativeqa": "NarrativeQA",
    # 7
    "sharc": "ShaRC",
    # 8
    "turkqa": "TurkQA",
    # 9
    "hotpotqa": "HotPotQA",
    # 10
    "medhop": "Qangaroo MedHop",
    # 11
    "wikihop": "Qangaroo WikiHop",
    # 12
    "recipeqa": "RecipeQA",
    # 13
    "quasars": "Quasar-S",
    # 14
    "quasart": "Quasar-T",
    # 15
    "amazonqa": "AmazonQA",
    # 16
    "naturalquestions": "NaturalQuestions",
    # 17
    "duorc": "DuoRC",
    # 18
    "drop": "DROP",
    # 19
    "clicr": "CliCR",
    # 20
    "wdw": "WhoDidWhat",
    # 21
    "cnn": "CNN",
    "dailymail": "DailyMail",
    # 22

    # 23
    "msmarco": "MSMARCO",
    # 24
    "newsqa": "NewsQA",
    # 25
    "searchqa": "SearchQA",
    # 26
    "wikireading": "WikiReading",
    # 27
    "tydi": "TyDi",
    # 28
    "quac": "QuAC",
    # 29
    "coqa": "CoQA",
    # 30
    "multirc": "MultiRC",
    # 31
    "cosmosqa": "CosmosQA",
    # 32
    "movieqa": "MovieQA",
    # 33
    "mcscript": "MCScript",
    # 34
    "mcscript2.0": "MCScript2.0",
    # 35
    "amazonyesno": "AmazonYesNo",
    # 36
    "babi": "bAbI",
    # 37
    "wikiqa": "WikiQA",
    # 38
    "mctest160": "MCTest 160",
    "mctest500": "MCTest 500",
    # 39
    "emrqa": "emrQA",
    # 40
    "lambada": "LAMBADA",
    #-------
    #41
    "cbt": "CBTest",
    #42
    "booktest": "BookTest",
    #43
    "race": "RACE",
    #44
    "racec": "RACE-C",
    #45
    "dream": "DREAM",
    #46
    "reclor": "ReClor",
    #47
    "triviaqa": "TriviaQA",
    #48
    #""
    #49
    #""
    #50
    "r3": "R^3",
    #51
    "sciq": "SciQ",
    #52
    "record": "ReCoRD",
    #53
    "tweetqa": "TweetQA",
    #54
    "quail": "QuAIL",

    ######
    #55
    "arc": "ARC",
    # 56
    "openbookqa": "OpenBookQA",
    # 57
    "subjqa": "SubjQA",
    # 58
    "iirc": "IIRC",
    # 59
    "bipar": "BiPaR",
    # 60
    "qasc": "QASC"

}
datapaths = {
    "boolq": data_folder + "BoolQ/",
    "pubmedqa": data_folder + "PubMedQA/",
    "squad": data_folder + "SQuAD/v1.1/",
    "squad2.0": data_folder + "SQuAD/v2.0/",
    "wikimovies": data_folder + "WikiMovies/movieqa/",
    "narrativeqa": data_folder + "narrativeqa-master/",
    "sharc": data_folder + "sharc1-official/json/",
    "turkqa": data_folder + "turkqa/",
    "hotpotqa": data_folder + "HotPotQA",
    "medhop": data_folder + "qangaroo_v1.1/medhop/",
    "wikihop": data_folder + "qangaroo_v1.1/wikihop/",
    "recipeqa": data_folder + "RecipeQA/",
    "quasars": data_folder + "quasar/quasar-s/",
    "quasart": data_folder + "quasar/quasar-t/",
    "amazonqa": data_folder + "AmazonQA/",
    "duorc": data_folder + "duorc-master/dataset/",
    "drop": data_folder + "drop_dataset/",
    "clicr": data_folder + "Clicr/",
    "wdw": data_folder + "WhoDidWhat/wdw_script/keys",
    "cnn": data_folder + "CNNDailyMail/cnn",
    "dailymail": data_folder + "CNNDailyMail/dailymail",
    "msmarco": data_folder + "MSMARCO/",
    "newsqa": data_folder + "NewsQA/",
    "searchqa": data_folder + "SearchQA/data_json/",
    "tydi": data_folder + "TyDi/",


    "quac": data_folder + "QuAC/",
    "coqa": data_folder + "CoQA/",
    "multirc": data_folder + "MultiRC",
    "cosmosqa": data_folder + "cosmosqa-master/data/",
    "movieqa": data_folder + "Movies/MovieQA/MovieQA_benchmark/",
    "mcscript": data_folder + "MCScript/",
    "mcscript2.0": data_folder + "MCScript/v2.0/",
    "amazonyesno": data_folder + "AmazonYesNo",
    "babi": data_folder + "bAbI/tasks_1-20_v1-2/en/",

    "mctest160": data_folder + "MCTest/160",
    "mctest500": data_folder + "MCTest/500",
    "emrqa": data_folder + "emrQA/n2c2-community-annotations_2014-pampari-question-answering/dataset",

    "dream": data_folder + "DREAM",
    "reclor": data_folder + "ReClor",
    "cbt": data_folder + "CBTest/data",
    "race": data_folder + "RACE/RACE",
    "racec": data_folder + "RACE-C/data",
    "triviaqa": data_folder + "TriviaQA",
    "sciq": data_folder + "SciQ",
    "record": data_folder + "ReCoRD",
    "tweetqa": data_folder + "TweetQA/TweetQA_data",
    "quail": data_folder + "QuAIL/quail-master/quail_v1.2/json/",

    "naturalquestions": data_folder + "NaturalQuestions/v1.0/v1.0/",
    "wikiqa": data_folder + "WikiQACorpus/",
    "wikireading": data_folder + "WikiReading/",
    "lambada": data_folder + "LAMBADA",


    # The newest
    "arc": data_folder + "ARC-V1-Feb2018-2",
    "bipar": data_folder + "BiPaR/Monolingual/EN",
    "openbookqa": data_folder + "OpenBookQA-V1-Sep2018/Data/Additional",
    "qasc": data_folder + "QASC_Corpus",
    "subjqa": data_folder + "SubjQA/SubjQA",
    "iirc": data_folder + "iirc_train_dev",



}
