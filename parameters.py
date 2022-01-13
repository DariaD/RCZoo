# Data paths
# Update Jan 2022

folder = "/home/pinecone/Data/"
the_folder = "/home/pinecone/spinning-storage/pinecone/Data/RC_Zoo/"

#folder = "/home/pinecone/spinning-storage/pinecone/Data/RC_Zoo/"
#the_folder = "/home/pinecone/Data/"
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
    "boolq":        folder + "BoolQ/",
    "pubmedqa":     folder + "PubMedQA/",
    "squad":        folder + "SQuAD/v1.1/",
    "squad2.0":     folder + "SQuAD/v2.0/",
    "wikimovies":   folder + "WikiMovies/movieqa/",
    "narrativeqa":  folder + "narrativeqa-master/",
    "sharc":        folder + "sharc1-official/json/",
    "turkqa":       folder + "turkqa/",
    "hotpotqa":     folder + "HotPotQA",
    "medhop":       folder + "qangaroo_v1.1/medhop/",
    "wikihop":      folder + "qangaroo_v1.1/wikihop/",
    "recipeqa":     folder + "RecipeQA/",
    "quasars":      folder + "quasar/quasar-s/",
    "quasart":      folder + "quasar/quasar-t/",
    "amazonqa":     folder + "AmazonQA/",
    "duorc":        folder + "duorc-master/dataset/",
    "drop":         folder + "drop_dataset/",
    "clicr":        folder + "Clicr/",
    "wdw":          folder + "WhoDidWhat/wdw_script/keys",
    "cnn":          folder + "CNNDailyMail/cnn",
    "dailymail":    folder + "CNNDailyMail/dailymail",
    "msmarco":      folder + "MSMARCO/",
    "newsqa":       folder + "NewsQA/",
    "searchqa":     folder + "SearchQA/data_json/",
    "tydi":         folder + "TyDi/",


    "quac":         folder + "QuAC/",
    "coqa":         folder + "CoQA/",
    "multirc":      folder + "MultiRC",
    "cosmosqa":     folder + "cosmosqa-master/data/",
    "movieqa":      folder + "Movies/MovieQA/MovieQA_benchmark/",
    "mcscript":     folder + "MCScript/",
    "mcscript2.0":  folder + "MCScript/v2.0/",
    "amazonyesno":  folder + "AmazonYesNo",
    "babi":         folder + "bAbI/tasks_1-20_v1-2/en/",

    "mctest160":    folder + "MCTest/160",
    "mctest500":    folder + "MCTest/500",
    "emrqa":        folder + "emrQA/n2c2-community-annotations_2014-pampari-question-answering/dataset",

    "dream":    folder + "DREAM",
    "reclor":   folder + "ReClor",
    "cbt":      folder + "CBTest/data",
    "race":     folder + "RACE/RACE",
    "racec":    folder + "RACE-C/data",
    "triviaqa": folder + "TriviaQA",
    "sciq":     folder + "SciQ",
    "record":   folder + "ReCoRD",
    "tweetqa":  folder + "TweetQA/TweetQA_data",
    "quail":    folder + "QuAIL/quail-master/quail_v1.2/json/",

    "naturalquestions": folder + "NaturalQuestions/v1.0/v1.0/",
    "wikiqa":           folder + "WikiQACorpus/",
    "wikireading":      folder + "WikiReading/",
    "lambada":          folder + "LAMBADA",


    # The newest
    "arc":  folder + "ARC-V1-Feb2018-2",
    "bipar":  folder + "BiPaR/Monolingual/EN",
    "openbookqa":  folder + "OpenBookQA-V1-Sep2018/Data/Additional",
    "qasc":  folder + "QASC_Corpus",
    "subjqa": folder + "SubjQA/SubjQA",
    "iirc": folder + "iirc_train_dev",



}
