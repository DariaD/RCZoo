## Welcome to English Machine Reading Comprehension Datasets

<!-- You can use the [editor on GitHub](https://github.com/DariaD/RCZoo/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/DariaD/RCZoo/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
-->


## Citation

If yupi find an information on this page usefull or use any code from this repository, please cite our  paper [English Machine Reading Comprehension Datasets: A Survey](https://aclanthology.org/2021.emnlp-main.693/):

```
@inproceedings{dzendzik-etal-2021-english,
    title = "{E}nglish Machine Reading Comprehension Datasets: A Survey",
    author = "Dzendzik, Daria  and
      Foster, Jennifer  and
      Vogel, Carl",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.693",
    doi = "10.18653/v1/2021.emnlp-main.693",
    pages = "8784--8804",
    abstract = "This paper surveys 60 English Machine Reading Comprehension datasets, with a view to providing a convenient resource for other researchers interested in this problem. We categorize the datasets according to their question and answer form and compare them across various dimensions including size, vocabulary, data source, method of creation, human performance level, and first question word. Our analysis reveals that Wikipedia is by far the most common data source and that there is a relative lack of why, when, and where questions across datasets.",
}
```


## Get Started with code
Attention: 

This project contains the code to process a number of datasets but does does not contains the datasets itselfs. 
To proceede with the processing, you should download a dataset yourself and set up paths in parameters.py


### Step 1: Preparation
 Get ready your python enviroment, create a new one if you need (recommended).
 
 Download or get ready a dataset for processing.
 
### Step 2: Paths
 Set up your paths in the 'parameters.py' file.
 
 To do so you need to specify the following: 
 
 'folder' is your primary folder for storage the datasets
 
 Make sure you specify name and patho of the datasets 


### Step 3: Run
To process one dataset run:

 ```
 python3 run_datasets.py --task_name=[TASK_NAME] --debug_flag=[True/False] --light=[True/False]

 ```
 To process all datasets or subset of datasets you can use 'run.sh'. Make sure you listed the datasets in the file. 
     
