### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

- Beyond the Anaconda distribution of Python, there are several libraries needed to be installed: `sqlalchemy`, `nltk`, `sklearn`, `pickle`, `flask`, `joblib`
- The code should run with no issues using Python versions 3.*. Currently using Python `3.11.3`

## Project Motivation<a name="motivation"></a>

During disaster events like earthquake, volcano, floods, etc, there are several messages sent by people who are suffering from disasters and need emergency aid. This project is to create an app that use a machine learning pipeline to categorize these messages so that you can send the messages to an appropriate disaster relief agency.


## File Descriptions <a name="files"></a>

Here's the file structure of the project:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md



<!-- Due to file size limit in Github, Stack Overflow survey data couldn't be pushed to this repo. You can download the full set of data [here](https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip)

There are three notebooks you can find to address the above questions:
-  [Code_Learning_Method.ipynb](https://github.com/hongtranthianh/STACKOVERFLOW-INSIGHT-2022/blob/main/Code_Learning_Method.ipynb) - `Question 1`
- [Education_and_Career.ipynb](https://github.com/hongtranthianh/STACKOVERFLOW-INSIGHT-2022/blob/main/Education_and_Career.ipynb) - `Question 2`
- [Technology.ipynb](https://github.com/hongtranthianh/STACKOVERFLOW-INSIGHT-2022/blob/main/Technology.ipynb) - `Question 3`

There is also three `.py` files in that runs the necessary code across the notebooks.
- [Wrangling_functions.py](https://github.com/hongtranthianh/STACKOVERFLOW-INSIGHT-2022/blob/main/Wrangling_functions.py)
- [Plot_functions.py](https://github.com/hongtranthianh/STACKOVERFLOW-INSIGHT-2022/blob/main/Plot_functions.py)
- [Networkx_function.py](https://github.com/hongtranthianh/STACKOVERFLOW-INSIGHT-2022/blob/main/Networkx_function.py) -->

## How to run web app<a name="results"></a>

The main findings of the code can be found at the post available [here](https://github.com/hongtranthianh/hongtranthianh.github.io/blob/main/_posts/Stack-Overflow-insight-2022.md)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Data is directly taken from [StackOverflow](https://insights.stackoverflow.com/survey/) and licensed under the [ODbL license](https://opendatacommons.org/licenses/odbl/1-0/).

TLDR: Free to use the data

Feel free to use the code here as you would like.

