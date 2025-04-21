# cap5771sp25-project
# Project Title


## Project Demo

Check out the project demo using the link below:

[Watch the Demo](https://uflorida-my.sharepoint.com/:v:/g/personal/chiplun_rushangs_ufl_edu/Ede6izdyrLVOtWypTeBKNgQByTsL4_nEojaHMs8i_G8ycw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=FslpTT)

## Project Presentation

Access the project presentation using the link below:

[View the Presentation](https://uflorida-my.sharepoint.com/:p:/g/personal/chiplun_rushangs_ufl_edu/EaoRWCU-iwlKklUQQGABobkBqGhlncNgRA_kYs0fGA45LQ?e=muH4qe)


Intro to Data Science - CAP5771SP25 Project

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)

## Introduction
The primary goal of this project is to develop a recommendation tool that aggregates and analyzes diverse datasets from Zomato, Swiggy, and Indian Restaurants to recommend restaurants in Bangalore. Currently, there is no solution that consolidates reviews and review counts from these platforms, which is a real-world problem for users seeking comprehensive insights. For this milestone, I am focusing on collecting and preprocessing data as well as performing exploratory data analysis. The long-term aim is to build an interactive tool that collects user input on preferred cuisines, addresses, costs, and localities, and recommends restaurants based on their interests.

## Requirements
List all the software and libraries required to run the project.
- Python 3.10
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation
Instructions to set up the environment and install the necessary dependencies.
```bash
# Clone the repository
git clone https://github.com/rushang01/cap5771sp25-project.git

# Navigate to the project directory
cd cap5771sp25-project

# Install the required libraries
pip install -r requirements.txt
```

## Usage
Step-by-step instructions to run the project and reproduce the findings:

Before running the notebooks, download the datasets from the provided links in the [Datasets](#datasets) section and place them into a folder named `data` within the project directory. Merged data csv has already been provided in the Data folder, so you can skip to notebooks 2 and 3 if you do not want to do those steps.
```bash
# Run the data extraction notebook
1_data_extraction.ipynb

# Run the data processing notebook
2_data_processing.ipynb

# Run the data exploration notebook
3_data_exploration.ipynb
```

## Datasets
Links to the datasets used in the project:
- [Indian Restaurants](https://www.kaggle.com/datasets/arnabchaki/indian-restaurants-2023/data)
- [Zomato](https://www.kaggle.com/datasets/absin7/zomato-bangalore-dataset/data?select=zomato.csv)
- [Swiggy](https://www.kaggle.com/datasets/ashishjangra27/swiggy-restaurants-dataset/data?select=swiggy.csv)

## Results
Summary of the results and findings of the data exploration process are present in the data exploration notebook.