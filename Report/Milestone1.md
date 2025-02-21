# Milestone 1 Report: Data Collection, Preprocessing, and Exploratory Data Analysis (EDA)
**Course**: CAP5771 – Spring 2025  
**Name**: Rushang Sunil Chiplunkar  
**Date**: February 21, 2025  

## 1. Objective of the Project
The primary goal of this project is to develop a recommendation tool that aggregates and analyzes diverse datasets from Zomato, Swiggy, and Indian Restaurants to recommend restaurants in Bangalore. Currently, there is no solution that consolidates reviews and review counts from these platforms, which is a real-world problem for users seeking comprehensive insights. For this milestone, I am focusing on collecting and preprocessing data as well as performing exploratory data analysis. The long-term aim is to build an interactive tool that collects user input on preferred cuisines, addresses, costs, and localities, and recommends restaurants based on their interests.

## 2. Type of Tool
**Planned Tool Options**:
- **Interactive Dashboard**: An interface to visualize trends, ratings, costs, cuisines, and other key metrics of restaurants in Bangalore using Streamlit.
- **Recommendation Engine**: A system that suggests personalized restaurant options based on user preferences and historical data, allowing users to input their preferences.

At this stage, my work lays the foundation for whichever final tool is chosen by integrating and analyzing data from multiple sources.

## 3. Data to Be Used
I am using three primary datasets containing restaurant information for Bangalore:
- **Swiggy Dataset**: Extracted from a JSON file (data/swiggy.json), this dataset provides details such as restaurant names, areas, ratings, rating counts, cost for two, addresses, cuisines, and menu items.
- **Zomato Dataset**: Provided in CSV format (data/zomato_bangalore.csv), this dataset includes restaurant name, address, location, cuisines, ratings, vote counts, and approximate cost for two.
- **Indian Restaurants Dataset**: This dataset (data/indian_restaurants.csv) offers similar details specific to Indian restaurants in Bangalore. It has been filtered to include relevant information such as restaurant name, location, locality, cuisines, ratings, votes, and cost.

Each dataset has been cleansed and standardized so that key features (e.g., name, address, rating, and cost for two) align for further analysis and integration.

## 4. Tech Stack
**Programming Language**: Python

**Libraries & Frameworks**:
- **Data Processing**:
  - ijson: For streaming JSON parsing as the file is almost 1 GB, making it harder to process.
  - Pandas: For data manipulation and cleaning.
  - NumPy: For numerical operations.
- **Data Visualization**:
  - Matplotlib and Seaborn: To generate charts, histograms, box plots, and other visualizations for the EDA.
- **Additional Tools**:
  - Jupyter Notebook / Python scripts: For developing and testing the data pipeline.
  - Git & GitHub: For version control and repository management.

Future milestones will also incorporate machine learning libraries such as Scikit-Learn for modeling.

## 5. Project Timeline & Future Tasks

### Milestone 1: Data Collection, Preprocessing, and Exploratory Data Analysis (EDA)
**Timeline**: February 5, 2025 – February 21, 2025 (2 weeks)  
**Tasks**:
- **Data Collection**:
  - Acquire datasets from approved sources (Swiggy, Zomato, and Indian Restaurants).
  - Verify dataset accessibility and document their properties (dimensions, attributes, and source URLs).
- **Data Preprocessing**:
  - Clean and preprocess data (e.g., handling missing values, converting data types, and addressing inconsistencies).
  - Standardize column names and formats across datasets.
- **Exploratory Data Analysis**:
  - Compute descriptive statistics (mean, median, standard deviation, etc.).
  - Identify patterns and anomalies using visualizations (histograms, scatter plots, box plots).
  - Merge datasets and remove duplicates to form a unified dataset.

### Future Milestones (Overview):

**Milestone 2**: Feature Engineering, Feature Selection, and Data Modeling  
**Timeline**: February 22, 2025 – March 28, 2025 (5 weeks)  
**Tasks**:
- **Feature Engineering**:
    - February 22, 2025 – March 1, 2025
    - Create new features from existing data to enhance model performance.
- **Feature Selection**:
    - March 2, 2025 – March 8, 2025
    - Identify and select key variables that significantly impact the model.
- **Data Modeling**:
    - March 9, 2025 – March 28, 2025
    - Build and train initial predictive models using selected features.

**Milestone 3**: Evaluation, Interpretation, Tool Development, and Presentation  
**Timeline**: March 29, 2025 – April 30, 2025 (5 weeks)  
**Tasks**:
- **Model Evaluation**:
    - March 29, 2025 – April 5, 2025
    - Evaluate model performance using appropriate metrics and validation techniques.
- **Interpretation**:
    - April 6, 2025 – April 12, 2025
    - Interpret the results and derive actionable insights from the models.
- **Tool Development**:
    - April 13, 2025 – April 23, 2025
    - Develop the final tool (dashboard/recommendation engine) based on the evaluated models.
- **Presentation**:
    - April 24, 2025 – April 30, 2025
    - Prepare and deliver the final presentation showcasing the project outcomes.

## 6. Exploratory Data Analysis (EDA) Report and Key Insights
### Summary of EDA
- **Data Integration**:
  After preprocessing and merging the datasets from Swiggy, Zomato, and Indian Restaurants, duplicate entries were removed (based on restaurant name and address), resulting in a consolidated dataset with unique restaurant records.

- **Data Quality**:
  Initial analyses revealed missing values in some fields (e.g., rating counts and costs), which were handled by removing incomplete records. Standardization of rating and cost columns was performed to ensure consistency.

### Key Insights and Visualizations
- **Ratings Distribution**:
  - **Observation**: The majority of restaurants have ratings clustered between 3.5 and 4.5, indicating generally positive customer experiences.
  - **Visualization**: A histogram was generated to show the frequency of restaurant ratings.

- **Cost Analysis**:
  - **Observation**: The box plot of 'cost for two' reveals a that most of the restaurants in the dataset have a cost for two between 125 INR and 500 INR.
  - **Visualization**: This plot illustrates the cost distribution and identifies outliers.

- **Cuisine Popularity**:
  - **Observation**: A bar chart summarizing ratings by cuisine type shows that dessert establishments are the among the highest rated in Bangalore (as expected).
  - **Visualization**: This chart helps in understanding the variation of ratings based on diversity of cuisine offerings.

- **Descriptive Statistics Table**:
  - **Observation**: Key metrics such as mean rating, median cost, and standard deviation are computed for numerical columns.