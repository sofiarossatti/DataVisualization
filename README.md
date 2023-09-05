# The gender-pay gap situation in Europe between 2010 and 2021 👩🏻👨🏻

** ABOUT THE DATASET 🔍**
The dataset analyzed is from Kaggle (https://www.kaggle.com/datasets/gianinamariapetrascu/gender-pay-gap-europe-2010-2021) and it has been used to conduct a gender pay gap analysis in the European countries. It has 23 columns of different data type: object, integer and float. Two of these columns, respectively “GDP” and “Urban_population” columns, have been delated since they were not relevant for my work.
The column “Year” contains integer variables, and they are numbers from 2010 to 2021, which are the years I consider in my analysis.
The column “Country” contains categorical variables which are all the European nations but Greece. Since the dataset does not provide information of the Greek gender pay gap, it is not part of the analysis.
The remaining variables are float numbers and indicate the percentage pay gap between women and men. The considered working fields are: Industry, Business, Mining, Manufacturing, Electricity Supply, Water Supply, Construction, Retail Trade, Transportation, Accommodation, Information, Financial, Real Estate, Professional Scientific, Administrative, Public Administration, Education, Human Health, Arts and other minor sectors that are unified under the name “Other”.

** PREVIOUS CLEANING 🧹 **
Before starting my analysis, I had to prepare the data. Since there were many null values, I firstly divided the dataset into smaller datasets, one for each European country. This step is fundamental to fill the Nan values with the mean values of the corresponding country’s variables.
Then, I concatenated all the smaller datasets into a bigger one, obtaining the final dataset.
After having cleaned my dataset, I proceeded with my analysis. 

** WHY THIS DATASET? 🤔**
The gender paygap remains a persistent issue, casting a shadow over efforts towards workplace equality and fair compensation. I chose this dataset because I wanted to shed light on this subject, providing some information about the ongoing problem. The decision to embark on this journey is driven by a resolute commitment to advocate for equitable workplaces and a shared vision of progress that leaves no room for gender-based discrepancies in earnings.
