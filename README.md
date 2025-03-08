# Data Challenge: Predicting Employee Absenteeism

## Overview
This data challenge focuses on predicting employee absenteeism in hours based on various personal, work-related, and health-related factors. Participants will build models to analyze patterns of absenteeism and improve workforce management strategies.

## Objective
The goal of this challenge is to develop a predictive model that accurately estimates absenteeism time in hours (`Absenteeism_time_in_hours`). Participants should experiment with different modeling approaches, feature engineering techniques, and evaluation metrics to achieve the best performance.

## Dataset Description
The dataset contains records of employee absenteeism with 21 input features and 1 target variable. Each record represents an absence instance with multiple attributes related to personal demographics, work conditions, and health status.

### Features
1. **Individual ID** (`ID`): Unique identifier for each employee.
2. **Reason for absence** (`Reason_for_absence`): Categorized according to the International Code of Diseases (ICD) and additional reasons such as consultations, physiotherapy, and unjustified absence.
3. **Month of absence** (`Month_of_absence`): Month in which the absence occurred.
4. **Day of the week** (`Day_of_the_week`): Encoded as Monday (2) to Friday (6).
5. **Seasons** (`Seasons`): Encoded as Spring (1), Summer (2), Autumn (3), Winter (4).
6. **Transportation expense** (`Transportation_expense`): Employee's transportation cost.
7. **Distance from Residence to Work** (`Distance_from_Residence_to_Work`): Distance in kilometers.
8. **Service time** (`Service_time`): Number of years the employee has been with the company.
9. **Age** (`Age`): Employee’s age in years.
10. **Work load Average/day** (`Work_load_Average/day_`): Average workload per day.
11. **Hit target** (`Hit_target`): Performance target hit percentage.
12. **Disciplinary failure** (`Disciplinary_failure`): 1 if the employee has disciplinary failures, otherwise 0.
13. **Education** (`Education`): Education level - High school (1), Graduate (2), Postgraduate (3), Master & Doctor (4).
14. **Son** (`Son`): Number of children.
15. **Social drinker** (`Drinker`): 1 if the employee drinks socially, otherwise 0.
16. **Social smoker** (`Smoker`): 1 if the employee smokes socially, otherwise 0.
17. **Pet** (`Pet`): Number of pets owned.
18. **Weight** (`Weight`): Employee’s weight.
19. **Height** (`Height`): Employee’s height.
20. **Body mass index** (`Body_mass_index`): Calculated BMI.
21. **Absenteeism time in hours** (`Absenteeism_time_in_hours`): Target variable representing total absence hours.

## Data Format
The dataset is provided in **CSV format**

## Baseline Model
A **Random Forest** model serves as the baseline for this challenge. Participants are encouraged to improve upon this by exploring:
- Feature engineering (e.g., interaction terms, binning categorical variables, handling missing values)
- Advanced modeling techniques (e.g., Gradient Boosting, Neural Networks, Time Series models)
- Hyperparameter tuning
- Feature selection strategies
- Data balancing techniques (if necessary)

## Evaluation Metric
The models will be evaluated based on **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** to assess their predictive performance.

## Abstract:

The database was created with records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil.

## Source:

Creators original owner and donors: Andrea Martiniano (1), Ricardo Pinto Ferreira (2), and Renato Jose Sassi (3).

E-mail address: 
andrea.martiniano'@'gmail.com (1) - PhD student;
log.kasparov'@'gmail.com (2) - PhD student;
sassi'@'uni9.pro.br (3) - Prof. Doctor.

Universidade Nove de Julho - Postgraduate Program in Informatics and Knowledge Management.

Address: Rua Vergueiro, 235/249 Liberdade, Sao Paulo, SP, Brazil. Zip code: 01504-001.

Website: http://www.uninove.br/curso/informatica-e-gestao-do-conhecimento/


**Happy Coding!** 🚀

