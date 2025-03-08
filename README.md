# Data Challenge: Predicting reason of employee absence

## Overview
This data challenge focuses on predicting the reason of absence of employees at a brazilian courier company based on various personal, work-related, and health-related factors. The motivation of this challenge is to permit companies and public administrations to help detect or asses potential fraud by comparing the reason of absence at work declared by employees and the reason predicted by a machine learning model.

## Dataset Description
The database was created with records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil.
It contains 21 input features and 1 categorical target variable that is `Reason_for_absence`. Each line represents an absence instance with multiple attributes related to personal demographics, work conditions, and health status.

Creators of the dataset : Martiniano, A. & Ferreira, R. (2012). Absenteeism at work [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X882. (The dataset is provided in **CSV format**.)

### Features
1. **Individual ID** (`ID`): Unique identifier for each employee.
2. **Reason for absence** (`Reason_for_absence`): Categorized according to the International Code of Diseases (ICD) and additional reasons such as consultations, physiotherapy, and unjustified absence. (TARGET VARIABLE)
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
21. **Absenteeism time in hours** (`Absenteeism_time_in_hours`): total absence hours.


## Evaluation Metric
Given the difficulty to predict the exact reason of absence among a lot of possibilities, the evaluation metric will be the top 4 F1 score.




**Happy Coding!** 🚀
