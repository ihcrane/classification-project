# Telco Churn Project

## Project Description
Telco is a internet and streaming provider that has experienced high levels of churn and wants to understand why in order to improve business practices to better develop and retain a customer base moving forwards.

## Project Goal
- To discover the drivers of churn within Telco
- Use the drivers to develop a ML program that predicts churn with at least 80% accuracy
- Deliver a report to a non-techinical supervisor in a digestable manner

## Questions to Answer
- Does monthly charges affect churn?
- Does payment type affect churn?
- Does contract type affect churn?
- Does having dependents affect churn?
- Of the listed drivers, which has the most impact in regards to churn?

## Initial Thoughts and Hypothesis
I believe that the main drivers behind churn will be monthly cost and contract type with the assumption that lower monthly charges and longer contracts being less likely to churn compared to the high monthly charges and shorter contract types. I also think that having dependants will also impact churn do to having more than one line to handle.

## Planning
- Use the aquire.py already used in previous exerices to aquire the data necessary
- Use the code already written prior to aid in sorting and cleaning the data
- Discover main drivers
  - First identify the drivers using statistical analysis
  - Create a pandas dataframe containing all relevant drivers as columns
- Develop a model using ML to determine churn based on the top 3 drivers
  - MVP will consist of one model per driver to test out which can most accurately predict churn
  - Post MVP will consist of taking most accurate and running it through multiple models
  - Goal is to achieve at least 80% accuracy with at least one model
- Draw and record conclusions

## Data Dictionary
| Feature | Description |
| --- | --- |
| churn | When a customer cancels contract or subscription with the company |
| contract_type | The type of contract that the customer has with Telco |
| payment_type | The form in which the customer pays their monthly bill |
| dependents | Whether or not the customer has a dependent on their account |
| monthly_charges | How much a customer pays per month |
| tenure | How long a customer has been with the company |
| total_charges | How much a customer has paid over their entire tenure |
| payment_type_id | Number assignments for stats purposes |
| contract_type_id | Number assignments for stats purposes |
| multiple_lines | Whether customer has more than one phone line on account |

