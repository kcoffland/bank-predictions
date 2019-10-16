# Banking Predictions

I have split this data science project into 4 different parts based on how I
like to structure my projects. The order of the parts is conveniently the prefix
to each Jupyter Notebook, and the name of the corresponding part follows.

[Define](./01_Define.ipynb) goes into depth discussing the overall problem, the
data that is available, and the metric utilized in my analysis.

[Discover](./02_Discover.ipynb) contains all of the EDA as well as my thoughts
on which models will work the best given the types and distributions of the data.

[Develop](./03_Develop.ipynb) creates and tunes the various models which were
hypothesized to be the best in Discover. This was the first time I utilized
Randomized Search to tune the models, so I tested its performance compared to 
hand tuning for the Gradient Boosting Classifier. Develop is also where I test to see
which model performs best in a 5-fold Cross-Validation. This is to ensure that
the models generalize well on any future data whether it be the holdout data or
any new clients.

[Deploy](./04_Deploy.ipynb) fits the
"best" model from Develop on a subset of the data that I did not utilize for
tuning. My final summary of the performance of my machine learning model as well
as recommendations for the bank can be found in [Final
Summary](./04_Deploy.ipynb#Final-Summary). My final model created a __27.2%
increase in profit__ for the bank based on my estimation for a bank's profit on 
 each term deposit.
The bank can increase their profit further than the model
predicted by utilizing both the information gained by EDA and the importance of features
in the final model. All of these suggestions can be found in the Final Summary.
My thoughts on the project as a whole as well as steps to take in the near future
can be found [here](./04_Deploy.ipynb#What-I'll-Change-in-Future-Projects'), 
directly after the final summary.

I hope you enjoy the project as much as I did!
