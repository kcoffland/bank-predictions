
# Bank Predictions

# Step 3: Develop


```python
# Importing general packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import sys
sys.path.insert(0, './scripts/')

from scripts.helpers import load_data, get_smotenc, get_upsample, bank_profit, get_downsample
from scripts.Preprocessing import Preprocessing
from scripts.ModelTuner import ModelTunerCV

%matplotlib inline
```


```python
# Importing all sci-kit learn packages
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
```

I need set aside a portion of the data to evaluate my final model on. This will be saved in a file called *bank_holdout.csv* in the *holdout_data* directory.


```python
bank_data_full = load_data('./cleaned_data/', 'bank-full.csv', verbose=True, index=0)
bank_data_tuning, bank_data_holdout = train_test_split(bank_data_full, random_state=849)
```

    --------------------------------bank-full shape---------------------------------
                                      (36013, 17)                                   
    
    
    ----------------------------bank-full's column types----------------------------
    job                object
    marital            object
    education          object
    default            object
    housing            object
    loan               object
    contact            object
    month              object
    day_of_week        object
    previous            int64
    poutcome           object
    emp.var.rate      float64
    cons.price.idx    float64
    cons.conf.idx     float64
    euribor3m         float64
    nr.employed       float64
    y                   int64
    dtype: object
    
    
    ---------------------------bank-full first five rows----------------------------
         job  marital    education default  housing    loan       contact month  \
    0  other  married     basic.4y      no  not yes      no  not cellular   may   
    1  other  married  high.school  not no  not yes      no  not cellular   may   
    2  other  married  high.school      no      yes      no  not cellular   may   
    4  other  married  high.school      no  not yes  not no  not cellular   may   
    5  other  married     basic.9y  not no  not yes      no  not cellular   may   
    
      day_of_week  previous     poutcome  emp.var.rate  cons.price.idx  \
    0         mon        -1  nonexistent           1.1          93.994   
    1         mon        -1  nonexistent           1.1          93.994   
    2         mon        -1  nonexistent           1.1          93.994   
    4         mon        -1  nonexistent           1.1          93.994   
    5         mon        -1  nonexistent           1.1          93.994   
    
       cons.conf.idx  euribor3m  nr.employed  y  
    0          -36.4      4.857       5191.0  0  
    1          -36.4      4.857       5191.0  0  
    2          -36.4      4.857       5191.0  0  
    4          -36.4      4.857       5191.0  0  
    5          -36.4      4.857       5191.0  0  



```bash
%%bash
if [ -d "holdout_data" ]; then rm -R "holdout_data"; fi
mkdir "holdout_data"
```


```python
bank_data_holdout.to_csv('./holdout_data/bank_holdout.csv')
bank_data_tuning.to_csv('./holdout_data/bank_train.csv')
```


```python
bank_data_tuning.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19687</th>
      <td>admin.</td>
      <td>married</td>
      <td>university.degree</td>
      <td>not no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>aug</td>
      <td>thu</td>
      <td>-1</td>
      <td>nonexistent</td>
      <td>1.4</td>
      <td>93.444</td>
      <td>-36.1</td>
      <td>4.968</td>
      <td>5228.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3702</th>
      <td>technician</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>not yes</td>
      <td>no</td>
      <td>not cellular</td>
      <td>may</td>
      <td>fri</td>
      <td>-1</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.859</td>
      <td>5191.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4486</th>
      <td>technician</td>
      <td>married</td>
      <td>professional.course</td>
      <td>no</td>
      <td>not yes</td>
      <td>no</td>
      <td>not cellular</td>
      <td>may</td>
      <td>tue</td>
      <td>-1</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.856</td>
      <td>5191.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23951</th>
      <td>technician</td>
      <td>divorced</td>
      <td>professional.course</td>
      <td>not no</td>
      <td>yes</td>
      <td>not no</td>
      <td>cellular</td>
      <td>aug</td>
      <td>fri</td>
      <td>-1</td>
      <td>nonexistent</td>
      <td>1.4</td>
      <td>93.444</td>
      <td>-36.1</td>
      <td>4.963</td>
      <td>5228.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7482</th>
      <td>other</td>
      <td>single</td>
      <td>high.school</td>
      <td>no</td>
      <td>not yes</td>
      <td>no</td>
      <td>not cellular</td>
      <td>may</td>
      <td>fri</td>
      <td>-1</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.864</td>
      <td>5191.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
clients_train, clients_tune, subscribed_train, subscribed_tune = train_test_split(
    bank_data_tuning.drop('y', axis=1), bank_data_tuning['y'], random_state=536)
```

I need to make the scoring object to be passed into my hyperparameter search and cross validation score calculations. Sci-kit Learn makes a convenient function if a user wants to make their own scorer, and it is exactly what I need to make my profit function into a viable scorer.


```python
profit_score = make_scorer(bank_profit, needs_proba=True)
```

## Engineering Features
* Ensure data is ready for modeling
* Create any new features to enhance the model

I've made a few changes to my Preprocessing class to make it more robust and utilize more of sci-kit learn's methods for feature transformations. These changes have made my code more robust and easier to read overall. 


```python
# p_ will stand for processed data
p = Preprocessing(classification=True)

# Preprocessing the normal data
p_clients_train, p_subscribed_train = p.fit_transform(
    clients_train, y=subscribed_train)
p_clients_tune, p_subscribed_tune = p.transform(
    clients_tune, subscribed_tune)
p_clients_cv, p_subscribed_cv = p.fit_transform(bank_data_tuning.drop('y', axis=1), bank_data_tuning['y'])

p_clients_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>previous</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>day_of_week_fri</th>
      <th>day_of_week_mon</th>
      <th>day_of_week_thu</th>
      <th>day_of_week_tue</th>
      <th>...</th>
      <th>housing_not yes</th>
      <th>housing_yes</th>
      <th>poutcome_nonexistent</th>
      <th>poutcome_not nonexistent</th>
      <th>job_admin.</th>
      <th>job_blue-collar</th>
      <th>job_other</th>
      <th>job_technician</th>
      <th>loan_no</th>
      <th>loan_not no</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29029</th>
      <td>-0.348419</td>
      <td>-1.193410</td>
      <td>-0.850611</td>
      <td>-1.423553</td>
      <td>-1.278385</td>
      <td>-0.946804</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>536</th>
      <td>-0.348419</td>
      <td>0.651892</td>
      <td>0.745600</td>
      <td>0.882558</td>
      <td>0.714621</td>
      <td>0.330398</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>21094</th>
      <td>-0.348419</td>
      <td>0.842785</td>
      <td>-0.209695</td>
      <td>0.947216</td>
      <td>0.776397</td>
      <td>0.846003</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>36726</th>
      <td>-0.348419</td>
      <td>-1.893352</td>
      <td>-1.045144</td>
      <td>-0.065749</td>
      <td>-1.357481</td>
      <td>-1.265062</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29680</th>
      <td>-0.348419</td>
      <td>-1.193410</td>
      <td>-0.850611</td>
      <td>-1.423553</td>
      <td>-1.278385</td>
      <td>-0.946804</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>



## Creating Models
* Creating and tuning models ([Logistic Regression](#Creating-Models,-Logistic-Regression), [Gradient Boosting](#Creating-Models,-Gradient-Boosting), [Random Forest](#Creating-Models,-Random-Forest))

### Creating Models, Baseline

Before anything, I want to have a baseline of how well my classifiers will end up being. To do so, I will be creating a plain LogisticRegression model without changing any of the hyperparameters. This is just a quick and dirty way to create a classifier that might be used in an environment unfamiliar with Data Science.


```python
lr = LogisticRegression()
cross_val_score(lr, p_clients_cv, p_subscribed_cv, scoring=profit_score, cv=5).mean()
```

    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    0.5859457821853848



A quick note, for this project, I will be utilizing RandomizedSearchCV instead of tuning the models by hand. I want to eventually automate the model tuning process, and this method seemed to be better than the more robust, but incredibly more time consuming, GridSearchCV.

### Creating Models, Logistic Regression


```python
params = {'C': sp.uniform(0.001, 5),
          'solver': ['newton-cg', 'sag', 'lbfgs']
          }
lr_tuner = ModelTunerCV(LogisticRegression(), profit_score, cv=5)

lr_tuner.tune(X=p_clients_cv, 
              y=p_subscribed_cv, 
              param_grid=params, 
              method='random')
print(f"Best Params: {lr_tuner.best_params_}")
print(f"Best Profit Score: {lr_tuner.best_score_}")
```

    Best Params: {'C': 1.413801090996802, 'solver': 'lbfgs'}
    Best Profit Score: 0.5872713913239094


    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)


### Creating Models, Gradient Boosting


```python
gbc = GradientBoostingClassifier(random_state=985)
cross_val_score(gbc, p_clients_cv, p_subscribed_cv, scoring=profit_score, cv=3).mean()
```




    0.6010280557861455




```python
gbc_param_dist = {'learning_rate': sp.uniform(.0001, .2), # Picks from a range of [.0001, .2001]
    'subsample': sp.uniform(0.5, .5), # Picks from a range of [.5, 1]
    'min_samples_leaf': sp.randint(1, 5),
    'max_depth': sp.randint(2, 15),
    'max_features': sp.uniform(.01, .99)
}

gbc_tuner = ModelTunerCV(gbc, profit_score, cv=3)
gbc_tuner.tune(X=p_clients_cv, 
              y=p_subscribed_cv, 
              param_grid=gbc_param_dist, 
              method='random',
              n_iter=15)
print(f"Best Params: {gbc_tuner.best_params_}")
print(f"Best Profit Score: {gbc_tuner.best_score_}")
```

    Best Params: {'learning_rate': 0.1020804819179213, 'max_depth': 5, 'max_features': 0.21603778511971944, 'min_samples_leaf': 2, 'subsample': 0.7283105544694686}
    Best Profit Score: 0.5995901208929921



```python
gbc = GradientBoostingClassifier(random_state=985, learning_rate=.11, subsample=.8, max_depth=5, min_samples_leaf=8)
cross_val_score(gbc, p_clients_cv, p_subscribed_cv, scoring=profit_score, cv=3).mean()
```




    0.5985529068481122



To test how well the randomized cv can select good enough parameters, I manually tuned the hyperparameters for Gradient 
Boosting as well as the randomized cv. The randomized cv took about 3 minutes and tuning manually took me a few hours of mainly waiting and working on other things like updating documentation while waiting for the cross validation scores to be calculated (This may be less significant of a time sink on your machine, but mine is rather old and slow with these amount of calculations). At the end of the day, there was barely any difference in the profit scores between the hand tuned and randomized search. The time gain of the randomized search is definitely worth the potential drop in score due to uncertainty in tuning. At the very least, randomized search will almost certainly always be "good enough" for production. The performance of the hyperparameter tuning might also be due to the choice of model. I'll repeat with Random Forest which will depend heavily on its hyperparameters.

### Creating Models, Random Forest


```python
rf = RandomForestClassifier(random_state=4215, n_estimators=100)
cross_val_score(rf, p_clients_cv, p_subscribed_cv, scoring=profit_score, cv=3).mean()
```




    0.5447568326912399




```python
rf_param_dist = {'max_depth': sp.randint(7, 50),
                 'min_samples_leaf': sp.uniform(.01, .49),
                 'max_features': sp.uniform(.5, .5)
}

rf_tuner = ModelTunerCV(rf, profit_score, cv=5)
rf_tuner.tune(X=p_clients_cv, 
              y=p_subscribed_cv, 
              param_grid=rf_param_dist, 
              method='random',
             n_iter=15)
print(f"Best Params: {rf_tuner.best_params_}")
print(f"Best Profit Score: {rf_tuner.best_score_}")
```

    Best Params: {'max_depth': 27, 'max_features': 0.7162708532243822, 'min_samples_leaf': 0.025569653025819587}
    Best Profit Score: 0.5908728886090605


## Testing Models
* Doing a 5-fold cross validation on models


```python
def perform_cv5(model, params, X, y):
    model.set_params(**params)
    return cross_val_score(model, X, y, scoring=profit_score, cv=5).mean()
```


```python
print(f"Logistic Regression: {perform_cv5(lr, lr_tuner.best_params_, p_clients_cv, p_subscribed_cv)}")
print(f"Gradient Boosting: {perform_cv5(gbc, gbc_tuner.best_params_, p_clients_cv, p_subscribed_cv)}")
print(f"Random Forest: {perform_cv5(rf, rf_tuner.best_params_, p_clients_cv, p_subscribed_cv)}")
```

    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /home/kyle/DSDJ/Module4/bank-predictions/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)


    Logistic Regression: 0.5857732004055214
    Gradient Boosting: 0.5988379987809191
    Random Forest: 0.5918078974929023


## Selecting the Best Model
* Selecting the model with the highest score for production

By an incredibly small margin, Gradient Boosting has the best profit score after averaging the 5-fold cross validation scores. The parameters will be listed below to be used in the model in the final steps.


```python
gbc = GradientBoostingClassifier(random_state=985).set_params(**gbc_tuner.best_params_)
gbc.get_params()
```




    {'criterion': 'friedman_mse',
     'init': None,
     'learning_rate': 0.1020804819179213,
     'loss': 'deviance',
     'max_depth': 5,
     'max_features': 0.21603778511971944,
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 2,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 100,
     'n_iter_no_change': None,
     'presort': 'auto',
     'random_state': 985,
     'subsample': 0.7283105544694686,
     'tol': 0.0001,
     'validation_fraction': 0.1,
     'verbose': 0,
     'warm_start': False}


