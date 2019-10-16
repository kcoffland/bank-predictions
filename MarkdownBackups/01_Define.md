
# Bank Predictions

# Step 1: Define

## Overall Problem and Motivation

The goal of the analysis is to predict whether a client will subscribe to a [term deposit](https://www.investopedia.com/terms/t/termdeposit.asp) for a banking institution in Portugal. Banks want term deposits so that they have a more consistent stream of capital to fund other investments they wish to make. The profit that banks make off of this type of transaction is called the [Net Interest Margin (NIM)](https://www.investopedia.com/ask/answers/061715/what-net-interest-margin-typical-bank.asp). Clients buy term deposits and are guaranteed a low interest rate. The difference between this and a normal checking/savings account with a bank is that the interest rate is higher and there is a penalty if the client wishes to withdraw their money prematurely. Term deposits are a low-risk investment for the client, but the reward is appropriately low. 

Banks would like to know if there are trends in which clients buy term deposits so that they can focus their time and resources contacting those potential investors rather than some who might never want to make this type of investment. A more in depth cost analysis can be found [here](#Evaluation-Metric,-Probability-Threshold). I want to find the most significant traits which would lead a client to subscribe to a term deposit.

One of the biggest influences on term deposits is interest rates. In general, the higher the interest rates, the more the client will earn from the term deposit. Conversely, when interest rates drop, then the economy is generally doing better. Potential clients might see more potential gains through the stock market rather than term deposits with a bank.

## The Data

The data was provided by

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

and can be found [here](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#)


The file I will be utilizing is called bank-additional-full.csv. It contains 41,118 clients and 20 features. There are additional features and more clients than the dataset which was originally created to do this analysis. The 20 features are:
* age (numeric)
* job: Type of job (categorical, might end up being ordinal)
    - 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'
* marital: marital status (categorical)
    - 'divorced', 'married', 'single', 'unknown'; note: 'divorced' means divorced or widowed
* education: (categorical, might be ordinal)
    - 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'
* default: is credit in default or not (categorical)
* housing: has a housing loan or not (categorical)
* loan: has a personal loan or not (categorical)
* contact: contact communication type (categorical)
    - 'cellular','telephone'
* month: last contact month of the year (categorical)
    - 'jan', 'feb', 'mar', ..., 'nov', 'dec'
* day_of_week: last contact day of week (categorical)
    - 'mon', 'tue', 'wed', 'thu', 'fri'
* duration: duration of last contact in seconds (I will explain why this feature needs to be removed)
* campaign: number of contacts performed during this campaign for this client (numeric)
    - includes last contact with the client
* pdays: number of days that passed by after the client was last contacted from a __previous__ campaign (numeric)
    - 999 means the client was never previously contacted
* previous: number of contacts performed before this campaign and for this client (numeric, includes last contact)
* poutcome: outcome of the previous marketing campaign (categorical)
    -  'failure', 'nonexistent', 'success'
    
Note that the rest of the feature variables shown below are social and economic context attributes
* emp.var.rate: employment variation rate - quarterly indicator (numeric)
* cons.price.idx: consumer price index - monthly indicator (numeric)
* cons.conf.idx: consumer confidence index - monthly indicator (numeric)
* euribor3m: euribor 3 month rate - daily indicator (numeric)
* nr.employed: number of employees - quarterly indicator (numeric)

The target variable is:
* y: has the client subscribed a term deposit (binary)


#### Dropping 'duration'
The variable duration will need to be removed since this variable will be known only after the phone call to the client has ended. The information cannot possibly be known before a call to the potential client subscription, and therefore cannot be used to predict whether the call will be successful. The goal of the model is to predict the success of a phone call __before__ the call has been made. I will however use *duration* to estimate a cost-benefit analysis for making these calls.

#### Explaining employment variation rate ("emp.var.rate")
Employment variation rate tracks how much a company is hiring during a given quarter. This metric can be viewed as proportional to how companies view the economy at each quarter. If the economy is perceived to be doing well, then the *emp.var.rate* will go up and vice versa. It can then be implied that if the *emp.var.rate* is up, then interest rates will be down as well as the rate at which clients will be subscribing for term deposits.

#### Explaining cosumer price index ("cons.price.idx")
[Consumer Price Index (CPI)](https://www.investopedia.com/terms/c/consumerpriceindex.asp) measures the average change in prices over time that consumers pay for a basket of goods and services. CPI is a monthly indicator used to track inflation in a country. Any sudden change in CPI can be disastrous for economies causing either hyperinflation or severe deflation. The CPI is a key indicator of changes in the interest rate generally held to an inversely proportional relationship by banks. Further reading on the relationship between inflation and interest rates can be found [here](https://www.investopedia.com/ask/answers/12/inflation-interest-rate-relationship.asp)

#### Explaining consumer confidence index ("cons.conf.idx")
The [Consumer Confidence Index (CCI)](https://www.investopedia.com/terms/c/cci.asp) measures how *consumers* feel about the near future of the economy on a monthly basis. It tries to predict whether consumers will have faith in the market and spend or they'll be skeptical and save. When consumers have faith in the market, it can be reasoned that the economy will generally grow making the interest rates fall. For example, the CCI hit record lows after the 2008 housing market collapse. This produces an inversely proportional relationship between interest rates and CCI which infers an inversely proportional relationship with term deposist subscription rates. For reference, the CCI hit record lows after the 2008 housing market collapse.

#### Explaining euribor 3 month rate ("euribor3m")
The euribor 3 month rate is the interest rate of a subset of European banks lend one another funds with a 3 month maturity. This rate is used to inform European banks on the interbank interest rates in the rest of the Europe. If there are significant changes in the euribor 3 month rate, then there is a high likelyhood that interest rates across Europe are increasing. More in-depth detail on Euribor can be found [here](https://www.euribor-rates.eu/what-is-euribor.asp)

## Evaluation Metric
As I was creating models and evaluating them based on Average Precision or ROC Area Under Curve, I always noticed that these metrics did not capture anything that I wanted to use for evaluating how my model was performing. They gave a great understanding of how the recall and precision were balanced in the various models, but they were completely useless for evaluating which model would be best for the bank given this data. After quite a few hours more experimenting and researching, I have come to a conclusion that should have probably been obvious from the start: I need to maximize profit, and use that profit as my metric. To ensure that the measurement does not vary because a sample size is different between tests, I will be normalizing the profit by the total amount of money that could have possibly been made if the model was perfect. 

$$
\text{Profit Score}= \frac{benefit \times TP - cost \times FP}
              {benefit \times TotalNumPositives}             
$$

where *benefit* is the amount of money made off of a client who has subscribed to a term deposit, and *cost* is the amount of money wasted when you have failed to secure the term deposit. These are calculated below as a part of finding the probability threshold. *TP* and *FP* are the number of true positives and false positives respectively. The profit score is maximal when there are only true positives and no false positives, but the values of benefit and cost will be so that the model highly values true positives and will not care if a few false positives occur. This is because the benefits of a newly subscribed term deposit heavily outweigh the cost of a few wasted minutes. The profit score will have a maximum of 1.

I thought about comparing it to the naive baseline of calling every potential subscriber, but this creates a wide variety of scores that are of different sample size or success rate. The baseline score can easily be calculated for each sample to ensure that the model produced will perform better than naive actions.

### Evaluation Metric, Probability Threshold

Right now, I need to determine what my threshold will be for a positive outcome. I want to do so by finding the probability of subscription threshold that still has a positive expected value. Classification algorithms calculate the probability of positive outcome, and it is up to me to supply whether that probability of positive outcome will be large enough to warrant a call.

There are two outcomes that can happen when the bank calls a client: either they subscribe to a term deposit, or they do not. The expected value can be broken down as so

$$
E = \text{benefit}*(\text{probability of subscription}) - \text{cost}*(1-\text{probability of subscription}) 
$$

$$
cost = \text{average duration for failure} * \text{number of calls for failure} * \text{employee hourly wage}
$$

$$
benefit = NIM * \text{average term deposit price} - \text{average duration for success} * \text{number of calls for success} * \text{employee hourly wage}
$$

The only values I can calculate from the data are the *average duration for failure/success* and *number of calss for failure/success*. Unfortunately, the rest of the terms need to be estimated. From my research, the average wage of a Portugese bank teller is &euro;5 per hour, which corresponds to an average of \\$7 per hour for the years 2008-2010 in which the data was collected. NIM for European Union banks has hovered around 2\% for many years, so I'll be using that as my value. The lowest minimum deposit for term deposits seems to be \\$1000. I'll use the lowest price to make sure I do not assume a deposit price that would inflate the benefit and lower the probability threshold.

Now I need to calculate the rest of the values from the data.


```python
import pandas as pd

bank_data = pd.read_csv('./raw_data/bank-full.csv', sep=';')

# Converting seconds to hours
bank_data['duration'] /= 3600
bank_data.head()
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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>...</th>
      <th>campaign</th>
      <th>pdays</th>
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
      <th>0</th>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>unknown</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>admin.</td>
      <td>married</td>
      <td>basic.6y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
failure = bank_data[bank_data['y']=='no']
success = bank_data[bank_data['y']=='yes']
hourly_wage = 7
NIM = .02
price = 1000

average_failure = failure[['duration', 'campaign']].mean()
average_success = success[['duration', 'campaign']].mean()

cost = average_failure['duration'] * average_failure['campaign'] * hourly_wage
benefit = NIM*price - average_success['duration']*average_success['campaign']*hourly_wage

print(f'Cost: ${cost}')
print(f'Benefit: ${benefit}')
```

    Cost: $1.1307006766058485
    Benefit: $17.793063987357968


Assuming I want a positive Expected Value, I get the equation:

$$
\text{benefit}*(\text{probability of subscription}) - \text{cost}*(1-\text{probability of subscription}) > 0
$$

Solving for *probability of subscription*, I arrive at:

$$
\text{probability of subscription} > \frac{cost}{benefit + cost}
$$


```python
prob_threshold = cost / (benefit + cost)

print(f"Probability of Subscription must be above {prob_threshold:.2f} to be marked positive")
```

    Probability of Subscription must be above 0.06 to be marked positive


Granted, this is a rough estimate, but it is a more informed threshold than a naive .5 probability threshold. The .06 threshold makes a lot of sense because there is not a lot of value lost by calling a client and not receiving a subscription. But, there is great value in maximizing the number of clients who will be subscribing. Thus, a low probability threshold is warranted. A higher threshold might be used in some other aspect when missing a positive (having a false negative) is something that can easily be lived with, such as missing on a client that would give little value. Other times, a lower threshold will be used because a false negative could have dire consequences, such as missing a cancer diagnosis.

## Output
The output of my model will be a csv file named "subscription_predictions.csv" in which there will be a single column filled with 1s and 0s. 1s will represent the clients who are predicted to subscribe to a term deposit this campaign. The 0s will be the clients who are predicted to not subscribe to a term deposit this marketing campaign. 
