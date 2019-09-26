import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import os
from imblearn.over_sampling import SMOTENC
import scipy.stats as sp
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample


def load_data(path, file, verbose=False, sep=',', index=None):
    """Reads in the data at the given path/filename. Currently file is assumed to be csv 
    Args:
        path: string, path to file containing data
        file: string, name of file containing data that will be merged together
        verbose: boolean, If the caller wanted information about the df to be displayed
        index: int, Number of column that is the index
    Returns: 
        df, holds data in the given csv files
    """
    
    df = pd.read_csv(path+file, index_col=index, sep=sep)
    
    if verbose:
        shape = f'{df.shape}'
        dtypes = f'{df.dtypes[:30]}'
        head = f'{df.head()[:10]}'
        name = file.split('.')[0]
        
        print(f'{name} shape'.center(80, '-'))
        print(shape.center(80))
        print()
        print()
        print(f"{name}'s column types".center(80, '-'))
        print(dtypes)
        print()
        print()
        print(f"{name} first five rows".center(80, '-'))
        print(head)
    
    return df

def mean_squared_error(predictions, actual):
    """Computes the MSE for a given prediction and observed data
    Args:
         predictions: Series or float/int, holding predicted target value
         actual: Series, holding actual target value
    """

    return ((predictions-actual)**2).sum() / len(actual)

def find_outliers(data, method='iqr'):
    """Finds the values for which outliers begin
    Args:
        data: DataFrame or Series, holds data you want to find the outliers for
        method: string, Method used to calculate an outlier
    Returns:
        lower, upper: floats or Series of floats, values less than lower are outliers and
                              values larger than upper are outliers. A Series of 
                              floats corresponding to the lower and upper values
                              of the columns will be returned if data is a DataFrame
    """

    if type(data) != pd.core.frame.DataFrame or type(data) != pd.core.series.Series:
        raise TypeError("data must be either a DataFrame or Series")

    if method=='iqr':
        # Finding the interquartile range
        q1 = data.quantile(.25)
        q3 = data.quantile(.75)
        iqr = q3-q1

        upper = q3 + iqr*1.5
        lower = q1 - iqr*1.5
    elif method=='std':
        std = data.std()
        lower = data.mean() - 3*std
        upper = data.mean() + 3*std
    else:
        raise ValueError("Invalid value for 'method' passed")


    return lower, upper

def classifier_categorical_variance(df, cat_cols, target_col, pos_val='yes'):
    """Calculates the variance of the target column due to changes in categorical columns.
       This method calculates how the target_col is split based on the cat_cols, normalizes 
       the result based on the total count, then calculates the variance in the normalized 
       target values. A higher variance corresponds to a feature that can better describe
       changes in rates of the target variable. For example, if the number of rejections 
       varies a lot from month to month, then that can help us separate the data
       
    Args:
        df: DataFrame, Data you want to find the variance on
        cat_cols: list of strings, columns in the dataframe which are categorical
        target_col: string, The column which we want to measure how it varies based
                            on changes due to the different categories
    Returns:
        variances: DataFrame, contains the calculated variances for each given cat_col
                              sorted from highest to lowest
    """
    variances = pd.DataFrame(columns=df[target_col].unique())
    for col in cat_cols:
        counts = df.groupby(col)[target_col].value_counts()
        total_counts = counts.groupby(col).sum()
        scaled_counts = counts / total_counts
        for df_col in variances.columns:
            variances.loc[col, df_col] = scaled_counts.xs(df_col, level=target_col).var()
    
    return variances.sort_values(pos_val, ascending=False)

def plot_success_rates(df, cat_cols, target_col, num_cols=2):
    """Plots the success rates of the given categorical columns. This is only to be used 
       with a Classification problem

    Args:
        df: DataFrame, holds all the data, assumes target_col is in the data
        cat_cols: list of strings, all the names of columns which the user wants to plot
        target_col: string, name of the target column
        num_cols: int, number of columns in the final figure
    
    Returns:
        Nothing
    """
    n = len(cat_cols)
    num_rows = ceil(n/num_cols)
    fig, axes = plt.subplots(nrows=num_rows, 
                           ncols=num_cols,
                           sharey='row', 
                           figsize=(4*num_cols, 5*num_rows)
                          )
    
    for cat_col, ax in zip(cat_cols, axes.flatten()):
        counts = df.groupby(cat_col)[target_col].value_counts()
        total_counts = counts.groupby(cat_col).sum()
        norm_counts = counts / total_counts

        norm_counts.unstack().plot(kind='bar', ax=ax)
        ax.set_ylabel('Success Rate')
        ax.set_title(f'Comparing {cat_col} Success Rates')
    
    # Removing any extra plots
    for i in range(n, len(axes.flatten())):
        fig.delaxes(axes.flatten()[i])
        
    plt.tight_layout()
    plt.show()

def get_smotenc(X, y, cat_cols):
    """Upsamples categorical and non-categorical data using SMOTENC package
    Args:
        X: DataFrame, feature data that needs to be upsampled
        y: Series, tags corresponding to the given features
    
    Returns:
        us_X: DataFrame, upsampled feature data
        us_y: Series, upsampled target data
    """
    # Finding which indexes are categorical
    categorical_mask = [index for index, col in enumerate(X) if col in cat_cols]
    X_dtypes = X.dtypes

    smote = SMOTENC(categorical_mask, random_state=44, n_jobs=-1, k_neighbors=3)

    upsampled_data, upsampled_results = smote.fit_resample(X, y)

    # Converting the numpy arrays back to dataframes and series objects
    us_X = pd.DataFrame(upsampled_data, columns=X.columns)
    us_y = pd.Series(upsampled_results)

    # The data types are all defaulted to 'object' so I am fixing that
    for col in us_X:
        us_X[col] = us_X[col].astype(X_dtypes[col])
    return us_X, us_y

def get_upsample(X, y):
    """Upsamples categorical and non-categorical data using bootstrapping
    Args:
        X: DataFrame, feature data that needs to be upsampled
        y: Series, tags corresponding to the given features
    
    Returns:
        us_X: DataFrame, upsampled feature data
        us_y: Series, upsampled target data
    """
    if y.nunique() != 2:
        raise ValueError("Target variable must only have 2 classes. Currently, "
                         f"there are {y.unique()} unique classes")
    if type(y) != pd.core.series.Series:
        raise TypeError(f"y parameter type is {type(y)} instead of Series")
    if type(X) != pd.core.frame.DataFrame:
        raise TypeError(f"X parameter type is {type(X)} instead of DataFrame")

    # Finding the minority value and the count of the majority class
    count = y.value_counts()
    majority_count = count.iloc[0]
    minority_value = count.index[1]

    # Creating the majority and minority dfs
    df = X.join(y)
    minority_df = df[df[y.name] == minority_value].copy()
    majority_df = df[df[y.name] != minority_value].copy()

    minority_us = resample(minority_df, replace=True, n_samples=majority_count, 
                           random_state=44)
    
    # Full data upsampled
    df_us = pd.concat([majority_df, minority_us])

    return df_us.drop(y.name, axis=1), df_us[y.name]

def bank_profit(y_true, y_pred):
    """Calculating the bank's profit using a cost benefit analysis.
    Args:
        y_true: array-like, what the predictions should be
        y_pred: array-like, prediction score, output of predict_proba or decision function
    """
    # THRESHOLD calculated in 01_Define of bank-predictions repo
    THRESHOLD = .06

    # Model's predicted confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred>=THRESHOLD).ravel()

    # Perfect Confusion Matrix
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true, y_true).ravel()

    del tn, fn, tn_p, fn_p
    # BENEFIT and COST were calculated in 01_Define of bank-predictions repo
    BENEFIT = 17.79
    COST = 1.13

    model_profit = tp*BENEFIT - fp*COST
    perfect_profit = tp_p*BENEFIT - fp_p*COST

    normalized_profit = model_profit / perfect_profit
    return normalized_profit

def baseline_profit(y_true):
    """Returns the profit with the assumption that every client will be called
    
    Args:
        y_true: array-like, The true values to be passed to bank_profit
    
    Returns:
        The baseline score
    """
    return bank_profit(y_true, np.ones(y_true.shape[0]))

def get_downsample(X, y):
    """Downsamples categorical and non-categorical data using bootstrapping
    Args:
        X: DataFrame, feature data that needs to be downsampled
        y: Series, tags corresponding to the given features
    
    Returns:
        us_X: DataFrame, downsampled feature data
        us_y: Series, downsampled target data
    """
    if y.nunique() != 2:
        raise ValueError("Target variable must only have 2 classes. Currently, "
                         f"there are {y.unique()} unique classes")
    if not check_type(y, 'SERIES'):
        raise TypeError(f"y parameter type is {type(y)} instead of Series")
    if not check_type(X, 'DATAFRAME'):
        raise TypeError(f"X parameter type is {type(X)} instead of DataFrame")

    # Finding the minority value and the count of the minority class
    count = y.value_counts()
    minority_count = count.iloc[1]
    minority_value = count.index[1]

    # Creating the majority and minority dfs
    df = X.join(y)
    minority_df = df[df[y.name] == minority_value].copy()
    majority_df = df[df[y.name] != minority_value].copy()

    majority_ds = resample(majority_df, replace=True, n_samples=minority_count, 
                           random_state=44)
    
    # Full data upsampled
    df_ds = pd.concat([majority_ds, minority_df])

    return df_ds.drop(y.name, axis=1), df_ds[y.name]

def check_type(obj, typ):
    """
    Args:
        obj: object, object that the type wants to be checked for
        typ: string, type the object should be
    Returns:
        ans: bool, False if the object is the wrong type and true if the object 
            is of the correct type
    """
    types = {'SERIES': pd.core.series.Series,
        'DATAFRAME': pd.core.frame.DataFrame,
        'LIST': list,
        'BOOL': bool}
    try:
        ans = type(obj) == types[typ]
    except KeyError:
        raise KeyError(f"Invalid input for typ, must be one of {list(types.keys())}")

    return ans
    
def z_score(mean, std, value):
    return (value-mean) / std

def significant_cat_classification(ser_data, target_data, n_samples=500, 
    pos_val=1, alpha=.05, value=None):
    """Testing to see if any value in the Series given is significantly different
       than the expected success rate

    Args:
        ser_data: Series, Categorical column to test if any values it has varies
                    greatly from the norm
        target_data: Series, Values for the given row
        n_samples: int, number of samples to calculate the distribution
        pos_val: int or string, the value which denotes the postitive value in 
                    the target data given
        alpha: float, the significance level used in the p-test to determine whether 
                    a value is statistically significantly different from the norm
                    expected of it. See (INSERT WIKIPEDIA LINK HERE) for more details
        value: string, specific value in ser_data that the user wants to test
                Defaults to testing every value in the series given
    
    Returns:
        bool, True if any of the values in ser_data is statistically significant
            False if none of the values in ser_data is statistically significant
            In the case that "value" is given, then the specific value is tested 
            for significance and True is returned if it is significant
    """
    if not check_type(ser_data, 'SERIES'):
        raise TypeError(f"ser_data must be a series, {type(ser_data)} was given")
    if not check_type(target_data, 'SERIES'):
        raise TypeError(f"target_data must be a series, {type(target_data)} was given")

    if value is not None:
        if value not in ser_data.unique():
            raise ValueError("value does not exist in the series given, the series "
                f"contains {ser_data.unique()}, and {value} was given")
            

    # Calculating the expected success rate and sample size to reconstruct
    # samples from a Bernoulli distribution
    success_rate = target_data.value_counts()[pos_val] / target_data.shape[0]
    sample_size = int(25 / success_rate)     # Chosen to ensure the distribution is normal
    sample_means = np.zeros(n_samples)
    
    # Finding the sample distribution of the sample means
    for i, _ in enumerate(sample_means):
        sample_means[i] = sp.bernoulli.rvs(success_rate, size=sample_size).mean()
    mean = sample_means.mean()
    std = sample_means.std()
    
    # Building the actual rate observed
    temp = pd.concat([ser_data, target_data], axis=1)
    p_vals = []

    if value is not None:
        if value not in ser_data.unique():
            raise ValueError("value does not exist in the series given, the series "
                f"contains {ser_data.unique()}, and {value} was given")
        observed = temp[temp[ser_data.name]==value][target_data.name]
        if observed.shape[0] < 30:
            return False
        try:
            value = observed.value_counts()[pos_val] / observed.shape[0]
        except KeyError:
            value = 0
        z = z_score(mean, std, value)
        p_val = sp.norm.sf(abs(z)) #one-sided p-value

        return p_val < alpha
    
    for val_of_interest in ser_data.unique():
        observed = temp[temp[ser_data.name]==val_of_interest][target_data.name]
        if observed.shape[0] < 30:
            continue
        try:
            value = observed.value_counts()[pos_val] / observed.shape[0]
        except KeyError:
            value = 0
        z = z_score(mean, std, value)
        p_val = sp.norm.sf(abs(z)) #one-sided p-value
        p_vals.append(p_val)
    return any([p_val<alpha for p_val in p_vals])

def cat_null_cleaner(df, null_vals, target):
    drop_cols = []
    drop_rows = []
    for col, unknown in null_vals.items():
        if col == target:
            continue

        unknown_rate = df[df[col] == unknown].shape[0] / df.shape[0]
        if unknown_rate <= .05:
            drop_rows.append(df[df[col]==unknown].index)
        elif not significant_cat_classification(ser_data=df[col], target_data=df[target], pos_val='yes'):
            drop_cols.append(col)
    return drop_cols, drop_rows

def cat_small_counts(df, cat_cols, min_frac=.05, max_frac=.9, inplace=False):
    """If a categorical feature value has a small enough fraction of the data, 
       then it will be replaced with 'other'
    Args:
        df: DataFrame, Data which needs to be assessed
        cat_cols: list of strings, List of categorical columns which need to be assessed
        min_frac: float, Minimum fraction of data a value is allowed to be
        inplace: bool, Whether the dataframe is changed in place or a copy is made
        max_frac: float, Maximum fraction that one category can be before the
                feature is just made to be binary.
    Returns:
        df_new: DataFrame, the necessary values replaced or dropped depending
            on the case. The object will be new if inplace is set to False
    """
    if not check_type(df, 'DATAFRAME'):
        raise TypeError(f"df was expected to be a DataFrame object, {type(df)} was given")
    if not check_type(cat_cols, 'LIST'):
        raise TypeError(f"cat_cols was expected to be a list object, {type(cat_cols)} was given")
    if not check_type(inplace, 'BOOL'):
        raise TypeError(f"inplace was expected to be a bool object, {type(inplace)} was given")
        
    if not inplace:
        df_new = df.copy()
    else:
        df_new = df
    for col in cat_cols:
        counts = df[col].value_counts() / df[col].shape[0]
        # List of values with counts less than the given minimum
        small_val = counts.index[counts <= min_frac]
        largest_cat = counts.index[0]
        if counts.shape[0] <= 3 and not small_val.empty :
            # If there are a small amount of categories and there is a small
            # count for a category, then the feature is turned into a binary column
            df_new.loc[df_new[col] != largest_cat, col] = 'not ' + largest_cat
        elif counts.iloc[0] >= max_frac and counts.shape[0] > 2:
            # If the largest column dominates enough, then the rest of the
            # categories will be joined together
            df_new.loc[df_new[col] != largest_cat, col] = 'not ' + largest_cat
        
        elif counts.shape[0] - len(small_val) <= 2:
            # If there are enough rows that need to be combined, then it is just 
            # turned into a binary feature
            df_new.loc[df_new[col] != largest_cat, col] = 'not ' + largest_cat
        
        elif counts[small_val].sum() <= min_frac:
            # Dropping rows if there is not enough data to group together
            df_new.drop(
                df_new.loc[df[col].isin(small_val), col].index, \
                inplace=True)
        else:
            df_new.loc[df_new[col].isin(small_val), col] = 'other'
    
    return df_new