import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder,\
    LabelBinarizer, PolynomialFeatures


class Preprocessing:
    """
    Processes the data and ensures that the test data cannot learn
    more information that what the training data will give (e.g. new
    columns will be filled with 0s to give no additional information).
    """

    def __init__(self, cols_to_filter=None, ordinal_cols=None, binned_cols=None,
                 inplace=False, classification=False, polynomial_features=False):
        """
        Args:
            cols_to_filter: list of strings, names columns which need
                            to be removed from the dataset
            ordinal_cols: dict {string: list of strings or list of ints}, the key is the ordinal
                            column and the value is the list of categories in
                            order of importance. ORDER OF IMPORTANCE WILL MATTER
            binned_cols: dict {string: int}, the key is the column which
                            the user wants to be binned. The int will be the number of
                            bins that will be created.The data will be binned into equal
                            quantiles using pd.qcut(). The new column will be added to
                            the dataframe as 'given_binned' where "given" is the key of
                            the dict. If the user wants the original column removed, then
                            they can add it to the columns to filter. Note: some engineered
                            features require the binning of certain columns and will
                            not work if they have not been binned.
            inplace: bool, Whether the transformations to the data are made in place or on
                           a copy of the given data
            
        """

        # stores whether the model preprocessor has been fit to the data
        self.is_fit = False

        if cols_to_filter:
            assert(type(cols_to_filter) == list)

        self.cols_to_filter = cols_to_filter

        self.categorical_cols = pd.Index([])
        self.one_hot_encoder = OneHotEncoder(sparse=False)

        self.numerical_cols = pd.Index([])

        # Setting up the binarizer if it is a classification problem
        self.classification = classification
        if self.classification:
            self.label_binarizer = LabelBinarizer()

        # Setting up ordinal columns encoding
        self.ordinal_cols = ordinal_cols
        if self.ordinal_cols:
            assert(type(ordinal_cols) == dict)
            categories = list(ordinal_cols.values())
            self.ordinal_enc = OrdinalEncoder(categories=categories)
        else:
            self.ordinal_enc = None

        # scaler to make numerical columns between 0 and 1
        self.min_max_scaler = MinMaxScaler()

        # Setting up the polynomial features
        self.polynomial_features = polynomial_features
        if self.polynomial_features:
            self.poly_creater = PolynomialFeatures()

        # Setting up binned columns
        self.binned_cols = binned_cols
        if self.binned_cols:
            self.bins = [[] for i in range(len(self.binned_cols))]

        # Giving the user the option to transform data inplace
        self.inplace = inplace

    def fit(self, X, y=None):
        """
        Finds out what which columns need to be converted to dummy
        values and stores it as an attribute to be used later in
        the transform method.
        """
        self.is_fit = True

        # Creating the bins from the training data to cut on later
        if self.binned_cols:
            for i, col in enumerate(self.binned_cols.keys()):
                assert (col in X.columns)
                column, self.bins[i] = pd.qcut(X[col], self.binned_cols[col],
                                            labels=False, retbins=True)
                # I only needed to save the bins
                del column
        
        # Saving the categorical columns of the data
        self.categorical_cols = pd.Index([col for col in X.columns
                                          if X[col].dtype == object
                                          ]
                                         )
        if self.ordinal_cols:
            self.numerical_cols = pd.Index([col for col in X.columns \
                                        if (col not in self.categorical_cols) \
                                            and (col not in self.ordinal_cols.keys())
                                        ]
                                       )
        else:
            self.numerical_cols = pd.Index([col for col in X.columns \
                                        if col not in self.categorical_cols
                                        ]
                                       )

        # Removing filtered columns from established numeric and categorical columns
        if self.cols_to_filter:
            for col in self.cols_to_filter:
                if col in self.categorical_cols:
                    self.categorical_cols = self.categorical_cols.drop([col])
                elif col in self.numerical_cols:
                    self.numerical_cols = self.numerical_cols.drop([col])

        # Finding all columns that are dummied, note this is added after columns
        # are filtered out of self.categorical_cols
        if not self.categorical_cols.empty:
            # Ensuring ordinal cols aren't dummied
            if self.ordinal_cols:
                nominal = set(self.categorical_cols) - set(self.ordinal_cols.keys())
            else:
                nominal = set(self.categorical_cols)
            self.one_hot_encoder.fit(X.loc[:, nominal])
            self.categorical_cols = pd.Index(nominal)

        # Creating ordinal columns
        if self.ordinal_cols:
            for col in self.ordinal_cols.keys():
                assert(col in X.columns)
            self.ordinal_enc.fit(X[list(self.ordinal_cols.keys())])

        # Fitting the scaler to the training data's numerical columns
        if not self.numerical_cols.empty:
            # Creating the polynomial features of the test set so that the 
            if self.polynomial_features:
                temp_data = self.poly_creater.fit_transform(X[self.numerical_cols].copy())
                self.min_max_scaler.fit(temp_data)
                del temp_data
            else:
                # If no polynomial features are created, then the numerical 
                # columns of the original data can be used
                self.min_max_scaler.fit(X[self.numerical_cols])

        if  type(y) != None and self.classification:
            self.label_binarizer.fit(y)
        
        return self

    def transform(self, X, y=None):
        # Ensuring data has been fit before this method can be used
        if not self.is_fit:
            raise RuntimeError("Fit method must be called before data can be transformed")

        # Maybe this naming convention is why software engineers hate Data 
        # Scientists. Maybe it's the stupid comments....
        if self.inplace:
            X_new = X
        else:
            X_new = X.copy()

        # Setting all values for columns not in the training data to 0
        # so no additional information can be learned outside of engineering
        # new features available in the training data
        if self.binned_cols:
            new_cols = set(X_new.columns) - set(self.categorical_cols) - \
                       set(self.numerical_cols) - set(self.binned_cols.keys())
        else:
            new_cols = set(X_new.columns) - set(self.categorical_cols) - \
                       set(self.numerical_cols)
        for col in new_cols:
            X_new[col] = 0

        ################### Transforming Data ########################
        # Encoding Ordinal columns as well as ensuring they're scaled properly
        if self.ordinal_cols:
            X_new[list(self.ordinal_cols.keys())] = self.ordinal_enc.transform(
                                                        X_new[list(self.ordinal_cols.keys())]
                                                    )
            X_new[list(self.ordinal_cols.keys())] /= X_new[list(self.ordinal_cols.keys())].max()

        # OneHotEncoding the Categorical columns
        if not self.categorical_cols.empty:
            # Transforming the data
            ohe_data = self.one_hot_encoder.transform(
                X_new.loc[:, self.categorical_cols])
            ohe_columns = self.one_hot_encoder.get_feature_names(self.categorical_cols)
            # Creating a new dataframe of the transformed data and new columns
            ohe_df = pd.DataFrame(ohe_data, columns=ohe_columns, index=X_new.index)
            # Removing not transformed columns
            X_new.drop(self.categorical_cols, inplace=True, axis=1)
            # Merging the transformed data back into the original dataframe
            X_new = X_new.join(ohe_df)
            # Cleaning up stored data
            del ohe_data
            del ohe_columns
            del ohe_df


        # Scale the numerical columns
        if not self.numerical_cols.empty:
            if self.polynomial_features:
                # Transforming all the numerical columns
                poly_data = self.poly_creater.transform(X_new[self.numerical_cols])
                poly_cols = self.poly_creater.get_feature_names(self.numerical_cols)
                # Creating new dataframe
                poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=X_new.index)
                # Removing old numerical columns
                X_new.drop(self.numerical_cols, inplace=True, axis=1)
                # Merging the transformed data
                X_new = X_new.join(poly_df)
                # Scaling all the numerical columns to between 0 and 1 for the data
                # the scaler was fit to on line 
                X_new[poly_cols] = self.min_max_scaler.transform(
                    X_new[poly_cols])
                # Cleaning up stored data
                del poly_data
                del poly_cols
                del poly_df
            else:
                X_new[self.numerical_cols] = self.min_max_scaler.transform(
                    X_new[self.numerical_cols])
        
        ################ Creating/Removing Columns ##################
        # Putting the removal of columns at the very end of this function allows
        # for columns that are created by ohe and poly_features to be dropped 
        # automatically instead of manually later

        # Binning appropriate columns
        if self.binned_cols:
            for i, col in enumerate(self.binned_cols.keys()):
                # I cast to int so that I don't have to worry about the Category dtype
                # If I don't include_lowest, then job postings with 0 experience
                # will not be included in the bin
                X_new[f'{col}_binned'] = pd.cut(X_new[col], self.bins[i],
                                                labels=False, include_lowest=True)\
                                                .astype(int).copy()
                X_new[f'{col}_binned'] /= X_new[f'{col}_binned'].max()

        # Removing columns that the user wants to filter
        if self.cols_to_filter:
            for col in self.cols_to_filter:
                assert(col in X_new.columns and f"Given column {col} not in data")

            X_new.drop(self.cols_to_filter, inplace=True, axis=1)
        
        # If there is a y column given to preprocess
        if type(y) != None and self.classification:
            y_new = self.label_binarizer.transform(y.copy())
            return X_new, pd.Series(y_new.ravel(), name=y.name)

        return X_new

    def fit_transform(self, X, y=None):

        return self.fit(X, y=y).transform(X, y=y)



