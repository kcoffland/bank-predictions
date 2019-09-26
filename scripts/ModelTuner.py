from sklearn.model_selection import GridSearchCV, StratifiedKFold, \
    RandomizedSearchCV
class ModelTunerCV:

    def __init__(self, model, scorer, cv=3):
        """
        Args:
            model: sklearn model, Model the user wishes to tune
            scorer: string or valid scorer, The scoring method to be used for 
                finding the optimal hyperparameters. For more information on what
                makes a valid scorer, see 
                https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            cv: int, number of folds in cross validation
        """
        self.model = model
        self.scorer = scorer
        self.cv = cv
        # Setting the defaults for values the user will want to see and access
        self.best_params_ = {}
        self.best_score_ = 0.0
    
    def tune(self, X, y, param_grid, method='grid', n_iter=10):
        """Tunes the model to the supplied data using the given method and parameters
           The model is tuned utilizing cross validation rather than train-test
           splitting
        Args:
            X: Dataframe, Data for the model to be tuned with
            y: Series, Target values for supervised learning models
            param_grid: dict, keys are hyperparameters and values are the values 
                that the user wants to compare. Note that they need to be valid
                for the model being used as well as the method to tune the model.
                For example, RandomizedCV requires distributions to randomly 
                choose values from.
            method: string, method to perform the tuning
            n_iter: int, number of trials performed by random cv, only matters 
                when method is 'random'
        
        Returns:
            Nothing, but self.best_params_ and self.best_score_ are set by this 
            method. self.best_params_ is a dict that holds which values were the 
            best for a given hyperparameter. self.best_score_ is the best average
            score from the cross validation performed.
        """
        method_options = ['grid', 'random']
        if method not in method_options:
            msg = f"method must be one of {method_options} "\
                f"{method} was given"
            raise ValueError(msg)

        if method == method_options[0]:
            self.best_params_, self.best_score_ = self.tune_grid(X, y, param_grid)
        elif method == method_options[1]:
            self.best_params_, self.best_score_ = self.tune_random(X, y, \
                param_grid, n_iter)
    
    def tune_grid(self, X, y, param_grid):
        """Tunes the model to the supplied data using grid search
        Args:
            X: Dataframe, Data for the model to be tuned with
            y: Series, Target values for supervised learning models
            param_grid: dict, keys are hyperparameters and values are the values 
                that the user wants to compare
        
        Returns:
            best_params: dict, key is hyperparameter value is the value for the 
                hyperparameter which performed best
            best_score: float, The best average score achieved from the given
                hyperparameter grid
        """
        # Ensuring classes are balanced in the kfold cross-validation
        kfold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=44)
        try:
            grid = GridSearchCV(estimator=self.model,
                param_grid=param_grid,
                scoring=self.scorer,
                cv=kfold,
                iid=False,
                n_jobs=-1)
        except ValueError as e:
            msg = e + f" The given values were {grid.get_params()}"
            raise ValueError(msg)
        grid.fit(X, y)
        best_params, best_score = grid.best_params_, grid.best_score_
        return best_params, best_score

    def tune_random(self, X, y, param_dist, n_iter):
        """Tunes the model to the supplied data using randomized search
        Args:
            X: Dataframe, Data for the model to be tuned with
            y: Series, Target values for supervised learning models
            param_dist: dict, keys are hyperparameters and values are the 
                distributions the user wants to randomly choose values from
            n_iter: int, number of trials performed in the cross-validation
        
        Returns:
            best_params: dict, key is hyperparameter value is the value for the 
                hyperparameter which performed best
            best_score: float, The best average score achieved from the given
                hyperparameter distributions
        """
        # Ensuring classes are balanced in the kfold cross-validation
        kfold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=44)
        try:
            random = RandomizedSearchCV(estimator=self.model, 
                param_distributions=param_dist, 
                random_state=44, 
                iid=True, 
                cv=kfold, 
                n_jobs=-1,
                n_iter=n_iter,
                scoring=self.scorer
            )
        except ValueError as e:
            msg = e + f" The given values were {random.get_params()}"
            raise ValueError(msg)
        random.fit(X, y)
        best_params, best_score = random.best_params_, random.best_score_
        return best_params, best_score
        