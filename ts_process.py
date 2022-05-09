from sklearn.model_selection._split import TimeSeriesSplit,_BaseKFold
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import pandas as pd 

class TsTools : 
    def __init__(self) : 
        pass 
    def get_metrics(self,ytrue,yhat) : 
        from sklearn.metrics import (mean_absolute_error,
                                    mean_absolute_percentage_error,
                                    mean_squared_error)
        RMSE_ = np.sqrt(mean_squared_error(ytrue,yhat))
        MAE_ = mean_absolute_error(ytrue,yhat)
        MAPE_= mean_absolute_percentage_error(ytrue,yhat)
        
        print(f'RMSE :{RMSE_}, MAE : {MAE_}, MAPE :{MAPE_} ')
        return {'RMSE' : RMSE_,
            'MAE' :MAE_, 
           'MAPE' : MAPE_}
    def transform_data(self,data,ts,window,diff_num)->pd.DataFrame :
        """
            Function to provide various TS transformation to make it stationary (classical) 
            Args : 
            data : pd.DataFrame 
            ts  : column name that contain TS data
        """
        data['diff'] = data[ts].diff(diff_num).fillna(0)
        data['ts_log'] = data[ts].apply(lambda x: np.log(x))
        
        #box cox transformation 
        from scipy.stats import boxcox 
        xbox_cox,_ = boxcox(data[ts])
        data['box_cox_transform'] = xbox_cox
        #rolling logarithm transformation with 12 month rolling mean 
        data['ts_log_moving_avg'] = data['ts_log'].rolling(window = window,
                                                                    center = False).mean()

        # moving avg with 12 month windows
        data['ts_moving_avg'] = data[ts].rolling(window = window,
                                                            center = False).mean()

        #differencing logarithm transformed value 
        data['ts_log_diff'] = data['ts_log'].diff()

        #differencing ts with its average 
        data['ts_moving_avg_diff'] = data[ts] - data['ts_moving_avg']

        # differencing log with its log moving average 
        data['ts_log_moving_avg_diff'] = data['ts_log'] - data['ts_log_moving_avg']


    


        data['ts_log_ewma'] = data['ts_log'].ewm(halflife = window,ignore_na = False, min_periods = 0,adjust = True).mean()




        #differencing EMWA with its log 
        data['ts_log_ewma_diff'] = data['ts_log'] - data['ts_log_ewma']
        #root square 
        data['sqrt_ts'] = np.sqrt(data[ts])
        
        #rolling sqrt 
        data['moving_avg_sqrt'] = data['sqrt_ts'].rolling(window = window,
                                                                    center = False).mean()
        data['diff_sqrt_moving_avg'] = data['sqrt_ts']-data['moving_avg_sqrt']
        data = data.dropna()
        return data
    def stationary_test(data,ts,alpha_threshold=0.05) : 
        """
        Function to conduct Stationary test 
        Args : 
        data : pd.DataFrame 
        ts   : column name from data that contain Time Series Feature 
        alpha_threshold : confidence interval 
        """
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(data[ts])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        if result[1] > alpha_threshold : 
            print('The Data Is Non Stationary') 
        else : 
            print('The Data Is Stationary')
    def plot_seasonal_decompose(self,result, title="Seasonal Decomposition"):
        from plotly.subplots import make_subplots 
        import plotly.graph_objects as go 
        return (
            make_subplots(
                rows=4,
                cols=1,
                subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
            )
            .add_trace(
                go.Scatter(x=result.seasonal.index, y=result.observed, mode="lines"),
                row=1,
                col=1,
            )
            .add_trace(
                go.Scatter(x=result.trend.index, y=result.trend, mode="lines"),
                row=2,
                col=1,
            )
            .add_trace(
                go.Scatter(x=result.seasonal.index, y=result.seasonal, mode="lines"),
                row=3,
                col=1,
            )
            .add_trace(
                go.Scatter(x=result.resid.index, y=result.resid, mode="lines"),
                row=4,
                col=1,
            )
            .update_layout(
                height=900, title=title, margin=dict(t=100), title_x=0.5, showlegend=False
            ,template='ggplot2')
        )
    #implementing cross validation 


class WindowedTestTimeSeriesSplit(TimeSeriesSplit):
    """
    parameters
    ----------
    n_test_folds: int
        number of folds to be used as testing at each iteration.
        by default, 1.
    """
    @_deprecate_positional_args
    def __init__(self, n_splits=5, *, max_train_size=None, n_test_folds=1):
        super().__init__(n_splits, 
                         max_train_size=max_train_size)
        self.n_test_folds=n_test_folds

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + self.n_test_folds
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        indices = np.arange(n_samples)
        fold_size = (n_samples // n_folds)
        test_size = fold_size * self.n_test_folds # test window
        test_starts = range(fold_size + n_samples % n_folds,
                            n_samples-test_size+1, fold_size) # splits based on fold_size instead of test_size
        for test_start in test_starts:
            if self.max_train_size and self.max_train_size < test_start:
                yield (indices[test_start - self.max_train_size:test_start],
                       indices[test_start:test_start + test_size])
            else:
                yield (indices[:test_start],
                       indices[test_start:test_start + test_size])



    
