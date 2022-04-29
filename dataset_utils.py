
import pandas as pd 

def downcast_dtypes(series:pd.Series,
                    enable_categorical: bool=True)->pd.Series : 
    #check if sparse 
    """
    Downcasting dtypes of a column / series from dataframe one by one
    by checking possible datatypes

    Args:
        series (pd.Series): series / data[col]
        enable_categorical (bool, optional): if True -> represent objects to categoircal otherwise -> encoding by factorize
        . Defaults to True.

    Returns:
        pd.Series: a converted-dtype series 
    """
    if pd.api.types.is_object_dtype(series) :   #check if object 
        if enable_categorical : # if convert to category True -> astype('category')
            series = series.astype('category')
            return series 
        else : 
            encoded,unique_val = series.factorize() 
            new_series = pd.Series(data=encoded,index=series.index)# if convert to category False -> series.factorize()
            return new_series
    elif pd.api.types.is_integer_dtype(series) : 
        series = pd.to_numeric(series,downcast='float')
        return series 
    #for now there is no menu for datetime data
     
        
            
             

def optimize_dataframe(df : pd.DataFrame | pd.Series) : 
    """
    Convert any type of each column dataframe : 
    according to pandas hint : 
    1. Loading columns that we need : e.g. pd.read_csv('path')[selected_cols]
    2. Use Efficient Data Types (We Are Here)
       a. Text ->convert into categorical (if has low cardinality)
       b. convert numeric into float32
    3. Use Chunking -> in read_csv() we can use chunking
    4. use another library : example like DASK -> implement lazy computing 
    """
    
    #step check if the dimension is 1 -> if 1 we can directly downcast
    starting_memory = np.sum(df.memory_usage()) / 1024 ** 2 
    print(f'The Data Took Memory Size = {starting_memory} MB ')
    if df.ndim == 1 : 
        df = downcast_dtypes(df)
        return df
    else : 
        for col in df.columns : 
            df[col] = downcast_dtypes(df[col])
            return df 
    after_optimization = np.sum(df.memory_usage()) / 1024 ** 2 
    print(f'After Optimization the data took Memory Size = {after_optimization} MB')
    
    
    
    