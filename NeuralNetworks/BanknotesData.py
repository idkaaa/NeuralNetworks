#
# This file contains helper methods to construct the data for the Banknotes Data
#
#
# 
from pandas import read_table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

#
# This method downloads the data from the web
#
def download_data():
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''

    # If your data is in an Excel file, install 'xlrd' and use
    # pandas.read_excel instead of read_table
    #from pandas import read_excel
    #frame = read_excel(URL)

    # If your data is in a private Azure blob, install 'azure-storage' and use
    # BlockBlobService.get_blob_to_path() with read_table() or read_excel()
    #from azure.storage.blob import BlockBlobService
    #service = BlockBlobService(ACCOUNT_NAME, ACCOUNT_KEY)
    #service.get_blob_to_path(container_name, blob_name, 'my_data.csv')
    #frame = read_table('my_data.csv', ...

    frame = read_table(
        URL,
        
        # Uncomment if the file needs to be decompressed
        #compression='gzip',
        #compression='bz2',

        # Specify the file encoding
        # Latin-1 is common for data from US sources
        encoding='latin-1',
        #encoding='utf-8',  # UTF-8 is also common

        # Specify the separator in the data
        sep=',',            # comma separated values
        #sep='\t',          # tab separated values
        #sep=' ',           # space separated values

        # Ignore spaces after the separator
        skipinitialspace=True,

        # Generate row labels from each row number
        index_col=None,
        #index_col=0,       # use the first column as row labels
        #index_col=-1,      # use the last column as row labels

        # Generate column headers row from each column number
        header=None,
        #header=0,          # use the first line as headers

        # Use manual headers and skip the first row in the file
        #header=0,
        #names=['col1', 'col2', ...],
    )

    # Return a subset of the columns
    #return frame[['col1', 'col4', ...]]

    # Return the entire frame
    return frame


#
# This method formats the data for a specific dataset. 
# It splits the data into the data value array and classification target array
# Also, this method scales or normalizes data if necessary.
#
def p_FormatData(frame):
    data = frame.values[:,:-1]
    target = frame.values[:,-1:]

     # Normalize the attribute values to mean=0 and variance=1
    scaler = StandardScaler()

    scaler.fit(data)
    data = scaler.transform(data)

    return data, target



#
# This method does the actual splitting of the data based on a ratio and a 
# Random seed to guarantee the data gets split the same way during multiple tests.
# 
def p_SplitData(TrainData, TargetData):

    return train_test_split(TrainData, TargetData, test_size=0.33, random_state=4)


