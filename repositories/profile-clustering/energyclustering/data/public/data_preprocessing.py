from pathlib import Path
import numpy as np
import pandas as pd
import wget

from energyclustering.data.public import data

# directory that contains all the data
DATA_DIR = Path(__file__).parent / 'data'

def preprocess_raw_measurements():
    """
        Load the raw measurement table (READING_2016.CSV) as a dataframe, add the following columns without modifying the original data:
            - timestamp (just a datetime object instead of a string)
            - offtake (the offtake of this measurement or NaN if it is not an offtake measurement)
            - injection (the injection of this measurement or NaN if it is not an injection measurement)
        and save the result to the file READING_2016_full.pkl

        This is just the first step
        This function should be followed by preprocess_measurement_table

        preprocessing steps done:
        - drop duplicate readings for the same meter/timestamp
    """
    file_name_reading_zip = DATA_DIR / 'raw'/"READING_2016.CSV"
    print(f"Reading raw data from: {file_name_reading_zip}...", sep = "")
    data_reading_full = pd.read_csv(file_name_reading_zip, sep=';', parse_dates=[3], dtype={'Meetwaarde': np.float64},
                                    decimal=',')
    print(" DONE ")
    print("preprocessing table...", end="")
    # convert strings to datetime format
    data_reading_full['timestamp'] = pd.to_datetime(data_reading_full['Meter read tijdstip'],
                                                                format="%d%b%y:%H:%M:%S")

    # drop duplicate measurements for the same installation, time and measurement type (injection or offtake)
    data_reading_full.drop_duplicates(subset = ['InstallatieID', 'timestamp', 'Afname/Injectie'], keep = 'first', inplace = True)

    # Add a signed readings column so that the readings are negative if energy is injected to the system
    is_nonzero_production = (data_reading_full['Afname/Injectie'] == 'Injectie') & (data_reading_full['Meetwaarde'] > 0)
    data_reading_full['signed_consumption'] = data_reading_full['Meetwaarde']
    data_reading_full.loc[is_nonzero_production, ['signed_consumption']] *= -1
    # Same result but much slower:
    # data_reading_full['signed_consumption'] = data_reading_full.apply(
    #     lambda o: -o['Meetwaarde'] if o['Meetwaarde'] > 0 and o['Afname/Injectie'] == 'Injectie' else o['Meetwaarde'],
    #     axis=1)
    
    # Add Injection and Offtake columns
    is_offtake = data_reading_full['Afname/Injectie'] == 'Afname'
    is_injection = data_reading_full['Afname/Injectie'] == 'Injectie'
    data_reading_full['Offtake'] = data_reading_full['Meetwaarde'].where(is_offtake, np.nan)
    data_reading_full['Injection'] = data_reading_full['Meetwaarde'].where(is_injection, np.nan)

    # Same result but much slower:
    # data_reading_full['consumption'] = data_reading_full.apply(
    #     lambda o: o['Meetwaarde'] if o['Afname/Injectie'] == 'Injectie' else np.nan,
    #     axis=1)
    # data_reading_full['production'] = data_reading_full.apply(
    #     lambda o: o['Meetwaarde'] if o['Afname/Injectie'] == 'Afname' else np.nan,
    #     axis=1)
    print(" DONE ")
    
    data_reading_full.to_pickle(DATA_DIR/ "READING_2016_full.pkl", compression='gzip')

def intermediate_measurements_to_preprocessed():
    """
        Simply processes the raw measurement table to a dataframe containing:
            - the net consumption (consumption - production)
            - consumption
            - production
        for each timestamp,
        and saves the result
    """
    print("preprocessing table further...", end="")
    data_reading_full = pd.read_pickle(DATA_DIR/ "READING_2016_full.pkl", compression='gzip')

    def custom_aggregation(values):
        # if all values are NaN
        # don't create a zero value
        if np.all(np.isnan(values)):
            return np.nan
        else:
            # treats nan's as 0
            return np.nansum(values)

    # Obtain net readings for each timestamp of each installationID
    # need custom function to handle nans
    drg: pd.DataFrame= data_reading_full.groupby(['InstallatieID', 'timestamp']).agg(
        Offtake=('Offtake', custom_aggregation),
        Injection=('Injection', custom_aggregation))
    drg['Consumption'] = drg['Offtake'] - drg['Injection']
    
    # Rename columns as "iID", "datetime",  where the first two are indices (and are sorted)
    drg.index.names = ['iID', 'datetime']
    print(" DONE ")
    
    drg.to_pickle(DATA_DIR/ "READING_2016_preprocessed.pkl")

def calculate_measurement_tables_per_day(value, only_single_meters = False):
    """
        Makes a dataframe based on READING_2016_preprocessed.pkl
        The dataframe has a multi-level index (iID, data)
        The columns are the hours of the day
        The values in the table are specified by the value parameter (this is supposed to be injection, offtake or consumption)

        Note that the injection and offtake values are not the actual total consumption of the household or the total production from solar panels the two values compensate each other
        However, the total consumption under consumption is correct

        If only_single_meters = True, locations with two meters are ignored
    """
    df = pd.read_pickle(DATA_DIR/ "READING_2016_preprocessed.pkl")
    df = df.reset_index()
    df['date'] = [d.date() for d in df['datetime']]
    df['time'] = [d.time() for d in df['datetime']]
    measures_per_day = pd.pivot_table(df, values=value, index=['iID','date'], columns=['time'])
    # if a full time series is missing drop the day
    measures_per_day.dropna(axis = 0, how = 'all', inplace=True)

    if only_single_meters:
        master_table = data.get_master_table()
        one_meter_ids = master_table[master_table['Aantal geïnstalleerde meters'] == 1].loc[:, "InstallatieID"].unique()
        measures_per_day = measures_per_day.loc[one_meter_ids]

    if only_single_meters:
        dir = DATA_DIR / "per_day_single_meters"
    else:
        dir = DATA_DIR / "per_day"

    dir.mkdir(parents = True, exist_ok=True)
    measures_per_day.to_pickle(dir/f'READING_2016_{value}_per_day.pkl')

def calculate_measurement_tables_per_week(value, only_single_meters = False):
    """
    More or less the same as calculate_measurement_tables_per_day but for weeks
    The resulting dataframe:
        - row index is a multi-level index (iID, weeknumber):
        - column index is also multi level (day_nb, time)
        - the values are again what you specify under value
    only_single_meters can be enabled to only use
    :param value:
    :param only_single_meters:
    :return:
    """
    df = pd.read_pickle(DATA_DIR / "READING_2016_preprocessed.pkl")
    df = df.reset_index()
    # week number from 0 to 52 (0 for the days before the first monday of the year)
    df['week_number'] = [int(d.strftime("%W")) for d in df['datetime']]
    # week day number from 0 - 6 (0 is monday, 6 is sunday)
    df['day_number'] = [d.weekday() for d in df['datetime']]
    df['time'] = [d.time() for d in df['datetime']]

    measures_per_week = pd.pivot_table(df, values = value, index = ['iID', 'week_number'], columns = ['day_number', 'time'])
    if only_single_meters:
        master_table = data.get_master_table()
        one_meter_ids = master_table[master_table['Aantal geïnstalleerde meters'] == 1].loc[:, "InstallatieID"].unique()
        measures_per_week = measures_per_week.loc[one_meter_ids]

    if only_single_meters:
        dir = DATA_DIR / "per_week_single_meters"
    else:
        dir = DATA_DIR / "per_week"

    dir.mkdir(parents = True, exist_ok=True)
    measures_per_week.to_pickle(dir/f'READING_2016_{value}_per_week.pkl')

def download_data():
    master_table_url = "https://www.fluvius.be/sites/fluvius/files/2019-02/master-table-meters.csv"
    reading_table_url = "https://www.fluvius.be/sites/fluvius/files/2019-02/READING_2016.zip"
    (DATA_DIR/ 'raw').mkdir(exist_ok=True, parents = True)
    print('\tDownloading data')
    wget.download(master_table_url, str(DATA_DIR/'raw'))
    wget.download(reading_table_url, str(DATA_DIR/'raw'))
    print("\tExtracting READING_2016.zip")
    import zipfile
    with zipfile.ZipFile(DATA_DIR/'raw'/'READING_2016.zip', 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR/'raw')
    print("\tRemoving READING_2016.zip")
    (Path(DATA_DIR)/'raw'/'READING_2016.zip').unlink()

def prepare_data_from_scratch(directory = DATA_DIR):
    print("Checking/preparing data directories")
    DATA_DIR = Path(directory)
    if not (DATA_DIR/'raw'/'READING_2016.CSV').exists():
        raise Exception("Could not find READING_2016.CSV in the data/raw directory \n please download manually from https://www.fluvius.be/sites/fluvius/files/2019-02/READING_2016.zip and unzip or run the get_data.sh script from the data directory")
    if not (DATA_DIR/'raw'/'master-table-meters.csv').exists():
        raise Exception("Could not find master-table-meters.csv in the data/raw directory \n please download manually from https://www.fluvius.be/sites/fluvius/files/2019-02/master-table-meters.csv")
    if not (DATA_DIR/ "READING_2016_full.pkl").exists():
        preprocess_raw_measurements()
    if not (DATA_DIR/ "READING_2016_preprocessed.pkl").exists():
        intermediate_measurements_to_preprocessed()

    for value in ['Consumption', "Offtake", "Injection"]:
        if not (DATA_DIR/ 'per_week' / f"READING_2016_{value}_per_week.pkl").exists():
            calculate_measurement_tables_per_week(value)
        if not (DATA_DIR/ "per_week_single_meters" / f"READING_2016_{value}_per_week.pkl").exists():
            calculate_measurement_tables_per_week(value,True)
        if not (DATA_DIR/ "per_day" / f"READING_2016_{value}_per_day.pkl").exists():
            calculate_measurement_tables_per_day(value)
        if not (DATA_DIR/ "per_day_single_meters" / f"READING_2016_{value}_per_day.pkl").exists():
            calculate_measurement_tables_per_day(value,True)

if __name__ == '__main__':
    prepare_data_from_scratch()

