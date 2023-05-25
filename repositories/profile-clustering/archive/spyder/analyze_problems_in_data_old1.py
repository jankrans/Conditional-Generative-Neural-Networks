#%% imports
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import itertools
#%%
DATA_DIR = Path().absolute()/ 'data'
data_reading_full = pd.read_pickle(DATA_DIR/"READING_2016_full.pkl", compression='gzip')

def get_master_table():
    file_name_master = DATA_DIR / 'master_table_meters_csv.csv'
    data_master = pd.read_csv(file_name_master, sep=';', usecols=range(9), dtype={'Meetwaarde':np.float64,
                                                                                  'Afname - Injectie':np.float64,
                                                                                  'Jaarlijkse injectie':np.float64,
                                                                                  'Jaarlijkse afname':np.float64,
                                                                                  'Geïnstalleerd vermogen':np.float64}, 
                              decimal=',', engine='python')
    return data_master

master_table = get_master_table()

# %% Find duplicate readings: 
# for the same InstallationID and meter, there are multiple values at the same timestamp !
# (only the first one is taken in preprocess_measurement_table() in data_preprocessing.py)

try:
    # Use the old version of the table where duplicate readings remain
    data_reading_full_old = pd.read_pickle(DATA_DIR/"READING_2016_full_old.pkl", compression='gzip')

    dpl = data_reading_full_old.duplicated(['InstallatieID', 'Afname/Injectie', 'timestamp'], keep=False)
    dpl_first = data_reading_full_old.duplicated(['InstallatieID', 'Afname/Injectie', 'timestamp'], keep='first')
    dpl_ind = np.where(dpl)[0] # first indices of duplicate readings
    dpl_first_ind = np.where(dpl_first)[0] # all indices of duplicate readings
    
    print('\nOut of {} readings, there are {} unique and {} repeated ones!!!'.format(len(data_reading_full_old),
                                                                                   len(data_reading_full_old) - len(dpl_first_ind), 
                                                                                   len(dpl_first_ind)))
    
    data_duplicated = data_reading_full_old[dpl]
    data_duplicated_sorted = data_duplicated.sort_values(['InstallatieID', 'timestamp', 'Afname/Injectie'])
    
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 100)
    print('\nDuplicate readings:')
    print(data_duplicated_sorted)

except:
    print("\nCouldn't find the old data file 'READING_2016_full_old.pkl' so cannot compute duplicate readings !")

# %% Check total offtake, injection, consumption in the data:
df = pd.read_pickle(DATA_DIR/ "READING_2016_preprocessed.pkl")

totals = df.groupby('iID').sum()
totals_extended = pd.merge(totals, master_table, how='left', left_on='iID', right_on='InstallatieID')

no_IDs = len(totals_extended)
iIDs_to_label = range(0, no_IDs, 10)
iIDs_labels = list("#"+str(o+1) for o in iIDs_to_label)

# Plot the values calculated from time series and the ones available in the master table:
plt.figure()
plt.subplot(2, 1, 1)
totals_extended.plot(x="InstallatieID", y=["Offtake", "Jaarlijkse afname"], kind="bar", ax=plt.gca())
plt.xticks(ticks=iIDs_to_label, labels=iIDs_labels, rotation=0)
plt.legend(["calculated from time series", "available in master table"])
plt.ylabel("total offtake\n(kWh)")
plt.xlabel("").set_visible(False)
plt.subplot(2, 1, 2)
totals_extended.plot(x="InstallatieID", y=["Injection", "Jaarlijkse injectie"], kind="bar", ax=plt.gca())
plt.xticks(ticks=iIDs_to_label, labels=iIDs_labels, rotation=0)
# plt.legend(["calculated from time series", "available in master table"])
plt.legend().set_visible(False)
plt.ylabel("total injection\n(kWh)")
plt.xlabel("installationID")
# plt.subplot(3, 1, 3)
# totals_extended.plot(x="InstallatieID", y=["Consumption", "Afname - Injectie"], kind="bar", ax=plt.gca())
# plt.xticks(ticks=iIDs_to_label, labels=iIDs_labels, rotation=0)
# plt.legend(["calculated from time series", "available in master table"])
# plt.ylabel("consumption\n(kWh)")
# plt.xlabel("installationID")

# Plot the difference between the values as percentage:
diffperc_offtake = np.divide(totals_extended['Jaarlijkse afname'] - totals_extended['Offtake'], totals_extended['Jaarlijkse afname'])*100
diffperc_injection = np.divide(totals_extended['Jaarlijkse injectie'] - totals_extended['Injection'], totals_extended['Jaarlijkse injectie'])*100
diffperc_consumption = np.divide(totals_extended['Afname - Injectie'] - totals_extended['Consumption'], totals_extended['Afname - Injectie'])*100
plt.figure()
plt.subplot(2, 1, 1)
plt.bar(list(range(len(diffperc_offtake))), diffperc_offtake)
plt.xlim((0,no_IDs))
plt.xticks(ticks=iIDs_to_label, labels=iIDs_labels)
plt.ylabel("total offtake\nerror (%)")
plt.xlabel("").set_visible(False)
plt.subplot(2, 1, 2)
plt.bar(list(range(len(diffperc_injection))), diffperc_injection)
plt.xlim((0,no_IDs))
plt.xticks(ticks=iIDs_to_label, labels=iIDs_labels)
plt.ylabel("total injection\nerror (%)")
plt.xlabel("installationID")
# plt.subplot(3, 1, 3)
# plt.bar(list(range(len(diffperc_consumption))), diffperc_consumption)
# plt.xlim((0,no_IDs))
# plt.xticks(ticks=iIDs_to_label, labels=iIDs_labels)
# plt.ylabel("consumption")
# plt.xlabel("installationID")

# %% Meter types and night meters:
# Three types of meters:
totals_extended.loc[totals_extended['Type register'].str.startswith('THNUTHNU'), 'meter_type'] = 'onerate'
totals_extended.loc[totals_extended['Type register'].str.startswith('HILOHILO'), 'meter_type'] = 'tworate'
totals_extended.loc[totals_extended['Type register'].str.startswith('EXNUEXNU'), 'meter_type'] = 'exclnight'

totals_meter_type = totals_extended.groupby('meter_type').sum()
print("\nStats for meter types:")
print(totals_meter_type.iloc[:,:3])

# totals_metertype = totals_extended.groupby('Type register').sum()
# total_onerate = totals_metertype.iloc[totals_metertype.index.str.startswith('THNUTHNU')].sum()
# total_tworate = totals_metertype.iloc[totals_metertype.index.str.startswith('HILOHILO')].sum()
# total_exclnight = totals_metertype.iloc[totals_metertype.index.str.startswith('EXNUEXNU')].sum()

# Night meter is defined as the meters except 'exclnight' in houses with multiple meters
mask_nightonly = (totals_extended['meter_type']!='exclnight') & (totals_extended['Aantal geïnstalleerde meters']==2)
totals_nightonly = totals_extended.loc[mask_nightonly].sum()
print("\nNight meters only:")
print(totals_nightonly.iloc[:3])

# %% Missing values:
# Here, for each installationID and timestamp, the sample is considered as a missing value if there are no readings at all (neither offtake nor injection)

# time_first, time_last = data_reading_full.timestamp.sort_values().iloc[[0, -1]].values
time_first, time_last = df.index.get_level_values(1).sort_values()[[0, -1]] #same result but faster

time_indices_full = pd.date_range(start=time_first, end=time_last, freq="15min")

# iIDs = data_reading_full.InstallatieID.unique()
iIDs = df.index.get_level_values(0).unique() #same result

times_missing = [] #missing datetime values for each installationID
times_missing_grouped = [] #missing datetime values, with successive ones grouped, for each installationID
for iID in iIDs:
    # tm = time_indices_full.difference(data_reading_full.loc[data_reading_full['InstallatieID']==iID]['timestamp']) #missing time indices
    tm = time_indices_full.difference(df.xs(iID, level='iID').index) #missing time indices - same result as the previous line
    times_missing.append(tm)
    tmg = np.split(tm, np.where(np.diff(tm)!=np.timedelta64(15, 'm'))[0]+1) #missing time indices with successive ones grouped
    times_missing_grouped.append(tmg)
times_missing_grouped_ = list(itertools.chain(*times_missing_grouped))
no_of_consecutive_missing = list(itertools.chain(*list(map(lambda oo: list(map(lambda o: len(o), oo)), times_missing_grouped)))) #number of successive missing values
missing_duration_hours = [i/4 for i in no_of_consecutive_missing] #duration of intervals with missing values
                                                                  #for a sampling period of 15 min, 4 missing values is considered as 1 hour although there are 1 hour and 15 minutes between the readings just before and just after that period

plt.figure()
plt.hist(missing_duration_hours, bins=100)
plt.axvline(np.mean(missing_duration_hours), color='g', linestyle='dashed', linewidth=1) #line for the mean value
plt.text(np.mean(missing_duration_hours), plt.ylim()[1]*0.9 + plt.ylim()[0]*0.1, " ← mean: {:.2f}".format(np.mean(missing_duration_hours)), color='g') #text for the mean value
plt.xlabel("duration (hours)")
plt.ylabel("number of periods with missing values")
plt.title("histogram")

print("\nThere are {:d} time intervals with missing values, with the average duration of {:.2f} hours.".format(len(missing_duration_hours), np.mean(missing_duration_hours)))
print("Out of {:d} samples, there are {:d} missing values, constituting {:.2f}% of them.".format(len(time_indices_full)*len(iIDs), np.sum(no_of_consecutive_missing), 100*np.sum(no_of_consecutive_missing)/(len(time_indices_full)*len(iIDs))))

plt.figure()
plt.bar(list(range(len(iIDs))), list(map(lambda o: 100*len(o)/len(time_indices_full), times_missing)))
plt.xlim((0,no_IDs))
plt.xticks(ticks=iIDs_to_label, labels=iIDs_labels)
plt.xlabel("installationID")
plt.ylabel("percentage of missing values")

plt.figure()
plt.bar(list(range(len(iIDs))), list(map(lambda oo: np.mean(list(map(lambda o: len(o)/4, oo))), times_missing_grouped)))
plt.xlim((0,no_IDs))
plt.xticks(ticks=iIDs_to_label, labels=iIDs_labels)
plt.axhline(np.mean(missing_duration_hours), color='g', linestyle='dashed', linewidth=1) #line for the mean value
plt.text(plt.xlim()[0]*0.95+plt.xlim()[1]*0.05, np.mean(missing_duration_hours), " ← mean: {:.2f}".format(np.mean(missing_duration_hours)), horizontalalignment='center', color='g', rotation=90)
plt.xlabel("installationID")
plt.ylabel("average duration of missing values (hours)")

# plot all missing values:
missing_mask = np.full((len(iIDs), len(time_indices_full)), False)
for i, iID in enumerate(iIDs):
    missing_mask[i, np.where(np.isin(time_indices_full, times_missing[i]))] = True
plt.figure()
plt.imshow(missing_mask, aspect='auto', extent=(mdates.date2num(time_first), mdates.date2num(time_last), 0, len(iIDs)-1))
plt.xticks(mdates.date2num(pd.date_range(time_first, time_last, freq='MS')))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
plt.yticks(ticks=iIDs_to_label, labels=iIDs_labels, rotation=0)
plt.title('missing values (light colors)')
plt.ylabel('installationID')

# %% Zeros:

# Zero offtake:
no_of_consecutive_zeros_offtake = [] #number of consecutive zero offtake values for each installationID
for iID in iIDs:
    zr = [len(list(o)) for oo, o in itertools.groupby(df.xs(iID, level='iID').Offtake) if oo == 0] #number of consecutive zero offtake values for each zero period
    no_of_consecutive_zeros_offtake.append(zr)
no_of_consecutive_zeros_offtake_ = list(itertools.chain(*no_of_consecutive_zeros_offtake))
zero_duration_hours_offtake = [i/4 for i in no_of_consecutive_zeros_offtake_]#duration of intervals with zeros

plt.figure()
plt.hist(zero_duration_hours_offtake, bins=100, bottom=0.5)
plt.yscale('log', nonposy='clip')
plt.axvline(np.mean(zero_duration_hours_offtake), color='g', linestyle='dashed', linewidth=1) #line for the mean value
plt.text(np.mean(zero_duration_hours_offtake), plt.ylim()[1]*0.5 + plt.ylim()[0]*0.5, " ← mean: {:.2f}".format(np.mean(zero_duration_hours_offtake)), color='g') #text for the mean value
plt.xlabel("duration (hours)")
plt.ylabel("number of periods with\nzero offtake measurements")
plt.title("histogram")

print("\nOut of {:d} time samples, there are {:d} zero offtake readings, constituting {:.2f}% of them.".format(len(time_indices_full)*len(iIDs), np.sum(no_of_consecutive_zeros_offtake_), 100*np.sum(no_of_consecutive_zeros_offtake_)/(len(time_indices_full)*len(iIDs))))

# Zero offtake excluding houses with two meters:
iID_indices_singlemeter = np.in1d(iIDs, master_table[master_table['Aantal geïnstalleerde meters'] == 1].InstallatieID.values) #installationIDs where there is only one meter in a house
iIDs_singlemeter = iIDs[iID_indices_singlemeter]
no_of_consecutive_zeros_offtake_singlemeter = list(itertools.compress(no_of_consecutive_zeros_offtake, iID_indices_singlemeter))
no_of_consecutive_zeros_offtake__singlemeter = list(itertools.chain(*no_of_consecutive_zeros_offtake_singlemeter))
zero_duration_hours_offtake_singlemeter = [i/4 for i in no_of_consecutive_zeros_offtake__singlemeter] #duration of intervals with missing values

plt.figure()
plt.hist(zero_duration_hours_offtake_singlemeter, bins=100, bottom=0.5)
plt.yscale('log', nonposy='clip')
plt.axvline(np.mean(zero_duration_hours_offtake_singlemeter), color='g', linestyle='dashed', linewidth=1) #line for the mean value
plt.text(np.mean(zero_duration_hours_offtake_singlemeter), plt.ylim()[1]*0.5 + plt.ylim()[0]*0.5, " ← mean: {:.2f}".format(np.mean(zero_duration_hours_offtake_singlemeter)), color='g') #text for the mean value
plt.xlabel("duration (hours)")
plt.ylabel("number of periods with\nzero offtake measurements\n(only for houses with a single meter)")
plt.title("histogram")

print("\nOut of {:d} time samples (for the houses with a single meter), there are {:d} zero offtake readings, constituting {:.2f}% of them.".format(len(time_indices_full)*len(iIDs_singlemeter), np.sum(no_of_consecutive_zeros_offtake__singlemeter), 100*np.sum(no_of_consecutive_zeros_offtake__singlemeter)/(len(time_indices_full)*len(iIDs_singlemeter))))

# Zero injection (only for meters with injection capability):
no_of_consecutive_zeros_injection = [] #number of consecutive zero injection values for each installationID with injection capability
iIDs_with_injection = master_table[master_table['Lokale productie'] == 'Ja'].InstallatieID.values
for iID in iIDs_with_injection:
    zr = [len(list(o)) for oo, o in itertools.groupby(df.xs(iID, level='iID').Injection) if oo == 0] #number of consecutive zero injection values for each zero period
    no_of_consecutive_zeros_injection.append(zr)
no_of_consecutive_zeros_injection_ = list(itertools.chain(*no_of_consecutive_zeros_injection))
zero_duration_hours_injection = [i/4 for i in no_of_consecutive_zeros_injection_]#duration of intervals with missing values

plt.figure()
plt.hist(zero_duration_hours_injection, bins=100, bottom=0.5)
plt.yscale('log', nonposy='clip')
plt.axvline(np.mean(zero_duration_hours_injection), color='g', linestyle='dashed', linewidth=1) #line for the mean value
plt.text(np.mean(zero_duration_hours_injection), plt.ylim()[1]*0.5 + plt.ylim()[0]*0.5, " ← mean: {:.2f}".format(np.mean(zero_duration_hours_injection)), color='g') #text for the mean value
plt.xlabel("duration (hours)")
plt.ylabel("number of periods with\nzero injection measurements\n(for installationIDs with injection capability)")
plt.title("histogram")

print("\nOut of {:d} time samples (for the meters with injection capability), there are {:d} zero injection readings, constituting {:.2f}% of them.".format(len(time_indices_full)*len(iIDs_with_injection), np.sum(no_of_consecutive_zeros_injection_), 100*np.sum(no_of_consecutive_zeros_injection_)/(len(time_indices_full)*len(iIDs_with_injection))))

# %% Relation between zero intervals and the non-zero values (possibly peaks) following them:
peaks_after_zeroofftakes = [] #the non-zero values following each period with zero offtake measurements (for each installationID)
for iID in iIDs:
    ps = [list(x[1])[0] for x in itertools.groupby(df.xs(iID, level='iID').Offtake, lambda x: x == 0) if not x[0]]
    if df.xs(iID, level='iID').Offtake[0]!=0: #if it starts with non-zero
        ps = ps[1:] #remove the first one because it corresponds to the first index and there cannot be a zero-period before it
    if df.xs(iID, level='iID').Offtake[-1]==0: #if it ends with zero
        ps.append(None) #add a None for the next non-zero sample is not available
    peaks_after_zeroofftakes.append(ps)
peaks_after_zeroofftakes_ = list(itertools.chain(*peaks_after_zeroofftakes)) #drop installationID

plt.figure()
plt.scatter(zero_duration_hours_offtake, peaks_after_zeroofftakes_, marker=".", s=2)
plt.xlabel("duration of zeros (hour)")
plt.ylabel("next non-zero offtake value")

plt.figure()
plt.scatter(zero_duration_hours_offtake, peaks_after_zeroofftakes_, marker=".", s=2)
plt.xlim((0, 100))
plt.ylim((0, 2))
plt.xlabel("duration of zeros (hour)")
plt.ylabel("next non-zero offtake value")

# histogram of all offtakes and those following zeros:
bins = np.linspace(0, np.nanmax(df.Offtake.values), 500)
plt.figure()
plt.hist([x for x in peaks_after_zeroofftakes_ if not pd.isnull(x)], bins, density=True, alpha=0.5, label='offtake values that follow zeros')
plt.hist(df.Offtake.values, bins, density=True, alpha=0.5, label='all offtake values')
plt.legend()
plt.xlim((0, 10))
plt.ylim((0, 2))
plt.xlabel('offtake')
plt.ylabel('density')
plt.title('histogram')

# %% Relation between intervals with missing values and the non-zero values (possibly peaks) following them:
peaks_after_missing = [] #the non-zero values following each period with missing values (for each installationID)
for i, iID in enumerate(iIDs):
    ps_ = []
    for j in range(len(times_missing_grouped[i])): #each interval with missing values
        t = (times_missing_grouped[i][j][-1] + np.timedelta64(15, 'm')) #timestamp of the next reading
        if (iID, t) in df.index: #next timestamp exists in the data
            ps_.append(df.xs([iID, t]).Offtake)
        else:
            ps_.append(None)
    peaks_after_missing.append(ps_)
peaks_after_missing_ = list(itertools.chain(*peaks_after_missing))

plt.figure()
plt.scatter(missing_duration_hours, peaks_after_missing_, marker=".", s=2)
plt.xlabel("duration of missing values (hour)")
plt.ylabel("next non-zero offtake value")

plt.figure()
plt.scatter(missing_duration_hours, peaks_after_missing_, marker=".", s=2)
plt.xlabel("duration of missing values (hour)")
plt.ylabel("next non-zero offtake value")
plt.xlim((0, 7))
plt.ylim((0, 40))

# histogram of all offtakes and those following missing values:
bins = np.linspace(0, np.nanmax(df.Offtake.values), 500)
plt.figure()
plt.hist(peaks_after_missing_, bins, density=True, alpha=0.5, label='offtake values that follow missing intervals')
plt.hist(df.Offtake.values, bins, density=True, alpha=0.8, label='all offtake values')
plt.legend()
plt.xlim((0, 10))
plt.ylim((0, 2))
plt.xlabel('offtake')
plt.ylabel('density')
plt.title('histogram')

# %% Table of missing values, zeros, injection capabilities, number of meters for each installationID:

# zeroinds_offtake = df[df.Offtake == 0]
zeros = df.groupby(level=0).agg(lambda o: o.eq(0).sum())

problems = pd.merge(zeros, master_table, how='left', left_on='iID', right_on='InstallatieID')

problems = problems.rename(columns={'Offtake': 'no. of zeros: offtake', 
                                    'Injection':'no. of zeros: injection', 
                                    'Consumption':'no. of zeros: consumption', 
                                    'InstallatieID': 'installationID', 
                                    'Locatie_ID': 'locationID', 
                                    'Aantal geïnstalleerde meters': 'number of meters installed',
                                    'Lokale productie': 'local production', 
                                    'Jaarlijkse afname': 'annual offtake',
                                    'Jaarlijkse injectie': 'annual injection',
                                    'Afname - Injectie': 'annual consumption', 
                                    'Geïnstalleerd vermogen': 'installed capacity'})

problems['local production'] = problems['local production'].map({'Nee':'no', 'Ja':'yes'})

problems.loc[problems['Type register'].str.startswith('THNUTHNU'), 'meter type'] = 'onerate'
problems.loc[problems['Type register'].str.startswith('HILOHILO'), 'meter type'] = 'tworate'
problems.loc[problems['Type register'].str.startswith('EXNUEXNU'), 'meter type'] = 'exclnight'

problems.loc[problems['local production']=='no', 'no. of zeros: injection'] = np.nan
problems.loc[problems['number of meters installed'], 'no. of zeros: injection']

problems = problems[problems.columns[[10,3, 11,12,5,6, 7,8,9,0,1,2]]]

problems = problems.sort_values(['number of meters installed', 'local production', 'locationID', 'installationID'], ascending=[True, True, True, True])

missing = pd.DataFrame(data={'installationID':iIDs, 'no. of missing values': map(len, times_missing)})
problems = pd.merge(problems, missing, how='right', on='installationID')

# problems.to_clipboard(excel=True) #to copy-paste into Excel