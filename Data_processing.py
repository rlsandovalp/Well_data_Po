# %% Import libraries
import numpy as np
from pyproj import Transformer
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import gc, fiona, cv2, skvideo.io

# %% Classes definition
class well_data:
    def __init__(self, well_ids, well_x_coords, well_y_coords, well_wtd, observation_dates):
        self.well_ids = well_ids
        self.well_x_coords = well_x_coords
        self.well_y_coords = well_y_coords
        self.well_wtd = well_wtd
        self.observation_dates = observation_dates

    def coordinate_transformation(self, x_coord_before, y_coord_before, CS_origin, CS_target):
        transformer = Transformer.from_crs(CS_origin, CS_target)
        lon, lat = transformer.transform(x_coord_before, y_coord_before)
        self.well_x_coords = lon
        self.well_y_coords = lat

    def average_wtd(self):
        unique_wells = np.unique(self.well_ids)
        x_coords = np.zeros(len(unique_wells))
        y_coords = np.zeros(len(unique_wells))
        avg_wtd = np.zeros(len(unique_wells))
        for i, well in enumerate(unique_wells):
            avg_wtd[i] = np.mean(self.well_wtd[self.well_ids == well])
            x_coords[i] = self.well_x_coords[self.well_ids == well][0]
            y_coords[i] = self.well_y_coords[self.well_ids == well][0]
        summary_data = pd.DataFrame({'Well ID': unique_wells, 'X_54012': x_coords, 'Y_54012': y_coords, 'Average WTD': avg_wtd})
        return summary_data
    
    def data_frame(self):
        data = pd.DataFrame({'Well ID': self.well_ids, 'X_54012': self.well_x_coords, 'Y_54012': self.well_y_coords, 'WTD': self.well_wtd, 'Date': self.observation_dates})
        return data
    
    def clean_data(self):
        measurements = self.well_wtd
        new_measurements = np.copy(measurements)
        for n_measurement, measurement in enumerate(measurements):
            if isinstance(measurement, str):
                measurement = measurement.strip()  # Remove any surrounding whitespace
                # Handle strings starting with '<' or '>'
                if measurement.startswith('<') or measurement.startswith('>'):
                    remaining = measurement[1:].strip()
                    
                # Check for '-' after the sign
                    if remaining.startswith('-'):
                        new_measurements[n_measurement] = 0
                    else:
                        if ',' in remaining:
                            remaining = remaining.replace(',', '.')
                        new_measurements[n_measurement] = float(remaining)
                else:
                    # Convert plain string to float if it's a measurementid number
                    new_measurements[n_measurement] = float(measurement.replace(',', '.'))
        self.well_wtd = new_measurements
class anagrafica:
    def __init__(self, well_ids, well_x_coords, well_y_coords):
        self.well_ids = well_ids
        self.well_x_coords = well_x_coords
        self.well_y_coords = well_y_coords

    def transform_coords(self, x_coord_before, y_coord_before, CS_origin, CS_target):
        transformer = Transformer.from_crs(CS_origin, CS_target)
        lon, lat = transformer.transform(x_coord_before, y_coord_before)
        self.well_x_coords = lon
        self.well_y_coords = lat

# %% LOMBARDY
provinces_lombardy = ['Bergamo', 'Brescia', 'Como', 'Cremona', 'Lecco', 'Lodi', 'Mantova', 'Milano', 'Monza', 'Sondrio', 'Pavia', 'Varese']

all_data_lombardy = pd.DataFrame()
for province in provinces_lombardy:
    all_data_lombardy = pd.concat([all_data_lombardy, pd.read_excel(f'Lombardia/Dati quantitativi {province}_2000_2021.xlsx')])

all_data_lombardy = well_data(all_data_lombardy['CODICE PUNTO'], all_data_lombardy['X_WGS84'], all_data_lombardy['Y_WGS84'], all_data_lombardy['MISURA SOGGIACENZA [m]'], all_data_lombardy['DATA'])

del provinces_lombardy, province
gc.collect()

all_data_lombardy.coordinate_transformation(all_data_lombardy.well_x_coords, all_data_lombardy.well_y_coords, 'epsg:32632', "+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs")

average_wtd_lombardy = all_data_lombardy.average_wtd()
transient_wtd_lombardy = all_data_lombardy.data_frame()

# %% PIEDMONT
shape = fiona.open("Piemonte/wellPositions.shp")
wells_piedmont = wells_piedmont = [ '00314310001',
          '00206110001', 
          '00307310001',
          '00310010001',
          '00310810001', 
          '09600410001', 
          '09602010001',
          '00212210001', 
          '00206210001', 
          '00303010001',  
          '00303210001', 
          '00103010001', 
          '09603510001', 
          '09603110001', 
          '00202110001',
          '00308310001',
          '00100410001',
          '09601610001',
          '00203210001',
          '00205910001', 
          '00301810001', 
          '00313510001', 
          '00313110001',
          '00310610001',
          '00314910001',
          '00304910001',
          '00126910001',
          '00200410001',
          '00201110001',
          '00205210001',
          '00212610001',
          '00207010001',
          '00215810001',
          '00201710001',
          '00304110001',
          '00316410001',
          '00307710001',
          '00315810001',
          '00121710001',
          '00104710001',
          '00130110001',
          '00124810001',
          '00113010001',
          '00131410001',
          '00108210001',
          '00122510001',
          '00212810001',
          '00129310001',
          '00214810002',
          '00211810001',
          '00214810001',
          '00610910001',
          '00209110001',
          '00209310001',
          '00208210001',
          '00108610002',
          '00106310001',
          '00109910003',
          '00129210001',
          '00607310001',
          '00617810001',
          '00109010001',
          '00127210003',
          '00127210001',
          '00117110001',
          '00105110001',
          '00112710001',
          '00608710001',
          '00615110001',
          '00605310001',
          '00613210001',
          '00104110001',
          '00126010001',
          '00131010001',
          '00105910002',
          '00105910001',
          '00500310001',
          '00500510001',
          '00609110001',
          '00600310003',
          '00600310004',
          '00610510001',
          '00600310002',
          '00600310001',
          '00617410002',
          '00103510001',
          '00107010001',
          '00414310002',
          '00417910001',
          '00404110001',
          '00604710001',
          '00607510001',
          '00602110001',
          '00401210001',
          '00421710001',
          '00421510001',
          '00402910001', 
          '00605210001',
          '00601210001',
          '00611410001',
          '00425010001',
          '00408910001',
          '00401910001',
          '00403410001',
          '00422510001',
          '00408910002',
          '00407810001',
          '00414410001',
          '00401610001'
          ]
ids_wells = []
x_coords = []
y_coords = []
wtds = []
dates = []
for well in wells_piedmont:
    df_well = pd.read_csv(f'Piemonte/{well}.csv')
    n_points_well = len(df_well)
    for feature in shape:
        if feature['properties']['CODICE_PUN']==well:
            x_coord, y_coord = feature['geometry']['coordinates']
            ids_wells.append([well]*n_points_well)
            x_coords.append([x_coord]*n_points_well)
            y_coords.append([y_coord]*n_points_well)
            wtds.append(df_well['wtd'])
            dates.append(df_well['date'])

x_coords = np.concatenate(x_coords)
y_coords = np.concatenate(y_coords)
ids_wells = np.concatenate(ids_wells)
wtds = np.concatenate(wtds)
dates = np.concatenate(dates)

all_data_piedmont = well_data(ids_wells, x_coords, y_coords, wtds, dates)
transient_wtd_piedmont = all_data_piedmont.data_frame()
del shape, wells_piedmont, ids_wells, x_coords, y_coords, wtds, dates, well, n_points_well, df_well, feature, x_coord, y_coord
gc.collect()
average_wtd_piedmont = all_data_piedmont.average_wtd()

# %% EMILIA-ROMAGNA
years_levels = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
all_data_emilia = pd.DataFrame()
for year in years_levels:
    all_data_emilia = pd.concat([all_data_emilia, pd.read_excel(f'EmiliaRomagna/LivelliPiezometriciEmiliaRomagna{year}.xlsx')])

A2012 = pd.read_excel(f'EmiliaRomagna/Anagrafica2012.xlsx')
A2013 = pd.read_excel(f'EmiliaRomagna/Anagrafica2013.xlsx')
A2014 = pd.read_excel(f'EmiliaRomagna/Anagrafica2014.xlsx')
A2015 = pd.read_excel(f'EmiliaRomagna/Anagrafica2015.xlsx')
A2016 = pd.read_excel(f'EmiliaRomagna/Anagrafica2016.xlsx')
A2017 = pd.read_excel(f'EmiliaRomagna/Anagrafica2017.xlsx')
A2018 = pd.read_excel(f'EmiliaRomagna/Anagrafica2018.xlsx')
A2019 = pd.read_excel(f'EmiliaRomagna/Anagrafica2019.xlsx')
A2020 = pd.read_excel(f'EmiliaRomagna/Anagrafica2020.xlsx')

A2012 = anagrafica(A2012['Codice'], A2012['X_UTM-ED50*'], A2012['Y_UTM-ED50*']+400000)
A2013 = anagrafica(A2013['Codice'], A2013['X_UTM-ED50*'], A2013['Y_UTM-ED50*']+400000)
A2014 = anagrafica(A2014['Codice'], A2014['X_UTM-ED50*'], A2014['Y_UTM-ED50*']+400000)
A2015 = anagrafica(A2015['Codice'], A2015['XUTM-ETRS89 (fuso 32)'], A2015['YUTM-ETRS89 (fuso 32)'])
A2016 = anagrafica(A2016['Codice'], A2016['XUTM-ETRS89 (fuso 32)'], A2016['YUTM-ETRS89 (fuso 32)'])
A2017 = anagrafica(A2017['Codice'], A2017['X_UTM-ED50*'], A2017['Y_UTM-ED50*'])
A2018 = anagrafica(A2018['Codice'], A2018['XUTM-ETRS89 (fuso 32)'], A2018['YUTM-ETRS89 (fuso 32)'])
A2019 = anagrafica(A2019['Codice'], A2019['XUTM-ETRS89 (fuso 32)'], A2019['YUTM-ETRS89 (fuso 32)'])
A2020 = anagrafica(A2020['Codice'], A2020['XUTM-ETRS89 (fuso 32)'], A2020['YUTM-ETRS89 (fuso 32)'])

A2012.transform_coords(A2012.well_x_coords, A2012.well_y_coords, 'epsg:23032', '+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs')
A2013.transform_coords(A2013.well_x_coords, A2013.well_y_coords, 'epsg:23032', '+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs')
A2014.transform_coords(A2014.well_x_coords, A2014.well_y_coords, 'epsg:23032', '+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs')
A2015.transform_coords(A2015.well_x_coords, A2015.well_y_coords, 'epsg:25832', '+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs')
A2016.transform_coords(A2016.well_x_coords, A2016.well_y_coords, 'epsg:25832', '+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs')
A2017.transform_coords(A2017.well_x_coords, A2017.well_y_coords, 'epsg:23032', '+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs')
A2018.transform_coords(A2018.well_x_coords, A2018.well_y_coords, 'epsg:25832', '+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs')
A2019.transform_coords(A2019.well_x_coords, A2019.well_y_coords, 'epsg:25832', '+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs')
A2020.transform_coords(A2020.well_x_coords, A2020.well_y_coords, 'epsg:25832', '+proj=eck4 +lon_0=0 +datum=WGS84 +units=m no_defs')

A2012 = pd.DataFrame({'Well ID': A2012.well_ids, 'X_54012': A2012.well_x_coords, 'Y_54012': A2012.well_y_coords})
A2013 = pd.DataFrame({'Well ID': A2013.well_ids, 'X_54012': A2013.well_x_coords, 'Y_54012': A2013.well_y_coords})
A2014 = pd.DataFrame({'Well ID': A2014.well_ids, 'X_54012': A2014.well_x_coords, 'Y_54012': A2014.well_y_coords})
A2015 = pd.DataFrame({'Well ID': A2015.well_ids, 'X_54012': A2015.well_x_coords, 'Y_54012': A2015.well_y_coords})
A2016 = pd.DataFrame({'Well ID': A2016.well_ids, 'X_54012': A2016.well_x_coords, 'Y_54012': A2016.well_y_coords})
A2017 = pd.DataFrame({'Well ID': A2017.well_ids, 'X_54012': A2017.well_x_coords, 'Y_54012': A2017.well_y_coords})
A2018 = pd.DataFrame({'Well ID': A2018.well_ids, 'X_54012': A2018.well_x_coords, 'Y_54012': A2018.well_y_coords})
A2019 = pd.DataFrame({'Well ID': A2019.well_ids, 'X_54012': A2019.well_x_coords, 'Y_54012': A2019.well_y_coords})
A2020 = pd.DataFrame({'Well ID': A2020.well_ids, 'X_54012': A2020.well_x_coords, 'Y_54012': A2020.well_y_coords})

all_anagrafica = pd.concat([A2012, A2013, A2014, A2015, A2016, A2017, A2018, A2019, A2020])
del years_levels, year, A2012, A2013, A2014, A2015, A2016, A2017, A2018, A2019, A2020

unique_wells = np.intersect1d(np.unique(all_data_emilia['Codice']), np.unique(all_anagrafica['Well ID']))

ids = []
x_coords = []
y_coords = []
wtds = []
dates = []

for well in unique_wells:
    data_well = all_data_emilia[all_data_emilia['Codice'] == well]
    n_points_well = len(data_well)
    ids.append([well]*n_points_well)
    x_coords.append([all_anagrafica[all_anagrafica['Well ID'] == well]['X_54012'].values[0]]*n_points_well)
    y_coords.append([all_anagrafica[all_anagrafica['Well ID'] == well]['Y_54012'].values[0]]*n_points_well)
    wtds.append(data_well['Soggiacenza (m)'])
    dates.append(data_well['Data'])

x_coords = np.concatenate(x_coords)
y_coords = np.concatenate(y_coords)
ids = np.concatenate(ids)
wtds = np.concatenate(wtds)
dates = np.concatenate(dates)

all_data_emilia = well_data(ids, x_coords, y_coords, wtds, dates)
del unique_wells, well, data_well, n_points_well, ids, x_coords, y_coords, wtds, dates, all_anagrafica
gc.collect()

all_data_emilia.clean_data()
average_wtd_emilia = all_data_emilia.average_wtd()
transient_wtd_emilia = all_data_emilia.data_frame()


# %% Join all regions and compute average wtd per well
wtd_all_regions = pd.concat([transient_wtd_lombardy, transient_wtd_piedmont, transient_wtd_emilia])
wtd_all_regions['Date'] = pd.to_datetime(wtd_all_regions['Date'])
wtd_all_regions['average_wtd'] = np.zeros(len(wtd_all_regions['WTD']))
for id in np.unique(wtd_all_regions['Well ID']):
	wtdAve = np.average(wtd_all_regions['WTD'][wtd_all_regions['Well ID']== id])
	wtd_all_regions.loc[wtd_all_regions['Well ID']== id, 'average_wtd'] = wtdAve

# %% Remove wells with less than 30 measurements or with negative values
wells_to_remove = []

for id in np.unique(wtd_all_regions['Well ID']):
    if len(wtd_all_regions['Well ID'][wtd_all_regions['Well ID']==id])<30 or np.any(wtd_all_regions['WTD'][wtd_all_regions['Well ID']==id]<0):
        wells_to_remove.append(id)

wtd_all_regions = wtd_all_regions[~wtd_all_regions['Well ID'].isin(wells_to_remove)]
# %% Save data for plots
startT = datetime.strptime('2000-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
endT = datetime.strptime('2020-12-31 00:00:00', "%Y-%m-%d %H:%M:%S")
deltaT = timedelta(days=30)
time_steps = np.arange(startT, endT, deltaT).astype(datetime)

n_wells_per_time_interval = []
delta_wtd_per_time_interval = []
for i in range(len(time_steps)-1):
	subset = wtd_all_regions[(wtd_all_regions['Date']>=time_steps[i]).values & (wtd_all_regions['Date']<time_steps[i+1]).values]
	n_wells_per_time_interval.append(len(subset['Well ID'].unique()))
	copy = subset.copy()
	copy['delta'] = copy['WTD'] - copy['average_wtd']
	copy.groupby('Well ID').mean()
	delta_wtd_per_time_interval.append(copy['delta'].values)

delta_mean = []
delta_95 = []
delta_5 = []
for data_time_step in delta_wtd_per_time_interval:
	delta_mean.append(np.mean(data_time_step))
	delta_95.append(np.percentile(data_time_step, 95))
	delta_5.append(np.percentile(data_time_step, 5))

# %% Plot Figures
for i in range(len(time_steps)-1):
    img = np.array(plt.imread("GIS_Data/background.tif"))
    img[:,:,3] = img[:,:,3]*0.32
    ext = [540000.0000, 1100000.0000,5400000.0000,5740000.0000]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1.2, 2]})
    ax[0].imshow(img, zorder=0, extent=ext)
    subset = wtd_all_regions[(wtd_all_regions['Date']>=time_steps[i]).values & (wtd_all_regions['Date']<time_steps[i+1]).values]
    lon0 = subset['X_54012']
    lat0 = subset['Y_54012']
    copy = subset.copy()
    copy['Delta'] = copy['WTD'] - copy['average_wtd']
    x = copy['X_54012']
    y = copy['Y_54012']
    delta = copy['Delta']

    im0 = ax[0].scatter(x, y, zorder=1, s=20, c=delta, cmap='bwr', vmin=-3, vmax=3)
    ax[0].set_title(time_steps[i].strftime("%Y-%m-%d"))
    plt.colorbar(im0, ax=ax[0], label = 'WTD[m] - average WTD', orientation = 'horizontal')

    ax[1].fill_between(time_steps[:-1], delta_5, delta_95, alpha = 0.5, label = '5th-95th percentile')
    ax[1].plot(time_steps[:-1], delta_mean, marker = 'o', markerfacecolor="w", label = 'Average deviation')
    ax[1].hlines(0, time_steps.min(), time_steps.max(), 'black', linewidth=1)
    ax[1].set_ylabel('WTD[m] - Average WTD[m]')
    ax[1].legend(ncol = 2, frameon = False, loc = 'upper left')
    axt = ax[1].twinx()
    axt.plot(time_steps[:-1], n_wells_per_time_interval, color = 'red', marker = 'o', markerfacecolor="w", alpha = 0.5, label = 'Number of wells')
    axt.vlines(time_steps[i], 0, 430)
    axt.set_ylabel('Number of wells')
    axt.set_xlim([startT, endT])
    axt.set_ylim(0, 430)
    axt.legend(ncol = 1, frameon = False, loc = 'lower right')
    plt.savefig(f"Figures/Figure_{i}.jpg")
    plt.close(fig)

# %% 
# Create a video from the figures
one_figure = cv2.imread(f'Figures/Figure_0.jpg')
height, width, layers = one_figure.shape
out_video =  np.empty([len(time_steps)-1, height, width, 3], dtype = np.uint8)
out_video =  out_video.astype(np.uint8)

for i in range(len(time_steps)-1):
    out_video[i] = cv2.imread(f'Figures/Figure_{i}.jpg')

skvideo.io.vwrite("my_video.avi", out_video, inputdict={'-framerate':str(2)})