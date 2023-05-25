def preprocess_weather_baseline(weather_df):
    return (
        weather_df
            .drop(
            columns=['moon_illumination', 'moonrise', 'moonset', 'sunrise', 'winddirDegree', 'location', 'DewPointC',
                     'sunset'])
            .set_index('date_time')
    )

def preprocess_weather_paper(weather_df):
    weather_attributes = dict(
        maxtempC = "maxTempC",
        mintempC = "minTempC",
        tempC = 'avgTempC',
        sunHour = 'sunHour',
        uvIndex = 'uvIndex',
        FeelsLikeC = 'feelsLikeC',
        windspeedKmph = 'windspeedKmph',
        humidity = 'humidity',
        date_time = 'date_time',
    )
    return (
        weather_df
            [list(weather_attributes.keys())]
            .rename(columns = weather_attributes)
            .set_index('date_time')
            .astype('float')
    )
