## Resources

This repo contains data from leak events from a 24 hour window. All documents are csv files and any date/time data is in UTC.

The specific facility the data is drawn from is shown here:

![](facility_map.png)

[Here is a link](https://www.google.com/maps/place/40%C2%B035%2746.0%22N+105%C2%B008%2724.3%22W/@40.5955073,-105.1399915,163m/data=!3m1!1e3!4m4!3m3!8m2!3d40.596114!4d-105.140075?entry=ttu) to the facility in Google Maps

## Files

```sensor_readings.csv```

This file gives the methane concentration readings for the 24 sensors at the facility. The sensors are capable of being effected by temperature and humidity. Every sensor has a unique key that gives a latitude and longitude within the key.

Units:
- `SensorData` - Parts per Billion (ppb)
---

```leak_locations_and_rate.csv```

This file gives the latitude and longitude coordinates of known leak points or leak sources. A leak point for instance could be an oil storage tank or a pipe connection between equipment. Known leak rates of the leak points at the facility are also included. This data was captured for model calibration purposes.


Units:
- `LeakRate` - kg/h

---

```weather_data.csv```

This file provides weather data such as pressure, temperature, wind speed etc. at the facility during the 24 hour timeframe.

Units:
- Barometric Pressure - millibars (mb)
- Temperature - Celsius
