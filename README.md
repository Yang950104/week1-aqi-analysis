# AQI Map Project

This small Python project fetches real-time AQI data from Taiwan's Environmental Protection Administration and visualizes all monitoring stations on an interactive map (`aqi_map.html`).

## Features

- Reads API key from `.env` file (ignored by git) using the variable `MOENV_API_KEY`.
- Automatically installs required packages (`dotenv`, `requests`, `folium`, `pandas`).
- Queries the MOE Environment API at `data.moenv.gov.tw` to retrieve AQI data.
- Generates a Leaflet map with station markers colored by AQI.

## Setup & Usage

1. Make sure `.env` in the project directory contains a valid `MOENV_API_KEY` value.
2. Run the main script:

   ```bash
   python aqi_map.py
   ```

   The script will install dependencies if necessary and produce `aqi_map.html`.

3. Open `aqi_map.html` in a browser to view the map.

Additional outputs:

   - A CSV file `outputs/aqi_with_distances.csv` will be generated. It contains each station's name, county, AQI, and the computed distance (kilometers) from Taipei Main Station (25.0478, 121.5170).



## Directory Structure

```
python_project/
├── aqi_map.py
├── data/          # (empty) for raw data files
├── outputs/       # (empty) for generated outputs
├── .env           # contains API key
├── .gitignore     # excludes .env, data, outputs
└── README.md
```

Feel free to extend the script or integrate it into a larger workflow.