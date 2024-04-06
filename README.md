# Welcome to my FastF1 Functions depository

This is effectively a folder structure for a locally installed package of functions which enables you to create graphs from data downloaded from FastF1 using a Juypter notebook.

My main aim with this is to make it as easy as possible to create detailed graphs and charts with minimal input into functions. The intended structure for analysing is a different folder for each race, e.g. "01_Bahrain", each containing Juypter notebooks for the race and qualifying. Inside these you call functions from the .py files in the data folder. 

Below I will outline the structure of the depository, and how to use it.

# Structure
### data/
The data folder contains two folders in itself: `qualifying/` and `race/`. Inside each of these folders is a `qualifying.py` and a `race.py`, each of which contains the functions for analysing either qualifying or the race. I will describe those functions in a second. 

The final thing inside data/ is `season_info.json` which contains five items. 
- `teams` is a list of all of the team names. 
- `team_colours` is a dictionary. The key is the team name, the value is a hex code string for the colour of that team. These are subjective, but should be agreeable to most. 
- `tyre_colours` is a dictionary. The key is the tyre name (both dries and wets) in lower case, the value is a hex code for the tyre colours. 
- `drivers` is a dictionary with sub-dictionaries. The key is each driver's three letter initials (e.g. VER), the value is a sub-dictionary. In that subdictionary there are three keys, Team, Full Name, and Surname, whose contents should be self-explanatory.
- `races` is a dictionary with sub-dictionaries. The key is the race name (e.g. China, Emilia-Romagna), the value is a dictionary. There are 8 keys: Round (int), Official name (str), Track (str), Date (str, DD/MM/YYYY), Time (str), Sprint (Bool), Laps (int), and Sprint Laps (int, if Sprint=False, will not exist).


### 00_Example/
This is an example folder to show you the basic template for how you could structure each folder. 
```
.
└── 01_Bahrain/
	├── qualifying.ipynb
	└── race.ipynb
```
Inside `00_Example/` is `example.ipynb`, which contains the import structure that is intended to be used before you can call functions. 

- Under the `grand_prix` variable, you enter the string name of the Grand Prix. This should match the format used in `season_info.json`, though this isn't necessary. 
- `safety_car_laps` is a list of strings for each safety car period. If *x* is the start lap of the first safety car, and *y* is its end lap, the string should be formatted `"x-y"`. For the second, third, etc. safety car periods the format should be the same. Obviously if you are looking at qualifying, there is no need to input anything here.
- `fastf1_dir` is a string with the intended filepath to FastF1's cache. It equals None by default as if you haven't used FastF1 before you won't have one. However, setting a cache will stop you from having to download the data everytime you wish to access a specific grand prix's data.
- `race` is a boolean. If `True`, then data from the race will be loaded, if `False`, data from qualifying will be loaded.

Next follows the imports from the data folder. I've called the race functions `f1r`, and from qualifying `f1q`. FastF1 also needs to be imported, which I've done as `f1`.  Finally, json needs to be imported to get the information from `season_info.json`.

The next code turns each key from `season_info.json` into separate variables. 

Then, the race/qualifying data is loaded from FastF1. This will take upwards of a minute. This data is stored in the variable `DATA` as a pandas DataFrame.

Finally, the `initial_setup()` function is called from `data/race/race.py`, which converts the safety car string into a list of tuples, if applicable, and also gives `all_laps_by_team`. This is a dictionary of all laps broken down by team, the key being the team name, while the value is a pandas DataFrame.

From here you are able to create any new code boxes and call functions from either f1r or f1q, using the format `f1r.function()`. Every function requires the `DATA` variable as an input, with different extra requirements depending on the function. 

<br>

# Functions
## `race.py`
### `f1r.initial_setup(data, safety_car_laps=None)`
Takes `DATA` from loading FastF1, and returns `all_laps_by_team`. This is a dictionary of all laps broken down by team, the key being the team name, while the value is a pandas DataFrame.
 
Converts the safety car string into a list of tuples, if applicable. If not provided, skips this process.

<br>

### `f1r.laps_scatter(data, grand_prix, sc_laps, drivers_in_graph, ylim=None)`
Produces a scatter of lap times over the course of a race for drivers provided, broken down by tyre. Provides a line of best fit for each stint, while trimming laps over the mean + standard deviation *of each stint*.
#### Parameters
- `data = DATA` from FastF1.
- `grand_prix` is the string name of the Grand Prix, e.g. "Australia". Must be the same format as in `season_info.json`.
- `sc_laps` takes the `sc_laps` list of tuples provided in by the `initial_setup()` variable. 
- `drivers_in_graph` a list of strings of the drivers three letter acronyms. E.g. `["ALB", "VER", "HAM"]`.
- `ylim` Optional. A tuple of floats, if you wish to change the y limits of the graph. Will be in the 0.0006 to 0.0015 range.

<br>

### `f1r.compare_telemetry(data, driver_info, vlines=None, xlim=None, delta_ylim=None)`
Displays the telemetry data of drivers you wish to compare. X axis is the distance into the lap, with three Y axes: speed, throttle %, and delta to the first provided driver. If only one driver is provided, will not show the delta graph.

#### Parameters
- `data = DATA` from FastF1.
- `driver_info` is a list of dictionaries. Each dictionary being the info from a driver. Example:
	```
	driver_info = [{
		"driver": "HAM",
		"lap": 56,
		"colour": "red" (opt.)
	}]
	```
	- `driver` is the three letter acronym if the driver
	- `lap` is the lap whose telemetry you wish to look at
	- `colour` is an optional override for the colour the driver will be, of any format Matplotlib takes (e.g. word, hex).
- `vlines` Optional. Puts two vertical lines on all charts,  highlighting an area, with a string equidistant between them. Is a list of dictionaries of any block you wish to put vlines over. 
	```
	vlines = [{
		"start": int,
		"stop": int,
		"label": str,
		"colour": str,
	}]
	```
	- `start` is the int of how far into the lap you want to put the first line (metres)
	- `end` is the second vline
	- `label` is the string which goes between them
	- `colour` is the colour of both the lines and the text.
- `xlim` Optional. A tuple of two integers, to change the x limit of the graph. E.g. (1000, 1500) would only show the telemetry from the 500 metres between 1km into the lap and 1.5km into the lap.
- `delta_ylim` Optional. A tuple of two floats. If you wish to constrain the Delta chart between two values. 

<br>

### `position_changes(data)`
A line graph showing each driver's position over the course of the race, and thus the changes in position. 

Takes `DATA` from FastF1.

<br>

### `corner_trace(data, driver_info, corner = None, zoom 10)`

Creates a trace of the track, also giving you the ability to zoom in on any corner. Need to provide driver info.

Mostly useless, as I thought it gave the line a driver drove, whereas it's actually just a pre-scripted series of X/Y coords.

#### Parameters
- `data = DATA` from FastF1.
- `driver_info` is a list of dictionaries containing info for the drivers you want on the graph. Completely superfluous, as drivers "drive" on a pre-scripted line (i.e. no intra-driver variation).
	```
	driver_info = [
		{"driver": str, "lap": int},
	]
	```
	- `driver` is the three-letter abbreviation of the driver's name (e.g. VER), "lap" is the integer lap you want to look at.
- `corner` Optional. None by default (for whole track view). If you want to zoom in on a corner, give the corner number as an integer.
- `zoom` Optional. If a corner is specified, will by default zoom in by 10 times on that corner if unspecified. If you give it a different integer, will use that zoom level instead. If `corner` is unspecified, will do nothing. 

<br>
<br>

## `qualifying.py`
### `compare_telemetry(data, driver_info, vlines] = None, xlim = None, delta_ylim = None)`
Compares the qualifying telemetry of any number of drivers you choose. X axis is the distance into the lap, with three Y axes: speed, throttle %, and delta to the first provided driver. 

#### Parameters
- `data = DATA` from FastF1.
- `driver_info` is a list of dictionaries. Each dictionary being the info from a driver. Example:
	```
	driver_info = [{
		"driver": "HAM",
		"session": q1,
		"colour": "red" (opt.)
	}]
	```
	- `driver` is the three letter acronym if the driver
	- `session` is which qualifying session from which that driver's fastest lap comes.
	- `colour` is an *optional* override for the colour the driver will be, of any format Matplotlib takes (e.g. word, hex).
- `vlines` Optional. Puts two vertical lines on all charts,  highlighting an area, with a string equidistant between them. Is a list of dictionaries of any block you wish to put vlines over. 
	```
	vlines = [{
		"start": int,
		"stop": int,
		"label": str,
		"colour": str,
	}]
	```
	- `start` is the int of how far into the lap you want to put the first line (metres)
	- `end` is the second vline
	- `label` is the string which goes between them
	- `colour` is the colour of both the lines and the text.
- `xlim` Optional. A tuple of two integers, to change the x limit of the graph. E.g. (1000, 1500) would only show the telemetry from the 500 metres between 1km into the lap and 1.5km into the lap.
- `delta_ylim` Optional. A tuple of two floats. If you wish to constrain the Delta chart between two values. 

### `compare_telemetry_different_years(driver_info, vlines = None, xlim = None, delta_ylim = None)`
Compares the qualifying telemetry of any number of drivers you choose -- from different years (or tracks if you so choose, though of limited use).

X axis is the distance into the lap, with three Y axes: speed, throttle %, and delta to the first provided driver. 

<br>

#### Parameters
N.B.: unlike every other function, this *does not* take DATA from fastf1 as an initial parameter. Instead, it is passed in in `driver_info`.
- `driver_info` is a list of dictionaries. Each dictionary being the info from a driver. Example:
	```
	driver_info = [{
		"data": fastf1_data,
		"driver": "HAM",
		"session": q1,
		"colour": "red" (opt.)
	}]
	```
	- `data` is the data from FastF1 for the session you want that data from. E.g. if you want the data from the 2023 Japanese Grand Prix, load `DATA = f1.get_session(year=2023, gp="Japan", identifier="Q")`, then `DATA.load()`, and then pass DATA into the `driver_info` dictionary.

	- `driver` is the three letter acronym if the driver
	- `session` is which qualifying session from which that driver's fastest lap comes.
	- `colour` is an optional override for the colour the driver will be, of any format Matplotlib takes (e.g. word, hex).
- `vlines` Optional. Puts two vertical lines on all charts,  highlighting an area, with a string equidistant between them. Is a list of dictionaries of any block you wish to put vlines over. 
	```
	vlines = [{
		"start": int,
		"stop": int,
		"label": str,
		"colour": str,
	}]
	```
	- `start` is the int of how far into the lap you want to put the first line (metres)
	- `end` is the second vline
	- `label` is the string which goes between them
	- `colour` is the colour of both the lines and the text.
- `xlim` Optional. A tuple of two integers, to change the x limit of the graph. E.g. (1000, 1500) would only show the telemetry from the 500 metres between 1km into the lap and 1.5km into the lap.
- `delta_ylim` Optional. A tuple of two floats. If you wish to constrain the Delta chart between two values. 

<br>

### `compare_telemetry_deprecated(data, driver1, driver2)`

**DEPRECATED**

Compares the qualifying telemetry for two drivers. X axis is the distance into the lap, with three Y axes: speed, throttle %, and delta to the first provided driver. 
- `data = DATA` from FastF1.
- `driver1` = the three letter initials of the first driver you wish to look at, e.g. "VER".
- `driver2` = the three letter initials of the second driver you wish to look at. 

<br>
