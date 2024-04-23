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
- `variables` is a dictionary with sub-dictionaries. The key is the fastf1 variable name (e.g. SpeedST), the value is another dictionary. In that dictionary there are two keys, "Title" and "Axis", which explains the variable name. E.g. `SpeedST: {"Title": "Speed Trap Speed", "Axis": "Speed Trap [km/h]"}`. Some graphs use these to properly title charts and their axes.


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

### `speed_trap_distribution(data, starters=20, ylim=None, include_drs=True)`
Gives a violin plot of the distribution of driver's speeds through the speed trap on track. Provides one violin plot per driver. 

#### Parameters
- Takes `DATA` from FastF1.
- `starters` is the number of drivers that start the race. I.e. if a driver does not start the race, by default an empty violin plot will be provided. Limiting finishers to 19 will eliminate this empty space.
- `ylim` the limit on the y axis. Is a tuple, e.g. (200, 320).
- `include_drs` is a boolean for whether laps which have DRS should be included or not. 

<br>

### `speed_trap_table(data, include_drs=True)`
Returns a pandas DataFrame. For each driver, you have the average speed they go through the speed trap (in km/h), called AverageSTSpeed, and the number of laps with usable data, called LapCount. 

`include_drs` is a boolean which is True by default. Like `speed_trap_distribution()`, it determines whether laps where a driver achieved DRS down the speed trap straight should be included. If this is set to False, the laps data will be gathered from will be limited. 

<br>

### `custom_scatter(data, drivers_in_graph, x_data, y_data, z_data=None, z_reverse=False, xlim=None, ylim=None, title=True)`
A custom scatter graph. You can give it any variable in the laps level of data and it will return a Seaborn scatter of them. 

#### Parameters
- Takes `DATA` from FastF1.
- `drivers_in_graph` a List of the three-letter acronyms of the drivers in the graph. E.g. ["HAM", "VER", "BOT"].
- `x_data` the FastF1 variable name for the column on the x axis. Can take custom inputs: described below.
- `y_data` the FastF1 variable name for the column on the y axis. Can take custom inputs: described below.
- `z_data` Optional, the FastF1 variable name for the column on the z axis. Can take custom inputs: described below. This takes the form of the dot colour and size for the scatter. Largest values will be largest dots by default.
- `z_reverse` Optional, False by default. If True, will reverse the dot size/colour scale of the z data, i.e. smallest values will be the largest dots.
- `xlim` Optional, the limits on the x axis. Takes the form of a Tuple.
- `ylim` Optional, the limits on the y axis. Takes the form of a Tuple.
- `title` Optional, True by default. Boolean to determine whether the graph has a title.

#### Custom inputs 
Certain custom inputs are applicable to this graph. This generally takes the form of a lap average of a certain type of telemetry-level data. The data for these variables are in season_info.json. 
- ThrottleAvg, average throttle % over each lap.
- RPMAvg, average RPM over a lap.
- GearAvg, average gear over a lap.

This list will evolve over time as I add more.

<br>

### `lap_average(laps, variable)`
This returns a table with an additional row for the custom varaible you want from it. These variables are from a specific list. The custom_scatter() function uses this function if you ask it to plot one of these varaibles. 

Has a dictionary entry from season_info.json for the names. 

#### Custom variables 
- ThrottleAvg, average throttle % over each lap.
- RPMAvg, average RPM over a lap.
- GearAvg, average gear over a lap.

<br>

### `low_speed_corner_analyser(data, low_speed_corners, search_range=150)`
Finds the average speed for an entire race for each driver through low speed corners and returns two tables. You have to provide a dictionary for which corners these are. 

#### Returns
Returns TWO tables, `average_min_speeds_per_driver` and `normalised_lowspeed`.
- `average_min_speeds_per_driver` is the average minimum speed for each corner given, in km/h terms. Also has an average.
- `normalised_lowspeed` is the previous graph but normalised. The driver with the quickest average minimum speed equals 1, and the rest are arrayed proportionally.

#### Parameters
- Takes `DATA` from FastF1.
- `low_speed_corners` is the dictionary of low-speed corners which you have determined. It takes the following form:

	`{"Corner Name": metres_into_track, etc.}`

	where corner_name is a string for the name you wish to give the corner (e.g. Spoon, or Turn 14), and metres_into_track is approximately how far into the circuit it is, as measured by FastF1. 

<br>
To get the distance into the track each corner is do,

```
circuit_info = DATA.get_circuit_info()
corner_location = circuit_info.corners["Distance"]
print(corner_location)
```
while noting that the table will start from 0, so Turn 1 will be on the 0 value and Turn 6 will be on the 5 value. DATA is what is generally labelled "session" in FastF1 docs.

<br>

### `normalised_lowspeed_bar(normalised_lowspeed, all_corners=False, xlim=[0.9, 1])`
Gives a vertical bar chart for the normalised low corner speeds from the previous function, low_speed_corner_analyser().

- `normalised_lowspeed` is the *second* return data from `low_speed_corner_analyser`.
- `all_corners` determines whether the data is shown for all corners analysed, or simply the average.
- `xlim` Optional. The x limits for the graph.

There isn't a reason why other data couldn't be provided to this graph (especially the first return data from low speed corner analyser), but you likely will have to change the x limits.

<br>

### `gap_to_x_gen(laps, relative_driver=None)`
Generates a DataFrame with a new column showing the gap in seconds to a certain driver, or to the leader, for the start of each lap.

#### Parameters
- `laps` takes *LAP LEVEL DATA*, unlike most graphs made with this.
- `relative_driver` Optional. If not provided, will generate the gap to the leader, else if you provide the three-letter acronym of any driver in the race, it will generate the gap to that driver.

<br>

###  `gap_to_x_graph(data, relative_driver=None, drivers_in_graph=[], ylim=None, sc_laps=[])`
Generates a line graph showing the gap in seconds to a certain driver, or to the leader, for the start of each lap over the course of a race.

#### Parameters
- Takes `DATA` from FastF1.
- `relative_driver` Optional. If not provided, will use `gap_to_x_gen()` to generate the gap to the leader for each driver. If a three letter driver acronym is provided, it will generate the gap to that driver instead.
- `drivers_in_graph` Optional. If not provided, will generate lines for all drivers. If only certain drivers are specified, will do lines for just. 
- `ylim` Optional, the limits on the y axis. Takes the form of a Tuple.
- `sc_laps` takes the `sc_laps` list of tuples provided in by the `initial_setup()` variable. For each safety car instance, will provide two vertical yellow lines for the start and end of them, and a "SC" label equa-distant between them.

<br>


<br>
<br>

## `qualifying.py`
### `compare_telemetry(data, driver_info, vlines] = None, xlim = None, delta_ylim = None)`

N.B.: the main change from the compare_telemetry() function in race.py is that in driver_info "lap" is now "session", requiring q1/q2/q3.

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
	- `session` is which qualifying session from which that driver's fastest lap comes. Acceptable inputs: q1/q2/q3 (or upper case).
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

<br>

### `compare_telemetry_different_years(driver_info, vlines = None, xlim = None, delta_ylim = None)`
Compares the qualifying telemetry of any number of drivers you choose -- from different years (or tracks if you so choose, though of limited use).

X axis is the distance into the lap, with three Y axes: speed, throttle %, and delta to the first provided driver. 

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
