import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator

import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.express as px

from timple.timedelta import strftimedelta
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any
from PIL import ImageColor


import fastf1 as f1
import fastf1.plotting
from fastf1.core import Laps
from fastf1.ergast import Ergast
from fastf1 import utils

ergast = Ergast(result_type='pandas')

with open("../data/season_info.json") as file:
    season_info = json.loads(file.read())
teams = season_info["teams"]
team_colours = season_info["team_colours"]
tyre_colours = season_info["tyre_colours"]
drivers = season_info["drivers"]
races = season_info["races"]
variables = season_info["variables"]

fastf1.plotting.setup_mpl(mpl_timedelta_support=True,
                          color_scheme="fastf1", misc_mpl_mods=True)

font_files = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

font = font_manager.FontProperties(
    family="Formula1", weight="normal", style="normal", size=16)


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def initial_setup(data, safety_car_laps=None):
    '''Takes a string list format for safety car beginning/end and converts it to a list of tuples'''

    return_data = dict()

    if safety_car_laps is not None:
        sc_laps = list()

        for sc in safety_car_laps:
            laps = sc.split("-")
            start_lap = int(laps[0])
            end_lap = int(laps[1])
            sc_laps.append((start_lap, end_lap))

        return_data["safety_car"] = sc_laps

    LAPS = data.laps
    all_laps_by_team = dict()
    for team1 in teams:
        team_data = LAPS[LAPS["Team"] == team1]
        all_laps_by_team[team1] = team_data.reset_index()
    return_data["team_laps"] = all_laps_by_team

    return return_data


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def lap_scatter(data, grand_prix, sc_laps: List[Tuple[int, int]], drivers_in_graph: List[str], ylim: Tuple[float, float] = None):
    '''Produces a scatter of lap times over the course of a race for drivers provided.'''

    LAPS = data.laps

    axis_fontsize = 16
    tick_fontsize = 12

    teams_in_graph = list()  # whether the line needs to be dashed or not
    handles = list()  # handles for legend
    max_time = None
    min_time = None

    plt.figure(figsize=(16, 8), dpi=120)
    ax = plt.gca()

    with sns.axes_style():
        for driver in drivers_in_graph:

            # GETTING INITIAL INFO ABOUT EACH DRIVER
            driver = driver.upper()
            current_team = drivers[driver]["Team"]
            team_colour = team_colours[current_team]
            driver_laps = LAPS[LAPS["Driver"] == driver].reset_index()

            driver_laps = driver_laps.pick_quicklaps()
            driver_laps = driver_laps.pick_track_status(
                how="equals", status="1")

            # if two drivers are of the same team, the dots in the scatter are squares
            # also gives them the legend handle with the correct colour and name
            if current_team not in teams_in_graph:
                marker = "o"
                best_fit_line_style = "-"
                dot_size = 65
                handle = mlines.Line2D(
                    [], [],
                    linestyle="-", color=team_colour, marker=marker, markersize=12,
                    label=driver
                )
            else:
                marker = "*"
                best_fit_line_style = "--"
                dot_size = 100
                handle = mlines.Line2D(
                    [], [],
                    linestyle="--", color=team_colour, marker=marker, markersize=12,
                    label=driver
                )

            teams_in_graph.append(current_team)
            handles.append(handle)

            # CREATING THE SCATTER PLOTS FOR EACH TYRE AND DRIVER
            # splitting drivers laps by tyre
            laps_by_tyre = {"soft": None, "medium": None, "hard": None}

            for tyre, tyre_laps in laps_by_tyre.items():

                tyre_laps = driver_laps[driver_laps["Compound"] == tyre.upper()].dropna(
                    subset="LapTime", inplace=False)

                grouped_tyres = tyre_laps.groupby("Stint")
                split_dataset_by_stint = {}
                for group_name, group_data in grouped_tyres:
                    split_dataset_by_stint[group_name] = group_data

                for stint_number, dataset in split_dataset_by_stint.items():
                    mean = dataset["LapTime"].mean(numeric_only=False)
                    std = dataset["LapTime"].std(numeric_only=False)
                    dataset = dataset[dataset["LapTime"] < (mean + std)]

                    # GRAPHS THEMSELVES
                    try:
                        ax = sns.regplot(
                            data=dataset, x="LapNumber", y="LapTime",
                            scatter=True,
                            marker=marker,
                            scatter_kws={"color": team_colour, "alpha": 1, "linewidths": 1.5,
                                         "edgecolor": tyre_colours[tyre], "s": dot_size},
                            line_kws={"color": team_colour,
                                      "linestyle": best_fit_line_style}
                        )
                    except TypeError:  # will get TypeError if a tyre hasn't been used
                        pass

                    # SETTING Y AXIS LIMITS. FINDS THE HIGHEST AND LOWEST POINTS IN THE GRAPH
                    min_time_tyre = dataset["LapTime"].min()
                    max_time_tyre = dataset["LapTime"].max()

                    if not np.isnat(min_time_tyre.to_numpy()):
                        if min_time == None:
                            min_time = min_time_tyre
                        elif (
                            min_time_tyre.seconds < min_time.seconds or
                            (min_time_tyre.seconds == min_time.seconds and
                             min_time_tyre.microseconds < min_time.microseconds)
                        ):
                            min_time = min_time_tyre

                    if not np.isnat(max_time_tyre.to_numpy()):
                        if max_time == None:
                            max_time = max_time_tyre
                        elif (
                            max_time_tyre.seconds > max_time.seconds or
                            (max_time_tyre.seconds == max_time.seconds and
                             max_time_tyre.microseconds > max_time.microseconds)
                        ):
                            max_time = max_time_tyre

    # SAFETY CAR VERTICAL LINES AND LABEL
    for sc in sc_laps:
        start_lap, end_lap = sc
        plt.axvline(
            x=start_lap, ymin=0, ymax=1,
            linestyle="dotted", color=tyre_colours["medium"]
        )
        plt.axvline(
            x=end_lap, ymin=0, ymax=1,
            linestyle="dotted", color=tyre_colours["medium"]
        )

        half_way_sc = (end_lap - start_lap) / 2 + start_lap

        if ylim is None:
            plt.text(
                x=half_way_sc, y=max_time * 0.998,
                s="SC", fontsize=16, fontname="Formula1", horizontalalignment='center',
                color=tyre_colours["medium"]
            )
        else:
            plt.text(
                x=half_way_sc, y=ylim[1] * 0.998,
                s="SC", fontsize=16, fontname="Formula1", horizontalalignment='center',
                color=tyre_colours["medium"]
            )

    # LEGEND
    handle = mlines.Line2D(
        [], [], linestyle="None",
        marker="o", markerfacecolor="None",
        markeredgecolor=tyre_colours["medium"], markersize=12,
        label="Tyre"
    )
    handles.append(handle)
    l = ax.legend(handles=handles, prop=font)
    for text in l.get_texts():
        text.set_color("white")

    # TITLE
    title = f"{data.event.year} {data.event["EventName"]} Race Pace\n"
    for driver in drivers_in_graph:
        title = title + f"{drivers[driver]["Full Name"]}, "
    title = title[:-2]
    plt.suptitle(title, fontsize=16, fontname="Formula1", color="white")

    # STYLE AND AXES
    sns.set_theme(rc={"axes.facecolor": "black",
                  "figure.facecolor": "#0F0F0F"})
    ax.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="--",
                  color="white", zorder=-1000)

    if ylim is not None:
        ax.set(
            ylim=ylim,
            xlim=(0, races[grand_prix]["Laps"] + 1)
        )
    else:
        ax.set(
            ylim=(min_time * 0.999, max_time * 1.001),
            xlim=(0, races[grand_prix]["Laps"] + 1)
        )

    # SETTING LAP TIME FORMAT
    time_format = mdates.DateFormatter("%M:%S")
    ax.yaxis.set_major_formatter(time_format)

    plt.yticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.xticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.xlabel("Lap Number", fontsize=axis_fontsize,
               color="white", fontname="Formula1")
    plt.ylabel("Lap Time", fontsize=axis_fontsize,
               color="white", fontname="Formula1")

    plt.show()


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def compare_telemetry(data, driver_info: List[Dict[str, Any]], throttle_tel: bool = True, brake_tel: bool = False, vlines: List[Dict[str, Any]] = None, xlim: Tuple[int, int] = None, delta_ylim: Tuple[float, float] = None):
    '''
    Takes a race data, and the info for drivers you want to compare as a list
    of dictionaries. Keys: driver, lap, colour override (opt.)

    EXAMPLES:
    driver_info = [{
        "driver": "HAM",
        "lap": 56,
        "colour": "red" (opt.)
    }]

    vlines = [{
        "start": int,
        "stop": int,
        "label": str,
        "colour": str,
    }]

    xlim = (0, 1000)
    limit for x axis 

    delta_ylim = (-2, 2)
    limit for y axis of delta 
    '''

    LAPS = data.laps
    circuit_info = data.get_circuit_info()

    axis_fontsize = 16
    tick_fontsize = 14

    line_styles = ["solid", "dotted", "dashdot", "dashed"]
    dash_capstyles = ["none", "round", "round", "round"]

    all_laps = list()
    all_teams = list()
    all_lap_numbers = list()

    brake_and_throttle = False
    if throttle_tel is True and brake_tel is False:
        middle_plot = "Throttle"
    elif throttle_tel is False and brake_tel is True:
        middle_plot = "Brake"
    else:
        brake_and_throttle = True

    for dict_ in driver_info:
        driver = dict_["driver"]
        lap = dict_["lap"]

        # if current lap number is not in all_lap_numbers, adds lap number to list
        # in order to see if the lap numbers need to be labelled
        if lap not in all_lap_numbers:
            all_lap_numbers.append(lap)

        try:
            colour_override = dict_["colour"]
        except KeyError:
            colour_override = None

        team = drivers[driver]["Team"]

        if colour_override is not None:
            team_colour = colour_override
        else:
            team_colour = team_colours[team]

        lap_info = LAPS.pick_driver(driver).pick_lap(lap).iloc[0]
        lap_tel = lap_info.get_car_data().add_distance()

        # counts number of times that team has appeared in the list. then references the line_style list to set a line style.
        if colour_override == None:
            same_team_in_list = all_teams.count(team)
            linestyle = line_styles[same_team_in_list]
            dash_capstyle = dash_capstyles[same_team_in_list]
            all_teams.append(team)
        else:
            linestyle = "-"
            dash_capstyle = "None"

        all_laps.append({
            "driver": driver,
            "team": team,
            "colour": team_colour,
            "linestyle": linestyle,
            "dash_capstyle": dash_capstyle,
            "lap number": lap,
            "lap info": lap_info,
            "lap tel": lap_tel
        })

    # if only one lap then don't need delta
    if len(all_laps) > 1:
        several_laps = True
    else:
        several_laps = False

    # WORKING OUT DELTAS
    if several_laps:
        for lap in all_laps[1:]:
            delta_time, ref_tel, compare_tel = utils.delta_time(
                all_laps[0]["lap info"], lap["lap info"]
            )

            lap["ref tel"] = ref_tel
            lap["delta time"] = delta_time

    '''GRAPH SETUP'''
    # FINDING MAX SPEED FOR UPPER BOUND OF SPEED GRAPH
    max_speed = 0
    for lap in all_laps:
        s_max = lap["lap tel"]["Speed"].max()
        if s_max > max_speed:
            max_speed = s_max

    # SET UP GRAPH AXES
    if not brake_and_throttle:
        if not several_laps:  # if only one driver cannot do delta chart
            fig, ax = plt.subplots(nrows=2, height_ratios=[3, 1], figsize=(
                16, 12), dpi=120, facecolor="#0F0F0F")
        else:
            fig, ax = plt.subplots(nrows=3, height_ratios=[3, 1, 1], figsize=(
                16, 12), dpi=120, facecolor="#0F0F0F")
    else:
        if not several_laps:  # if only one driver cannot do delta chart
            fig, ax = plt.subplots(nrows=3, height_ratios=[3, 1, 1], figsize=(
                16, 12), dpi=120, facecolor="#0F0F0F")
        else:
            fig, ax = plt.subplots(nrows=4, height_ratios=[3, 1, 1, 1], figsize=(
                16, 12), dpi=120, facecolor="#0F0F0F")

    # CORNER LINES
    for _, corner in circuit_info.corners.iterrows():

        if xlim is not None:
            if corner["Distance"] > xlim[0] and corner["Distance"] < xlim[1]:
                txt = f"{corner["Number"]}{corner["Letter"]}"
                ax[0].text(corner["Distance"], max_speed+20, txt, va="center_baseline",
                           ha="center", size="medium", fontname="Formula1", color="white")
        else:
            txt = f"{corner["Number"]}{corner["Letter"]}"
            ax[0].text(corner["Distance"], max_speed+20, txt, va="center_baseline",
                       ha="center", size="medium", fontname="Formula1", color="white")

    # CORNER MINOR AXIS LINE LOCATIONS
    corner_location = circuit_info.corners["Distance"]
    minor_tick_locations = list()
    for corner in corner_location:
        if xlim is not None:
            if corner > xlim[0] and corner < xlim[1]:
                minor_tick_locations.append(corner)
        else:
            minor_tick_locations.append(corner)

    '''GRAPHS'''
    for lap in all_laps:

        # if len(all_lap_numbers) > 1, then we are looking at several different laps, so the lap numbers need to be labelled
        if several_laps:
            label = f"{lap["driver"]}, Lap {lap["lap number"]}"
        else:
            label = lap["driver"]

        if brake_and_throttle is False:

            ax[0].plot(
                lap["lap tel"]["Distance"], lap["lap tel"]["Speed"], color=lap["colour"],
                label=label, linestyle=lap["linestyle"], linewidth=2
            )
            if brake_tel is True:

                ax[1].plot(
                    lap["lap tel"]["Distance"], lap["lap tel"]["Brake"] * 100, color=lap["colour"],
                    label=label, linestyle=lap["linestyle"], linewidth=2
                )
            else:
                ax[1].plot(
                    lap["lap tel"]["Distance"], lap["lap tel"]["Throttle"], color=lap["colour"],
                    label=label, linestyle=lap["linestyle"], linewidth=2
                )

            if len(all_laps) > 1 and lap != all_laps[0]:
                ax[2].plot(
                    lap["ref tel"]["Distance"], lap["delta time"],
                    color=lap["colour"], linestyle=lap["linestyle"], linewidth=2
                )

        else:
            ax[0].plot(
                lap["lap tel"]["Distance"], lap["lap tel"]["Speed"], color=lap["colour"],
                label=label, linestyle=lap["linestyle"], linewidth=2
            )
            ax[1].plot(
                lap["lap tel"]["Distance"], lap["lap tel"]["Throttle"], color=lap["colour"],
                label=label, linestyle=lap["linestyle"], linewidth=2
            )
            ax[2].plot(
                lap["lap tel"]["Distance"], lap["lap tel"]["Brake"] * 100, color=lap["colour"],
                label=label, linestyle=lap["linestyle"], linewidth=2
            )

            if len(all_laps) > 1 and lap != all_laps[0]:
                ax[3].plot(
                    lap["ref tel"]["Distance"], lap["delta time"],
                    color=lap["colour"], linestyle=lap["linestyle"], linewidth=2
                )

    '''GRAPH AESTHETICS'''

    plt.xticks(color="white")
    l = ax[0].legend(facecolor="black", labelcolor="white")
    plt.setp(l.texts, family="Formula1", size=axis_fontsize)
    ax[0].get_xaxis().set_ticklabels([])

    if not brake_and_throttle:
        if several_laps:
            ax[1].get_xaxis().set_ticklabels([])
            ax[2].set_xlabel("Distance [metres]", fontname="Formula1",
                             fontsize=axis_fontsize, color="white")
            y_labels = ["Speed [km/h]", f"{middle_plot} [%]",
                        f"Delta to {all_laps[0]["driver"]} [s]"]
        else:
            ax[1].set_xlabel("Distance [metres]", fontname="Formula1",
                             fontsize=axis_fontsize, color="white")
            y_labels = ["Speed [km/h]", f"{middle_plot} [%]"]
    else:
        if several_laps:
            ax[1].get_xaxis().set_ticklabels([])
            ax[2].get_xaxis().set_ticklabels([])
            ax[3].set_xlabel("Distance [metres]", fontname="Formula1",
                             fontsize=axis_fontsize, color="white")
            y_labels = ["Speed [km/h]", "Throttle [%]", "Brake [%]",
                        f"Delta to {all_laps[0]["driver"]} [s]"]
        else:
            ax[1].set_xlabel("Distance [metres]", fontname="Formula1",
                             fontsize=axis_fontsize, color="white")
            y_labels = ["Speed [km/h]", "Throttle [%]", "Brake [%]"]

    # how many charts there are (i.e. 2/3: several laps = true/false)
    range_ = len(y_labels)

    for n in range(0, range_):
        ax[n].tick_params(labelsize=tick_fontsize)
        ax[n].patch.set_facecolor("black")
        ax[n].set_ylabel(y_labels[n], fontname="Formula1",
                         fontsize=axis_fontsize, color="white")
        ax[n].xaxis.set_minor_locator(FixedLocator(minor_tick_locations))
        ax[n].grid(which="minor", linestyle="-",
                   linewidth=3, color="#333333")
        ax[n].grid(which="minor", axis="y", linewidth=0)
        ax[n].grid(which="major", axis="x", linestyle="-", linewidth=2)
        ax[n].tick_params(axis="y", colors="white")

        # SETTING X LIMITS FOR GRAPH
        if xlim is not None:
            ax[n].set(xlim=xlim)

    # SETTING PARAMETERS FOR DELTA AXIS
    if brake_and_throttle:
        delta_axis = 3
    else:
        delta_axis = 2

    if several_laps:
        if delta_ylim is not None:
            ax[delta_axis].set(ylim=delta_ylim)
        else:
            # will curtail y axis at the tightest boundary within -3 to +3.
            ax[delta_axis].set(ylim=(
                max(min(delta_time) - 0.2, -3),
                min(max(delta_time) + 0.2, 3))
            )

    # VLINES
    if vlines is not None:
        for line in vlines:
            start = line["start"]
            stop = line["stop"]
            label_x = (stop - start) / 2 + start

            for n in range(0, range_):
                ax[n].axvline(x=line["start"], ymin=0, ymax=1, color=line["colour"],
                              linestyle="dotted")
                ax[n].axvline(x=line["stop"], ymin=0, ymax=1, color=line["colour"],
                              linestyle="dotted")
            ax[0].text(
                x=label_x, y=max_speed,
                s=line["label"], color=line["colour"],
                fontsize=16, fontname="Formula1", horizontalalignment='center'
            )

    main_lap_time = strftimedelta(
        all_laps[0]["lap info"]["LapTime"], "%m:%s.%ms")

    # DOING THE TITLE
    title = f"Telemetry Comparison, {data.event["EventName"]} {data.event.year}\n{
        all_laps[0]["driver"]} Lap {all_laps[0]["lap number"]}: {main_lap_time}"

    if several_laps:
        title = title + "\n"

        # if tel from 2+ laps, need to include lap no.
        if len(all_lap_numbers) > 0:
            for lap in all_laps[1:]:
                try:
                    diff = strftimedelta(
                        lap["lap info"]["LapTime"] -
                        all_laps[0]["lap info"]["LapTime"], "%s.%ms"
                    )

                    if "-" not in diff:
                        sign = "+"
                    else:
                        sign = ""

                    title = title + f"{lap["driver"]
                                       } Lap {lap["lap number"]}: {sign}{diff} "
                except ValueError:
                    pass
        else:
            for lap in all_laps[1:]:
                diff = strftimedelta(
                    lap["lap info"]["LapTime"] -
                    all_laps[0]["lap info"]["LapTime"], "%s.%ms"
                )

                if "-" not in diff:
                    sign = "+"
                else:
                    sign = ""

                title = title + f"{lap["driver"]}: {sign}{diff}"
    plt.suptitle(title, fontsize=16, fontname="Formula1", color="white")

    fig.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.05)
    plt.show()


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def position_changes(data):
    axis_fontsize = 20
    tick_fontsize = 12

    plt.figure(figsize=(16, 8), dpi=120, facecolor="#0F0F0F")
    ax = plt.gca()

    team_list = list()
    LAPS = data.laps

    for drv in data.drivers:

        driver_df = LAPS[LAPS["DriverNumber"] == drv]
        driver = driver_df["Driver"].unique()[0]

        team = drivers[driver]["Team"]
        team_colour = team_colours[team]

        if team in team_list:

            if team == "McLaren":
                team_colour = "#FF4D01"
            elif team == "Mercedes":
                team_colour = "#19464C"
            else:
                team_colour = ImageColor.getcolor(team_colour, "RGB")
                team_colour = tuple([(0.5 * x)/255 for x in team_colour])

            linestyle = "-"
            dash_capstyle = "round"
        else:
            linestyle = "-"
            dash_capstyle = None
            team_list.append(team)

        driver_laps = LAPS.pick_driver(drv)

        ax.plot(
            driver_laps['LapNumber'], driver_laps['Position'],
            label=driver, color=team_colour, linewidth=4, linestyle=linestyle,
            dash_capstyle=dash_capstyle
        )

    ax.set_ylim([20.5, 0.5])
    ax.set_xlim([2, LAPS["LapNumber"].max()])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.patch.set_facecolor("black")
    ax.grid(which="major", linewidth=2, color="white", linestyle="dotted")

    plt.yticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.xticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.xlabel(
        "Lap Number",
        fontsize=axis_fontsize, color="white", fontname="Formula1"
    )
    plt.ylabel(
        "Position",
        fontsize=axis_fontsize, color="white", fontname="Formula1"
    )

    legend = ax.legend(bbox_to_anchor=(1.0, 1.02))
    for text in legend.get_texts():
        text.set_color("white")
        text.set_font("Formula1")
        text.set_size(axis_fontsize)

    plt.suptitle(
        f"Driver Position during the {
            data.event.year} {data.event["EventName"]}",
        fontname="Formula1", color="white", fontsize=axis_fontsize
    )

    plt.tight_layout()

    plt.show()


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def speed_trap_distribution(data, starters: int = 20, ylim: Tuple[int, int] = None, include_drs: bool = True):

    LAPS = data.laps

    if include_drs is False:
        def find_speed_trap(round_no):
            for race_name, race_info in races.items():
                if race_info["Round"] == round_no:
                    speed_trap_between = race_info.get("SpeedTrapBetween")
                    return speed_trap_between[0], speed_trap_between[1]

        round_no = data.event.RoundNumber
        x, y = find_speed_trap(round_no)

        drs_on = [10, 12, 14]
        for index, row in LAPS.iterrows():
            df = row.get_car_data().add_distance()
            df = df[(df["Distance"] >= x) & (df["Distance"] <= y)]
            contains_values = (df["DRS"].isin(drs_on)).any()
            LAPS.at[index, "SpeedSTDRS"] = contains_values

        non_drs_laps = LAPS[LAPS["SpeedSTDRS"] == False]
        racing_laps = non_drs_laps.pick_quicklaps()

    else:
        racing_laps = LAPS.pick_quicklaps()

    drivers_in_race = LAPS["Driver"].unique()

    finishing_order = [data.get_driver(i)["Abbreviation"]
                       for i in drivers_in_race]
    finishing_order = finishing_order[:starters]

    driver_colours = {}
    for driver in finishing_order:
        team = drivers[driver]["Team"]
        colour = team_colours.get(team, "Unknown")
        driver_colours[driver] = colour

    axis_fontsize = 20
    tick_fontsize = 12

    stint_column = racing_laps["Stint"]
    no_stints = int(stint_column.max() - stint_column.min() + 1)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    sns.violinplot(data=racing_laps,
                   x="Driver",
                   y="SpeedST",
                   hue="Driver",
                   inner=None,
                   density_norm="area",
                   order=finishing_order,
                   palette=driver_colours
                   )

    sns.swarmplot(data=racing_laps,
                  x="Driver",
                  y="SpeedST",
                  order=finishing_order,
                  hue="Stint",
                  palette=sns.color_palette("coolwarm", as_cmap=True),
                  size=2.5,
                  )

    if ylim != None:
        ax.set_ylim(ylim)

    ax.yaxis.grid(True, which="minor", linestyle="")
    ax.xaxis.grid(True, which="minor", linestyle="")

    plt.suptitle(
        f"{data.event.year} {data.event["EventName"]} Speed Trap Distribution",
        fontsize=16, fontname="Formula1", color="white"
    )
    sns.despine(left=True, bottom=True)

    plt.yticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.xticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.ylabel(
        "Speed trap speed [km/h]",
        fontsize=axis_fontsize, color="white", fontname="Formula1"
    )
    plt.xlabel("")

    l = ax.legend(facecolor="black", labelcolor="white", title="Stint No.")
    plt.setp(l.texts, family="Formula1", size=tick_fontsize)
    plt.setp(l.get_title(), family="Formula1",
             color="white", size=tick_fontsize)

    plt.tight_layout()
    plt.show()


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def speed_trap_table(data, include_drs: bool = True):
    LAPS = data.laps
    round_no = data.event.RoundNumber

    if include_drs is False:
        def find_speed_trap(round_no):
            for race_name, race_info in races.items():
                if race_info["Round"] == round_no:
                    speed_trap_between = race_info.get("SpeedTrapBetween")
                    return speed_trap_between[0], speed_trap_between[1]

        try:
            x, y = find_speed_trap(round_no)
        except TypeError:
            raise ValueError(
                "Please input 'SpeedTrapBetween' data in season_info.json.")

        drs_on = [10, 12, 14]
        for index, row in LAPS.iterrows():
            df = row.get_car_data().add_distance()
            df = df[(df["Distance"] >= x) & (df["Distance"] <= y)]
            contains_values = (df["DRS"].isin(drs_on)).any()
            LAPS.at[index, "SpeedSTDRS"] = contains_values

        non_drs_laps = LAPS[LAPS["SpeedSTDRS"] == False]
        racing_laps = non_drs_laps.pick_quicklaps()

    else:
        racing_laps = LAPS.pick_quicklaps()

    df = racing_laps[["Driver", "SpeedST"]]
    grouped_data = df.groupby('Driver').agg(
        {'SpeedST': 'mean', 'Driver': 'count'})
    grouped_data.columns = ['AverageSTSpeed', 'LapCount']
    grouped_data["Round"] = round_no

    return grouped_data


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def custom_scatter(data, drivers_in_graph, x_data: str, y_data: str, z_data: str = None, z_reverse: bool = False, xlim: Tuple[int, int] = None, ylim: Tuple[int, int] = None, title: bool = True):
    font = font_manager.FontProperties(
        family="Formula1", weight="normal", style="normal", size=12)
    axis_fontsize = 16
    tick_fontsize = 12

    def lap_average_calculator(row, averaging):
        df = row.get_car_data()

        df["Time"] = pd.to_timedelta(df["Time"])
        df["Duration"] = df["Time"].diff().fillna(pd.Timedelta(seconds=0))
        weighted_sum = (df[averaging] *
                        df["Duration"].dt.total_seconds()).sum()
        total_duration = df["Duration"].sum().total_seconds()
        weighted_avg = weighted_sum / total_duration

        return weighted_avg

    LAPS = data.laps

    teams_in_graph = list()  # whether the line needs to be dashed or not
    handles = list()  # handles for legend

    if z_data is not None:
        if z_reverse is False:
            small_marker = "low"
            large_marker = "high"
        else:
            small_marker = "high"
            large_marker = "low"

        handle = mlines.Line2D(
            [], [], linestyle="None", marker="o", markerfacecolor="#372952",
            markeredgecolor="None", markersize=6,
            label=f"{variables[z_data]["Axis"]} ({small_marker})"
        )
        handles.append(handle)
        handle = mlines.Line2D(
            [], [], linestyle="None", marker="o", markerfacecolor="#59CBAC",
            markeredgecolor="None", markersize=12,
            label=f"{variables[z_data]["Axis"]} ({large_marker})"
        )
        handles.append(handle)

    plt.figure(figsize=(16, 8), dpi=120)
    ax = plt.gca()

    with sns.axes_style():
        for driver in drivers_in_graph:

            # GETTING INITIAL DATA
            team = drivers[driver]["Team"]
            colour = team_colours[team]
            laps = LAPS[LAPS["Driver"] == driver]
            laps = laps.pick_quicklaps()
            laps = laps.pick_track_status(how="equals", status="1")

            lap_average_data = {"ThrottleAvg": "Throttle",
                                "RPMAvg": "RPM", "GearAvg": "nGear"}

            for var_name, ff1_name in lap_average_data.items():
                if var_name in [x_data, y_data, z_data]:
                    laps[var_name] = laps.apply(
                        lap_average_calculator,
                        axis=1, averaging=ff1_name
                    )

            if team not in teams_in_graph:
                marker = "o"
            else:
                marker = "^"

            handle = mlines.Line2D(
                [], [], linestyle="None", marker=marker, markerfacecolor="None",
                markeredgecolor=colour, markersize=10, markeredgewidth=2,
                label=driver
            )

            teams_in_graph.append(team)
            handles.append(handle)

            if z_reverse is False:
                scatter_sizes = (30, 120)
            else:
                scatter_sizes = (120, 30)

            if z_data is not None:
                ax = sns.scatterplot(
                    data=laps, x=x_data, y=y_data,
                    hue=z_data, size=z_data, sizes=scatter_sizes, palette="mako",
                    marker=marker,
                    linewidth=2, edgecolors=colour
                )
            else:
                ax = sns.scatterplot(
                    data=laps, x=x_data, y=y_data,
                    marker=marker, palette=[colour], hue=y_data,
                    linewidth=4, edgecolors=colour
                )

    l = ax.legend(handles=handles, prop=font)
    for text in l.get_texts():
        text.set_color("white")

    sns.set_theme(rc={"axes.facecolor": "black",
                  "figure.facecolor": "#0F0F0F"})
    plt.yticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.xticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.xlabel(variables[x_data]["Axis"], fontsize=axis_fontsize,
               color="white", fontname="Formula1")
    plt.ylabel(variables[y_data]["Axis"], fontsize=axis_fontsize,
               color="white", fontname="Formula1")

    if xlim is not None and ylim is not None:
        ax.set(
            ylim=ylim,
            xlim=xlim
        )
    elif xlim is not None:
        ax.set(xlim=xlim)
    elif ylim is not None:
        ax.set(ylim=ylim)

    # TITLE
    if title is True:
        title_str = (f"{variables[y_data]["Title"]} against {variables[x_data]["Title"]}\n"
                     f"{data.event.year} {data.event['EventName']}\n")
        for driver in drivers_in_graph:
            title_str = title_str + f"{drivers[driver]['Full Name']}, "
        title_str = title_str[:-2]
        plt.suptitle(title_str, fontsize=16,
                     fontname="Formula1", color="white")

    plt.show()


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def lap_average(laps, variable):
    def lap_average_calculator(row, averaging):
        df = row.get_car_data()

        df["Time"] = pd.to_timedelta(df["Time"])
        df["Duration"] = df["Time"].diff().fillna(pd.Timedelta(seconds=0))
        weighted_sum = (df[averaging] *
                        df["Duration"].dt.total_seconds()).sum()
        total_duration = df["Duration"].sum().total_seconds()
        weighted_avg = weighted_sum / total_duration

        return weighted_avg

    ff1_name = variables[variable]["FastF1Name"]

    laps[variable] = laps.apply(
        lap_average_calculator, axis=1, averaging=ff1_name
    )

    return laps


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def low_speed_corner_analyser(data, low_speed_corners: dict, search_range: int = 150):
    '''
    low_speed_corners = {
        "Turn 3": 1060,
        "Turn 11": 4068,
        "Turn 13": 4576
    }
    '''

    LAPS = data.laps

    turn_names = list(low_speed_corners.keys())
    troughs = list(low_speed_corners.values())

    min_speeds_list = []

    for driver in drivers.keys():
        driver_laps = LAPS.pick_driver(driver)
        driver_laps = driver_laps.pick_quicklaps()

        driver_min_speeds = []

        for index, row in driver_laps.iterrows():
            df = row.get_car_data().add_distance()
            df = df[["Speed", "Distance"]].copy()

            trough_min_speeds = []

            for trough in troughs:
                min_distance = trough - search_range
                max_distance = trough + search_range

                min_speed = df[(df["Distance"] >= min_distance) & (
                    df["Distance"] <= max_distance)]["Speed"].min()
                trough_min_speeds.append(min_speed)

            driver_min_speeds.append(trough_min_speeds)

        driver_min_speeds_df = pd.DataFrame(
            driver_min_speeds,
            columns=[f"Trough_{i+1}" for i in range(len(troughs))]
        ).reset_index(drop=True)
        driver_min_speeds_df["Driver"] = driver
        driver_min_speeds_df['Count'] = len(driver_min_speeds_df)

        min_speeds_list.append(driver_min_speeds_df)

    # Concatenate the DataFrames for all drivers
    min_speeds = pd.concat(min_speeds_list, ignore_index=True)

    # Calculate the average minimum speed for each driver
    average_min_speeds_per_driver = min_speeds.groupby("Driver").mean()

    trough_names = [f"Trough_{i+1}" for i in range(len(troughs))]

    for i in range(len(trough_names)):
        trough_name = trough_names[i]
        turn_name = turn_names[i]

        average_min_speeds_per_driver = average_min_speeds_per_driver.rename(columns={
            trough_name: turn_name,
        })

    average_min_speeds_per_driver["Average"] = average_min_speeds_per_driver.mean(
        axis=1)

    max_avg_row = average_min_speeds_per_driver[
        average_min_speeds_per_driver['Average'] == average_min_speeds_per_driver['Average'].max(
        )
    ]

    # Normalize to have all values equal to 1
    normalised_lowspeed = average_min_speeds_per_driver.div(
        max_avg_row.values[0], axis=1)
    normalised_lowspeed["Count"] = average_min_speeds_per_driver["Count"]

    return average_min_speeds_per_driver, normalised_lowspeed


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def normalised_lowspeed_bar(normalised_lowspeed, all_corners: bool = False, xlim=[0.9, 1]):

    sorted_df = normalised_lowspeed.sort_values(by='Average', ascending=False)

    if all_corners:
        # Transpose the DataFrame for easy plotting
        df_transposed = sorted_df.reset_index().melt(
            id_vars='Driver',
            var_name='Turn',
            value_name='Normalised Performance'
        )

        fig = px.bar(
            df_transposed,
            x="Driver", y="Normalised Performance", color="Turn",
            barmode="group",
            template="plotly_dark",

        )
        fig.update_yaxes(range=[0.86, 1])
        fig.show()

    else:
        turn_numbers = sorted_df.columns.to_list()[:-1]
        turn_numbers = [turn.split(" ")[1] for turn in turn_numbers]
        if len(turn_numbers) == 1:
            number_string = turn_numbers[0]
        else:
            number_string = ", ".join(
                turn_numbers[:-1]) + ", and " + turn_numbers[-1]

        color_map = {driver: drivers[driver]['Colour']
                     for driver in sorted_df.index}
        fig = px.bar(
            sorted_df,
            x=sorted_df.index, y="Average",
            color=sorted_df.index,
            color_discrete_map=color_map,
            template="plotly_dark",
            width=1600, height=600,
            title="Normalised Minimum Speed in Low Speed Corners"
        )

        fig.update_yaxes(range=xlim, title_text='Normalised Minimum Speed')
        fig.update_traces(showlegend=False)
        fig.update_xaxes(title="")

        custom_font_family = "Open Sans, sans-serif"
        fig.update_layout(
            font=dict(size=18),
            annotations=[
                dict(
                    x=-0.0035, y=1.1,
                    xref="paper", yref="paper",
                    text=f"Normalised from the Race Average Minimum Speeds in Turns {
                        number_string}",
                    font=dict(size=18, family=custom_font_family),
                    showarrow=False,
                    align="left"
                )
            ]
        )
        fig.show()


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def gap_to_x_gen(laps, relative_driver: str = None):
    gap_to_leader_list = []

    for lap_number in range(1, (int(laps["LapNumber"].max()) + 1)):
        lap_data = laps[laps["LapNumber"] == lap_number].copy()

        if relative_driver is None:
            # if relative driver is None, we're looking at the gap to the leader
            relative_lap = lap_data[lap_data["Position"] == 1]
            var_name = "GapToLeader"
        else:
            relative_lap = lap_data[lap_data["Driver"] == relative_driver]
            var_name = f"GapTo{relative_driver}"

        relative_time = relative_lap["Time"]

        lap_data.loc[:, var_name] = lap_data["Time"].apply(
            lambda x: x - relative_time)

        gap_to_leader_list.append(lap_data)

    gap_leader = pd.concat(gap_to_leader_list)

    gap_leader[var_name] = gap_leader[var_name].to_numpy(dtype=np.timedelta64)

    return gap_leader


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''


def gap_to_x_graph(data, drivers_in_graph=[], ylim=None, sc_laps=[], relative_driver: str = None):

    if relative_driver is None:
        var_name = "GapToLeader"
        axis_label = "Leader"
    else:
        var_name = f"GapTo{relative_driver}"
        axis_label = relative_driver

    laps = data.laps

    LAPS = gap_to_x_gen(laps, relative_driver)

    axis_fontsize = 20
    tick_fontsize = 12

    plt.figure(figsize=(16, 8), dpi=120, facecolor="#0F0F0F")
    ax = plt.gca()

    team_list = list()

    for drv in data.drivers:

        driver_df = LAPS[LAPS["DriverNumber"] == drv]
        driver = driver_df["Driver"].unique()[0]

        if len(drivers_in_graph) != 0:
            if driver not in drivers_in_graph:
                continue

        team = drivers[driver]["Team"]
        team_colour = team_colours[team]

        if team in team_list:

            if team == "McLaren":
                team_colour = "#FF4D01"
            elif team == "Mercedes":
                team_colour = "#19464C"
            else:
                team_colour = ImageColor.getcolor(team_colour, "RGB")
                team_colour = tuple([(0.5 * x)/255 for x in team_colour])

        else:
            team_list.append(team)

        driver_laps = LAPS.pick_driver(drv)

        ax.plot(
            driver_laps['LapNumber'], driver_laps[var_name],
            label=driver, color=team_colour, linewidth=3, linestyle="-"
        )

    max_time = LAPS[var_name].max()

    for sc in sc_laps:
        start_lap, end_lap = sc
        plt.axvline(
            x=start_lap, ymin=0, ymax=1,
            linestyle="dotted", color=tyre_colours["medium"]
        )
        plt.axvline(
            x=end_lap, ymin=0, ymax=1,
            linestyle="dotted", color=tyre_colours["medium"]
        )

        half_way_sc = (end_lap - start_lap) / 2 + start_lap

        if ylim is None:
            plt.text(
                x=half_way_sc, y=max_time * 0.9,
                s="SC", fontsize=20, fontname="Formula1",
                horizontalalignment='center', color=tyre_colours["medium"]
            )
        else:
            plt.text(
                x=half_way_sc, y=ylim[1] * 0.9,
                s="SC", fontsize=20, fontname="Formula1",
                horizontalalignment='center', color=tyre_colours["medium"]
            )

    if ylim != None:
        ax.set_ylim(ylim)

    ax.patch.set_facecolor("black")
    ax.grid(which="major", linewidth=2, color="#333333", linestyle="dotted")

    plt.yticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.xticks(color="white", fontsize=tick_fontsize, fontname="Formula1")
    plt.xlabel(
        "Lap Number",
        fontsize=axis_fontsize, color="white", fontname="Formula1"
    )
    plt.ylabel(
        f"Gap to {axis_label}",
        fontsize=axis_fontsize, color="white", fontname="Formula1"
    )

    def timeTicks(x, pos):

        def timedeltaToString(td):
            total, prefix = td.total_seconds(), ""
            if total < 0:  # catch negative timedelta
                total *= -1
                prefix = "-"
            hours, remainder = divmod(total, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{prefix}{int(minutes):02d}:{seconds:02.0f}"

        d = timedelta(days=x)
        d = timedeltaToString(d)

        return str(d)
    formatter = matplotlib.ticker.FuncFormatter(timeTicks)
    ax.yaxis.set_major_formatter(formatter)

    legend = ax.legend(bbox_to_anchor=(1.0, 1.02))
    for text in legend.get_texts():
        text.set_color("white")
        text.set_font("Formula1")
        text.set_size(axis_fontsize)

    plt.suptitle(
        f"Gap to {axis_label} during the {
            data.event.year} {data.event["EventName"]}",
        fontname="Formula1", color="white", fontsize=axis_fontsize
    )

    plt.tight_layout()

    plt.show()


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''
