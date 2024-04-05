import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator

import seaborn as sns
from sklearn.linear_model import LinearRegression

from timple.timedelta import strftimedelta
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any

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

    all_laps_by_team = initial_setup(data)["team_laps"]

    drivers_in_graph = drivers_in_graph

    axis_fontsize = 16
    tick_fontsize = 12

    teams_in_graph = list()
    handles = list()
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
            driver_laps = all_laps_by_team[current_team][all_laps_by_team[current_team]
                                                         ["Driver"] == driver].reset_index()
            driver_laps = driver_laps[(driver_laps['PitInTime'].isnull()) & (
                driver_laps['PitOutTime'].isnull())]  # removing box laps

            # if two drivers are of the same team, the dots in the scatter are squares
            # also gives them the legend handle with the correct colour and name
            if current_team not in teams_in_graph:
                marker = "o"
                best_fit_line_style = "-"
                dot_size = 65
                handle = mlines.Line2D(
                    [], [], linestyle="-", color=team_colour, marker=marker, markersize=12, label=driver)
            else:
                marker = "*"
                best_fit_line_style = "--"
                dot_size = 100
                handle = mlines.Line2D(
                    [], [], linestyle="--", color=team_colour, marker=marker, markersize=12, label=driver)

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
                            min_time_tyre.seconds == min_time.seconds and min_time_tyre.microseconds < min_time.microseconds
                        ):
                            min_time = min_time_tyre

                    if not np.isnat(max_time_tyre.to_numpy()):
                        if max_time == None:
                            max_time = max_time_tyre
                        elif (
                            max_time_tyre.seconds > max_time.seconds or
                            max_time_tyre.seconds == max_time.seconds and max_time_tyre.microseconds > max_time.microseconds
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
                s="SC", fontsize=16, fontname="Formula1", horizontalalignment='center', color=tyre_colours["medium"]
            )
        else:
            plt.text(
                x=half_way_sc, y=ylim[1] * 0.998,
                s="SC", fontsize=16, fontname="Formula1", horizontalalignment='center', color=tyre_colours["medium"]
            )

    # LEGEND
    handle = mlines.Line2D(
        [], [], linestyle="None",
        marker="o", markerfacecolor="None", markeredgecolor=tyre_colours["medium"], markersize=12,
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


def compare_telemetry(data, driver_info: List[Dict[str, Any]], vlines: List[Dict[str, Any]] = None, xlim: Tuple[int, int] = None, delta_ylim: Tuple[float, float] = None):
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

    axis_fontsize = 18
    tick_fontsize = 14

    line_styles = ["solid", "dotted", "dashdot", "dashed"]

    all_laps = list()
    all_teams = list()
    all_lap_numbers = list()

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
            all_teams.append(team)
        else:
            linestyle = "-"

        all_laps.append({
            "driver": driver,
            "team": team,
            "colour": team_colour,
            "linestyle": linestyle,
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
    if not several_laps:  # if only one driver cannot do delta chart
        fig, ax = plt.subplots(nrows=2, height_ratios=[3, 1], figsize=(
            16, 12), dpi=120, facecolor="#0F0F0F")
    else:
        fig, ax = plt.subplots(nrows=3, height_ratios=[3, 1, 1], figsize=(
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

        ax[0].plot(lap["lap tel"]["Distance"], lap["lap tel"]["Speed"], color=lap["colour"],
                   label=label, linestyle=lap["linestyle"], linewidth=2)
        ax[1].plot(lap["lap tel"]["Distance"], lap["lap tel"]["Throttle"], color=lap["colour"],
                   label=label, linestyle=lap["linestyle"], linewidth=2)

        if len(all_laps) > 1 and lap != all_laps[0]:
            ax[2].plot(lap["ref tel"]["Distance"], lap["delta time"],
                       color=lap["colour"], linestyle=lap["linestyle"], linewidth=2)

    '''GRAPH AESTHETICS'''

    plt.xticks(color="white")
    l = ax[0].legend(facecolor="black", labelcolor="white")
    plt.setp(l.texts, family="Formula1", size=axis_fontsize)
    ax[0].get_xaxis().set_ticklabels([])

    if several_laps:
        ax[1].get_xaxis().set_ticklabels([])
        ax[2].set_xlabel("Distance [metres]", fontname="Formula1",
                         fontsize=axis_fontsize, color="white")
        y_labels = ["Speed [km/h]", "Throttle [%]",
                    f"Delta to {all_laps[0]["driver"]} [s]"]
    else:
        ax[1].set_xlabel("Distance [metres]", fontname="Formula1",
                         fontsize=axis_fontsize, color="white")
        y_labels = ["Speed [km/h]", "Throttle [%]"]

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
    if several_laps:
        if delta_ylim is not None:
            ax[2].set(ylim=delta_ylim)
        else:
            # will curtail y axis at the tightest boundary within -3 to +3.
            ax[2].set(ylim=(
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
                        lap["lap info"]["LapTime"] - all_laps[0]["lap info"]["LapTime"], "%s.%ms")

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
                    lap["lap info"]["LapTime"] - all_laps[0]["lap info"]["LapTime"], "%s.%ms")

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

    plt.figure(figsize=(16, 8), dpi=120)
    ax = plt.gca()

    team_list = list()
    LAPS = data.laps

    for drv in data.drivers:

        driver_df = LAPS[LAPS["DriverNumber"] == drv]
        driver = driver_df["Driver"].unique()[0]

        team = drivers[driver]["Team"]
        team_colour = team_colours[team]

        if team in team_list:
            linestyle = "dotted"
            dash_capstyle = "round"
        else:
            linestyle = "-"
            dash_capstyle = None
            team_list.append(team)

        drv_laps = LAPS.pick_driver(drv)

        ax.plot(
            drv_laps['LapNumber'], drv_laps['Position'],
            label=driver, color=team_colour, linewidth=4, linestyle=linestyle, dash_capstyle=dash_capstyle
        )

    ax.set_ylim([20.5, 0.5])

    ax.set_yticks([1, 5, 10, 15, 20])

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


def corner_trace(data, driver_info: List[Dict[str, Any]], corner: int = None, zoom: int = 10):
    '''
    Creates a trace of the track, also giving you the ability to zoom in on any corner. Need to provide driver info.
    Mostly useless, as I thought it gave the line a driver drove, whereas it's actually just a pre-scripted series of X/Y coords.

    driver_info = [
        {"driver": str, "lap": int},
    ]
    '''

    axis_fontsize = 20

    LAPS = data.laps
    plt.figure(figsize=(16, 8), dpi=120)
    ax = plt.gca()

    circuit_info = data.get_circuit_info()
    corner_coords = [
        {
            "corner": index + 1,
            "x": round(row["X"]),
            "y": round(row["Y"]),
            "distance": round(row["Distance"])
        }
        for index, row in circuit_info.corners.iterrows()
    ]

    for dict_ in driver_info:

        driver = dict_["driver"]
        lap = dict_["lap"]

        lap = LAPS.pick_driver(driver).pick_lap(lap).iloc[0]
        pos_data = lap.get_pos_data()

        try:
            colour_override = dict_["colour"]
        except KeyError:
            colour_override = None

        if colour_override is not None:
            colour = colour_override
        else:
            colour = team_colours[drivers[driver]["Team"]]

        plt.plot(
            pos_data["X"], pos_data["Y"],
            color=colour, linewidth=2
        )

    # REMOVING GRID AND AXIS LABELS
    ax.grid(which="major", linewidth=0)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    # DOING THE CORNER ZOOM
    if corner is not None:
        y_lim = ax.get_ylim()
        y_span = y_lim[1] - y_lim[0]
        y_span_zoomed = y_span / zoom

        x_lim = ax.get_xlim()
        x_span = x_lim[1] - x_lim[0]
        x_span_zoomed = x_span / zoom

        corner_info = corner_coords[corner - 1]

        y_bound = (
            corner_info["y"] - y_span_zoomed / 2,
            corner_info["y"] + y_span_zoomed / 2,
        )

        x_bound = (
            corner_info["x"] - x_span_zoomed / 2,
            corner_info["x"] + x_span_zoomed / 2,
        )

        ax.set(
            ylim=y_bound,
            xlim=x_bound
        )

    # TITLE
    title = f"Driver Traces during the {
        data.event.year} {data.event["EventName"]}\n"
    for dict_ in driver_info:  # adds driver initials
        driver = dict_["driver"]
        title += f"{driver}, "

    # if a corner to zoom on is specified, adds the corner number to the title
    if corner is not None:
        title += f"Turn {corner}"
    else:
        title = title[:-2]

    plt.suptitle(
        title, fontname="Formula1", color="white", fontsize=axis_fontsize
    )


'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''
