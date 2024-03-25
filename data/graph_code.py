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


def qualifying_telemetry_data(data, driver1, driver2):
    '''Takes two drivers and compares their fastest qualifying laps'''

    LAPS = data.laps
    q1, q2, q3 = LAPS.split_qualifying_sessions()
    circuit_info = data.get_circuit_info()

    axis_fontsize = 18
    tick_fontsize = 14

    team1 = drivers[driver1]["Team"]
    team2 = drivers[driver2]["Team"]
    team1_colour = team_colours[team1]
    team2_colour = team_colours[team2]

    lap1 = data.pick_driver(driver1).pick_fastest()
    lap2 = data.pick_driver(driver2).pick_fastest()
    lap1_tel = lap1.get_car_data().add_distance()
    lap2_tel = lap2.get_car_data().add_distance()

    delta_time, ref_tel, compare_tel = utils.delta_time(lap1, lap2)

    v_min = lap1_tel["Speed"].min()
    v_max = lap1_tel["Speed"].max()

    '''GRAPH START'''

    fig, ax = plt.subplots(nrows=3, height_ratios=[3, 1, 1], figsize=(
        16, 12), dpi=120, facecolor="#0F0F0F")

    # CORNER LINES
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner["Number"]}{corner["Letter"]}"
        ax[0].text(corner["Distance"], v_max+20, txt, va="center_baseline",
                   ha="center", size="large", fontname="Formula1", color="white")

    # CORNER MINOR AXIS LINE LOCATIONS
    corner_location = circuit_info.corners["Distance"]
    minor_tick_locations = list()
    for corner in corner_location:
        minor_tick_locations.append(corner)

    ax[0].plot(lap1_tel["Distance"], lap1_tel["Speed"],
               color=team1_colour, label=driver1, linewidth=2)
    ax[0].plot(lap2_tel["Distance"], lap2_tel["Speed"],
               color=team2_colour, label=driver2, linewidth=2)

    ax[1].plot(lap1_tel["Distance"], lap1_tel["Throttle"],
               color=team1_colour, label=driver1, linewidth=2)
    ax[1].plot(lap2_tel["Distance"], lap2_tel["Throttle"],
               color=team2_colour, label=driver2, linewidth=2)

    ax[2].plot(ref_tel["Distance"], delta_time, color="white", linewidth=3)
    ax[2].axhline(y=0, color="gray")

    l = ax[0].legend(facecolor="black", labelcolor="white")
    plt.setp(l.texts, family="Formula1", size=axis_fontsize)

    ax[0].get_xaxis().set_ticklabels([])
    ax[1].get_xaxis().set_ticklabels([])
    ax[2].set_xlabel("Distance [metres]", fontname="Formula1",
                     fontsize=axis_fontsize)

    y_labels = ["Speed [km/h]", "Throttle [%]", "Delta [s]"]
    for n in range(0, 3):
        ax[n].tick_params(labelsize=tick_fontsize)
        ax[n].patch.set_facecolor("black")
        ax[n].set_ylabel(y_labels[n], fontname="Formula1",
                         fontsize=axis_fontsize)

        ax[n].xaxis.set_minor_locator(FixedLocator(minor_tick_locations))
        ax[n].grid(which="minor", linestyle="-", linewidth=3, color="gray")
        ax[n].grid(which="minor", axis="y", linewidth=0)
        ax[n].grid(which="major", axis="x", linestyle="-", linewidth=2)

    fastest_lap_time = strftimedelta(lap1["LapTime"], "%m:%s.%ms")
    diff = strftimedelta(lap2["LapTime"] - lap1["LapTime"], "%s.%ms")
    plt.suptitle(f"Fastest Lap Comparison, {data.event["EventName"]} {data.event.year}\n{driver1}: {fastest_lap_time}, {driver2}: +{diff}s",
                 fontsize=20, fontname="Formula1", color="white")

    fig.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.05)
    plt.show()


def race_compare_lap_telemetry(data, driver1: str, driver2: str, lap_number: int, driver1_overwrite_colour=None, driver2_overwrite_colour=None, dotswap=False):
    '''
    Shows the telemetry data for two drivers on any specific lap. Inputs are "driver1" and "driver2" as the driver initials, and the lap number to compare.
    Also takes overwrite colours for either driver. By default "None" and so drivers will get their team colour.
    '''

    LAPS = data.laps
    circuit_info = data.get_circuit_info()

    axis_fontsize = 18
    tick_fontsize = 14

    team1 = drivers[driver1]["Team"]
    team2 = drivers[driver2]["Team"]
    team1_colour = team_colours[team1]
    team2_colour = team_colours[team2]

    if driver1_overwrite_colour != None:
        team1_colour = driver1_overwrite_colour

    if driver2_overwrite_colour != None:
        team2_colour = driver2_overwrite_colour

    lap1 = LAPS.pick_driver(driver1).pick_lap(lap_number).iloc[0]
    lap2 = LAPS.pick_driver(driver2).pick_lap(lap_number).iloc[0]

    lap1_tel = lap1.get_car_data().add_distance()
    lap2_tel = lap2.get_car_data().add_distance()

    delta_time, ref_tel, compare_tel = utils.delta_time(lap1, lap2)

    v_max = lap1_tel["Speed"].max()

    '''GRAPH START'''

    fig, ax = plt.subplots(nrows=3, height_ratios=[3, 1, 1], figsize=(
        16, 12), dpi=120, facecolor="#0F0F0F")

    # DRIVER LINESTYLES. IF OF SAME TEAM, SECOND IS DASHED, IF THERE IS NO OVERWRITE COLOUR FOR DRIVER 2
    if dotswap == False and driver2_overwrite_colour == None and team1 == team2:
        driver1_linestyle = "-"
        driver2_linestyle = "dotted"
    elif dotswap == True and driver2_overwrite_colour == None and team1 == team2:
        driver1_linestyle = "dotted"
        driver2_linestyle = "-"
    else:
        driver1_linestyle = "-"
        driver2_linestyle = "-"

    # CORNER LINES
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner["Number"]}{corner["Letter"]}"
        ax[0].text(corner["Distance"], v_max+20, txt, va="center_baseline",
                   ha="center", size="medium", fontname="Formula1", color="white")

    # CORNER MINOR AXIS LINE LOCATIONS
    corner_location = circuit_info.corners["Distance"]
    minor_tick_locations = list()
    for corner in corner_location:
        minor_tick_locations.append(corner)

    # PLOTTING THE LINES
    ax[0].plot(lap1_tel["Distance"], lap1_tel["Speed"], color=team1_colour,
               label=driver1, linestyle=driver1_linestyle, linewidth=2)
    ax[0].plot(lap2_tel["Distance"], lap2_tel["Speed"], color=team2_colour,
               label=driver2, linestyle=driver2_linestyle, linewidth=2)

    ax[1].plot(lap1_tel["Distance"], lap1_tel["Throttle"], color=team1_colour,
               label=driver1, linestyle=driver1_linestyle, linewidth=2)
    ax[1].plot(lap2_tel["Distance"], lap2_tel["Throttle"], color=team2_colour,
               label=driver2, linestyle=driver2_linestyle, linewidth=2)

    ax[2].plot(ref_tel["Distance"], delta_time, color="white", linewidth=3)
    ax[2].axhline(y=0, color="gray")
    plt.xticks(color="white")

    l = ax[0].legend(facecolor="black", labelcolor="white")
    plt.setp(l.texts, family="Formula1", size=axis_fontsize)

    ax[0].get_xaxis().set_ticklabels([])
    ax[1].get_xaxis().set_ticklabels([])
    ax[2].set_xlabel("Distance [metres]", fontname="Formula1",
                     fontsize=axis_fontsize, color="white")

    y_labels = ["Speed [km/h]", "Throttle [%]", "Delta [s]"]
    for n in range(0, 3):
        ax[n].tick_params(labelsize=tick_fontsize)
        ax[n].patch.set_facecolor("black")
        ax[n].set_ylabel(y_labels[n], fontname="Formula1",
                         fontsize=axis_fontsize, color="white")

        ax[n].xaxis.set_minor_locator(FixedLocator(minor_tick_locations))
        ax[n].grid(which="minor", linestyle="-", linewidth=3, color="#333333")
        ax[n].grid(which="minor", axis="y", linewidth=0)
        ax[n].grid(which="major", axis="x", linestyle="-", linewidth=2)

        ax[n].tick_params(axis="y", colors="white")

    fastest_lap_time = strftimedelta(lap1["LapTime"], "%m:%s.%ms")
    diff = strftimedelta(lap2["LapTime"] - lap1["LapTime"], "%s.%ms")
    if "-" not in diff:
        sign = "+"
    else:
        sign = ""

    plt.suptitle(f"Lap {lap_number} Comparison, {data.event["EventName"]} {data.event.year}\n{driver1}: {fastest_lap_time}, {driver2}: {sign}{diff}s",
                 fontsize=20, fontname="Formula1", color="white")

    fig.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.05)
    plt.show()


def race_laps_scatter(data, grand_prix, sc_laps, drivers_in_graph: list):

    all_laps_by_team = initial_setup(data)["team_laps"]

    drivers_in_graph = drivers_in_graph

    axis_fontsize = 16
    tick_fontsize = 12

    teams_in_graph = list()
    handles = list()
    max_time = None
    min_time = None

    plt.figure(figsize=(16, 8), dpi=120)

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
                mean = tyre_laps["LapTime"].mean(numeric_only=False)
                std = tyre_laps["LapTime"].std(numeric_only=False)
                tyre_laps = tyre_laps[tyre_laps["LapTime"] < (mean + std)]

                # GRAPHS THEMSELVES
                try:
                    ax = sns.regplot(
                        data=tyre_laps, x="LapNumber", y="LapTime",
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
                min_time_tyre = tyre_laps["LapTime"].min()
                max_time_tyre = tyre_laps["LapTime"].max()

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
        plt.vlines(
            x=start_lap, ymin=min_time * 0.999, ymax=max_time * 1.001,
            linestyles="dotted", colors=tyre_colours["medium"]
        )
        plt.vlines(
            x=end_lap, ymin=min_time * 0.999, ymax=max_time * 1.001,
            linestyles="dotted", colors=tyre_colours["medium"]
        )
        half_way_sc = (end_lap - start_lap) / 2 + start_lap
        plt.text(
            x=half_way_sc, y=max_time * 0.998,
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
    title = f"{grand_prix} Race Pace\n"
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
