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


def compare_telemetry_depracated(data, driver1, driver2):
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

    lap1 = LAPS.pick_driver(driver1).pick_fastest()
    lap2 = LAPS.pick_driver(driver2).pick_fastest()
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


def compare_telemetry(data, driver_info: List[Dict[str, Any]], vlines: List[Dict[str, Any]] = None, xlim: Tuple[int, int] = None, delta_ylim: Tuple[float, float] = None):
    '''
    Takes a race data, and the info for drivers you want to compare as a list
    of dictionaries. Keys: driver, lap, colour override (opt.)

    EXAMPLES:
    driver_info = [{
        "driver": "HAM",
        "session": Q3,
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
    Q1, Q2, Q3 = LAPS.split_qualifying_sessions()
    circuit_info = data.get_circuit_info()

    axis_fontsize = 18
    tick_fontsize = 14

    line_styles = ["solid", "dotted", "dashdot", "dashed"]

    all_laps = list()
    all_teams = list()
    all_lap_numbers = list()

    for dict_ in driver_info:
        driver = dict_["driver"]
        session = dict_["session"]

        # if current lap number is not in all_lap_numbers, adds lap number to list
        # in order to see if the lap numbers need to be labelled
        if session not in all_lap_numbers:
            all_lap_numbers.append(session)

        try:
            colour_override = dict_["colour"]
        except KeyError:
            colour_override = None

        team = drivers[driver]["Team"]

        if colour_override is not None:
            team_colour = colour_override
        else:
            team_colour = team_colours[team]

        if session == "q1":
            lap_info = Q1.pick_driver(driver).pick_fastest()
        elif session == "q2":
            lap_info = Q2.pick_driver(driver).pick_fastest()
        else:
            lap_info = Q3.pick_driver(driver).pick_fastest()

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
            "session": session,
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
            label = f"{lap["driver"]}, {lap["session"]}"
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
        all_laps[0]["driver"]} {all_laps[0]["session"]}: {main_lap_time}"

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
                                       } {lap["session"]}: {sign}{diff} "
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


def compare_telemetry_different_years(driver_info: List[Dict[str, Any]], vlines: List[Dict[str, Any]] = None, xlim: Tuple[int, int] = None, delta_ylim: Tuple[float, float] = None):
    '''
    Takes a race data, and the info for drivers you want to compare as a list
    of dictionaries. Keys: driver, lap, colour override (opt.)

    EXAMPLES:
    driver_info = [{
        "driver": "HAM",
        "session": q1,
        "data": DATA,
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

    axis_fontsize = 18
    tick_fontsize = 14

    line_styles = ["solid", "dotted", "dashdot", "dashed"]

    all_laps = list()
    all_teams = list()
    all_lap_numbers = list()

    for dict_ in driver_info:
        driver = dict_["driver"]
        session = dict_["session"].upper()
        data = dict_["data"]

        laps = data.laps
        Q1, Q2, Q3 = laps.split_qualifying_sessions()
        circuit_info = data.get_circuit_info()

        # if current lap number is not in all_lap_numbers, adds lap number to list
        # in order to see if the lap numbers need to be labelled
        if session not in all_lap_numbers:
            all_lap_numbers.append(session)

        try:
            colour_override = dict_["colour"]
        except KeyError:
            colour_override = None

        team = drivers[driver]["Team"]

        if driver in ["BOT", "ZHO"] and data.event.year != 2024:
            team = "Alfa Romeo"
        elif driver in ["RIC", "TSU"] and data.event.year != 2024:
            team = "RB"

        if colour_override is not None:
            team_colour = colour_override
        else:
            team_colour = team_colours[team]

        if session == "Q1":
            lap_info = Q1.pick_driver(driver).pick_fastest()
        elif session == "Q2":
            lap_info = Q2.pick_driver(driver).pick_fastest()
        else:
            lap_info = Q3.pick_driver(driver).pick_fastest()

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
            "session": session,
            "lap info": lap_info,
            "lap tel": lap_tel,
            "year": data.event.year
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
            label = f"{lap["driver"]}, {lap["session"]}, {lap["year"]}"
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
    title = f"Telemetry Comparison, {data.event["EventName"]} Qualifying\n{
        all_laps[0]["driver"]} {all_laps[0]["session"]} {all_laps[0]["year"]}: {main_lap_time}"

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

                    title = title + \
                        f"{lap["driver"]} {lap["session"]}, {
                            lap["year"]}: {sign}{diff} "
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
