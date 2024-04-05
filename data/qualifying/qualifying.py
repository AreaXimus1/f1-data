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


def compare_telemetry(data, driver1, driver2):
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
