# -*- coding: utf-8 -*-

import matplotlib

matplotlib.rc('font', family='DejaVu Sans')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np


steps = [1000,3000,6000,9000,12000]
colors ={"se": "green", "pe": "blue", "da": "red"}
ylims = {"dtc": 25, "hg": 2100}
results = {}
for method in ["se", "pe", "da"]:
    results[method] = {}
    for scenario in ["dtc", "hg"]:
        results[method][scenario] = {}

results["se"]["dtc"]["p25"] = [4, 6.75, 8, 11, 11.75]
results["se"]["dtc"]["mean"] = [7.64, 11.87, 12.41, 14.6, 14.41]
results["se"]["dtc"]["p75"] = [10.25, 17, 17.25, 19, 19]

results["se"]["hg"]["p25"] = [316, 412, 444, 444, 476]
results["se"]["hg"]["mean"] = [605.8, 777.1, 854.6, 885.6, 929.9]
results["se"]["hg"]["p75"] = [764, 1020, 1148, 1212, 1276]

results["pe"]["dtc"]["p25"] = [3, 7, 16, 15, 17.75]
results["pe"]["dtc"]["mean"] = [5.39, 11.86, 18.44, 17.46, 18.49]
results["pe"]["dtc"]["p75"] = [6.25, 18, 21, 20, 21]

results["pe"]["hg"]["p25"] = [348, 444, 508, 572, 572]
results["pe"]["hg"]["mean"] = [635.4, 854.2, 979.9, 1040, 1078.4]
results["pe"]["hg"]["p75"] = [796, 1166.2, 1348.2, 1436, 1528]

dagger_steps = [6000, 8000, 10000, 12000]
results["da"]["hg"]["mean"] = [897.15, 931.9, 978.16, 942.29]

st = {"da" : dagger_steps, "se" : steps, "pe": steps}
labels = {"se": u"zwykły ekspert", "pe": u"prezentujący ekspert", "da": u"agregacja"}


def draw_plot(method, scenario):
    color = colors[method]
    steps = st[method]
    plt.clf()
    plt.xticks(steps)
    ax = plt.figure().add_subplot(111)
    if "p25" in results[method][scenario] and "p75" in results[method][scenario]:
        ax.fill_between(steps, results[method][scenario]["p25"], results[method][scenario]["p75"], color='light' + color, label ="kwartyle")
    plt.plot(steps, results[method][scenario]["mean"], color=color, linewidth=2, label =u"średnia")
    axes = plt.gca()
    axes.set_xlim([steps[0], steps[len(steps) -1]])
    axes.set_ylim([0, ylims[scenario]])
    plt.xlabel(u"liczba kroków uczących")
    plt.ylabel(u"średni wynik agenta")
    plt.legend()
    plt.savefig(method + "_" + scenario + "_plot.png")


for method in ["se", "pe"]:
    for scenario in ["dtc", "hg"]:
        if results[method][scenario]:
            draw_plot(method, scenario)

draw_plot("da", "hg")


scenario = "dtc"
plt.clf()
plt.xticks(steps)
ax = plt.figure().add_subplot(111)
axes = plt.gca()
axes.set_xlim([steps[0], steps[len(steps) - 1]])
axes.set_ylim([0, ylims[scenario]])
plt.xlabel(u"liczba kroków uczących")
plt.ylabel(u"średni wynik agenta")

for method in ["se", "pe"]:
    plt.plot(st[method], results[method][scenario]["mean"], color=colors[method], linewidth=2, label = labels[method])
plt.legend()
plt.savefig("comp" + "_" + scenario + "_plot.png")

scenario = "hg"
plt.clf()
plt.xticks(steps)
ax = plt.figure().add_subplot(111)
axes = plt.gca()
axes.set_xlim([6000, 12000])
axes.set_ylim([800, 1200])
plt.xlabel(u"liczba kroków uczących")
plt.ylabel(u"średni wynik agenta")
for method in ["se", "pe", "da"]:
    plt.plot(st[method], results[method][scenario]["mean"], color=colors[method], linewidth=2, label = labels[method])
plt.legend()
plt.savefig("comp" + "_" + scenario + "_plot.png")
