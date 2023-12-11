# -*- coding:utf-8 -*-
import copy
import math
import random

import matplotlib
import matplotlib.pyplot as plt

# 处理乱码

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['savefig.dpi'] = 300

import re


def plot_results(list_results_from_methods, list_methods, task_name=None):
    list_colors = [
        "r", "g", "blue", "black",
        "olive", "honeydew",

    ]
    list_markers = [
        "d", "o", "v",
        "1", "^",
        "p", "*",
        "x", "2",
    ]

    list_linestyles = [
        "-", "--", "-.", ":",
    ]

    for linestyle, color, marker, results, method_name in zip(
            list_linestyles,
            list_colors,
            list_markers,
            list_results_from_methods,
            list_methods
    ):
        X = results[0]
        Y = results[1]
        plt.plot(
                X, Y,
                color=color,
                marker=marker,
                markersize=8,
                linestyle=linestyle,
                label=method_name,
                linewidth=1.5
            )


    # plt.xticks(rotation=45)
    plt.xlabel("Proportion of dictionary", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel("F1 score", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.title("{}".format(task_name), fontdict={'family': 'Times New Roman', 'size': 18})

    # plt.xlim(math.log10(100e+3), math.log10(2000e+3))
    # plt.ylim(69, 79)

    plt.xticks(
        (
            0.25, 0.50, 0.75, 1.00,
        ),
        ("25%", "50%", "75%", "100%", )
    )

    # upper left 将图例a显示到左上角
    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 16})
    plt.tight_layout()

    # plt.show()
    print(f"./src/span_ner/画图/dict_affects.png")
    plt.savefig(f"./src/span_ner/画图/dict_affects.png")


if __name__ == "__main__":

    ##########################
    # TASK: RTE
    ##########################
    task_name = "CMeEE-V2"

    method0 = "Dict-matching"
    x0 = [
        0.25, 0.5, 0.75, 1.0
    ]
    y0 = [0.228, 0.324, 0.407, 0.473]

    method1 = "TopNeg"
    x1 = [
        0.25, 0.5, 0.75, 1.0
    ]
    y1 = [0.301, 0.412, 0.506, 0.543]

    method2 = "CLNER"
    x2 = [
        0.25, 0.5, 0.75, 1.0
    ]
    y2 = [0.347,  0.442, 0.521, 0.553]

    plot_results(
        [(x0, y0), (x1, y1), (x2, y2) ],
        [method0, method1, method2, ],
        task_name=task_name
    )




