#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


#%%

dists = {
    "Disagreeable": [0.40, 0.40, 0.15, 0.04, 0.01],
    "Neutral": [0.05, 0.25, 0.40, 0.25, 0.05],
    "Agreeable": [0.01, 0.04, 0.15, 0.40, 0.40],
}

df = pd.DataFrame(dists)
assert sum(df.sum() == 1) == 3
# df = df / 100
df.index = df.index + 1
cmap = plt.get_cmap("gray_r")
cmap = truncate_colormap(cmap, 0.15, 1)

df.plot(
    kind="bar",
    subplots=True,
    layout=(1, 3),
    xlabel="Likert-score",
    ylabel="Probability",
    sharex=True,
    sharey=True,
    fontsize=9,
    rot=1,
    yticks=np.arange(0, 0.5, 0.1),
    legend=False,
    colormap=cmap,
    figsize=(6, 2.5),
)
# plt.suptitle("Answer Distributions", fontsize=15)
plt.tight_layout()
plt.savefig("answer_dists.pdf", format="pdf")

#%%

indicator = [
    "Strongly disagree",
    "Disagree",
    "Slightly disagree",
    "Slightly agree",
    "Agree",
    "Strongly agree",
]
scale = [i for i in range(1, 7)]
data = {"Scale": scale, "Indicator": indicator}
df_6_likert = pd.DataFrame(data).T
# df_6_likert = df_6_likert.set_index("Scale")
latex_code_6_likert = df_6_likert.to_latex(
    caption="Six-point Likert scale",
    label="tab:6point",
    header=False,
    index=False,
)

#%%

groups = [i for i in range(1, 5)]
ns = [250, 250, 253, 249]
mssg = [
    "-",
    "Note that this survey employs an advanced statistical technique that detects inattentive behavior. Please pay attention to the questions asked and respond carefully.",
    "Please respond to all subsequent items without effort but pretend that you want your inattention in filling out this survey to remain undetected.",
    "Please respond to all subsequent items without effort with and with no risk of penalty: in fact, we request that you do so.",
]
data = {"Group": groups, "Size": ns, "Additional instructions": mssg}
df_additional = pd.DataFrame(data)
latex_code_additional = df_additional.to_latex(
    caption="Group allocation overview", label="tab:additional", index=False
)

#%%

scales = [
    "Anxiety",
    "Anger",
    "Friendliness",
    "Gregariousness",
    "Imagination",
    "Artistic Interest",
    "Trust",
    "Altruism",
    "Orderliness",
    "Self-Discipline",
]
alpha = [
    "0.78",
    "0.87",
    "0.81",
    "0.79",
    "0.76",
    "0.76",
    "0.86",
    "0.76",
    "0.83",
    "0.73",
]
data = {"Construct": scales, "Cronbach alpha": alpha}
df_rl = pd.DataFrame(data)
latex_code_rl = df_rl.to_latex(
    caption="Personality constructs with corresponding Cronbach alpha values",
    label="tab:rl",
    index=False,
)

#%% SIM

import pyreadr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

methods = ["id", "mmd"]
scatters = ["COV", "MLC", "MCD"]
scatter_pairs = ["$COV-COV_4$", "$MLC-COV$", "$MCD_{.75}-COV$"]
epsilons = ["0.05", "0.10", "0.15"]
epsilons_pct = ["0.05", "0.10", "0.15"]

start_idx = [0, 1, 2]
cr_type = ["row", "col"]

designs = ["1a", "1b", "2"]
latex_mean_results = [[], []]  # [rowwise, cellwise] -> 1a-1b-2a-2b-3
latex_top_e_results = []
latex_top_e_bm_results = []
latex_mean_bm_results = []  # 1a-1b-2a-2b-3
for design in designs:
    import_name = "sim_results_{}.RData".format(design)
    results = pyreadr.read_r(import_name)

    # How many components are selected?
    no_com = results["no_com"]
    box_no_com = pd.DataFrame()
    for scatter in range(3):
        df = pd.DataFrame()
        for start in start_idx:
            temp_e = no_com.iloc[start::3, scatter]
            temp_e = pd.DataFrame(temp_e.value_counts()).reset_index()
            box_values = []
            for i in range(temp_e.shape[0]):
                temp_i = [temp_e.iloc[i, 0] for j in range(temp_e.iloc[i, 1])]
                box_values.extend(temp_i)
            temp = pd.DataFrame(
                {
                    "Number of ICs selected": box_values,
                    "Scatter": scatter,
                    "Epsilon": start,
                }
            )
            df = pd.concat([df, temp])
        box_no_com = pd.concat([box_no_com, df])
    box_no_com["Scatter"] = box_no_com["Scatter"].replace(
        [0, 1, 2], scatter_pairs
    )
    box_no_com["Epsilon"] = box_no_com["Epsilon"].replace([0, 1, 2], epsilons)
    y = box_no_com["Number of ICs selected"].astype(int)
    yint = range(min(y), max(y) + 1, 2)
    my_pal = {"0.05": "salmon", "0.10": "lightblue", "0.15": "gray"}
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    ax1 = sns.boxplot(
        data=box_no_com,
        x="Scatter",
        y="Number of ICs selected",
        hue="Epsilon",
        palette=my_pal,
    )
    ax1.set(xlabel="", yticks=yint)
    axes.get_legend().remove()
    h, l = axes.get_legend_handles_labels()  # Extracting handles and labels
    ph = [plt.plot([], marker="", ls="")[0]]  # Canvas
    handles = ph + h
    labels = ["$\epsilon$"] + l  # Merging labels
    plt.legend(
        handles,
        labels,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
    )
    plt.tight_layout()
    plt.savefig("3. Plots\{}_boxplots_no_com.pdf".format(design), format="pdf")

    # Rowwise and cellwise results: mean and boxplots of F2, precision and recall
    for rc in cr_type:
        mean_results = pd.DataFrame()
        df_box_id = pd.DataFrame()
        df_box_mmd = pd.DataFrame()
        for method in methods:
            df = pd.DataFrame()
            for start in start_idx:
                e_p = (
                    results["ps_{}_{}".format(rc, method)]
                    .iloc[start::3, :]
                    .reset_index(drop=True)
                )
                e_p.columns = scatters
                e_r = (
                    results["rs_{}_{}".format(rc, method)]
                    .iloc[start::3, :]
                    .reset_index(drop=True)
                )
                e_r_box = e_r.melt().loc[:, "value"]

                if method == "id":
                    temp = e_p.melt()
                    temp["temp"] = e_r_box
                    temp["temp2"] = start
                    temp.columns = [
                        "Scatter",
                        "Precision",
                        "Recall",
                        "Epsilon",
                    ]
                    df_box_id = pd.concat([df_box_id, temp])
                    df_box_id["F2"] = (
                        5 * df_box_id.Precision * df_box_id.Recall
                    ) / (4 * df_box_id.Precision + df_box_id.Recall)
                    df_box_id.fillna(0, inplace=True)
                else:
                    temp = e_p.melt()
                    temp["temp"] = e_r_box
                    temp["temp2"] = start
                    temp.columns = [
                        "Scatter",
                        "Precision",
                        "Recall",
                        "Epsilon",
                    ]
                    df_box_mmd = pd.concat([df_box_mmd, temp])
                    df_box_mmd["F2"] = (
                        5 * df_box_mmd.Precision * df_box_mmd.Recall
                    ) / (4 * df_box_mmd.Precision + df_box_mmd.Recall)
                    df_box_mmd.fillna(0, inplace=True)
                e_pr = pd.concat([e_p, e_r], axis=1).dropna()
                e_pr["F2_COV"] = (
                    5 * e_pr["COV"] * e_pr["rs_{}_{}_cov".format(rc, method)]
                ) / (4 * e_pr["COV"] + e_pr["rs_{}_{}_cov".format(rc, method)])
                e_pr["F2_MLC"] = (
                    5 * e_pr["MLC"] * e_pr["rs_{}_{}_mlc".format(rc, method)]
                ) / (4 * e_pr["MLC"] + e_pr["rs_{}_{}_mlc".format(rc, method)])
                e_pr["F2_MCD"] = (
                    5 * e_pr["MCD"] * e_pr["rs_{}_{}_mcd".format(rc, method)]
                ) / (4 * e_pr["MCD"] + e_pr["rs_{}_{}_mcd".format(rc, method)])
                order = [
                    "F2_COV",
                    "COV",
                    "rs_{}_{}_cov".format(rc, method),
                    "F2_MLC",
                    "MLC",
                    "rs_{}_{}_mlc".format(rc, method),
                    "F2_MCD",
                    "MCD",
                    "rs_{}_{}_mcd".format(rc, method),
                ]
                e_pr = e_pr[order]
                e_pr.fillna(0, inplace=True)
                mean_e_pr = e_pr.mean()
                df = pd.concat([df, mean_e_pr], axis=1)
            df = df.transpose()
            df.columns = range(df.columns.size)
            mean_results.columns = range(mean_results.columns.size)
            mean_results = pd.concat([mean_results, df], ignore_index=True)

        mean_results.index = [0.05, 0.10, 0.15, 0.05, 0.10, 0.15]
        xlab = "$\epsilon$"
        if rc == "col":
            if design == "1a":
                mean_results.index = [0.025, 0.050, 0.075, 0.025, 0.050, 0.075]
                epsilons_pct = ["0.025", "0.050", "0.075"]
                xlab = "$\epsilon^c$"
            elif design == "1b":
                mean_results.index = [0.013, 0.025, 0.038, 0.013, 0.025, 0.038]
                epsilons_pct = ["0.013", "0.025", "0.038"]
                xlab = "$\epsilon^c$"
            else:
                mean_results.index = [0.019, 0.038, 0.056, 0.019, 0.038, 0.056]
                epsilons_pct = ["0.019", "0.038", "0.056"]
                xlab = "$\epsilon^c$"

        mean_results.columns = [
            "F2",
            "Precision",
            "Recall",
            "F2",
            "Precision",
            "Recall",
            "F2",
            "Precision",
            "Recall",
        ]

        # Output
        latex_code_mean_results = mean_results.to_latex(
            caption="Mean {}wise careless responding detection results of simulation design {} using ICS distance (ID) and MMD".format(
                rc, design
            ),
            label="tab:sim_{}_results_{}".format(design, rc),
            index=True,
            position="H",
            float_format="{:.3f}".format,
        )
        latex_mean_results[cr_type.index(rc)].append(latex_code_mean_results)

        df_box_id["Epsilon"] = df_box_id["Epsilon"].replace(
            [0, 1, 2], epsilons_pct
        )
        df_box_mmd["Epsilon"] = df_box_mmd["Epsilon"].replace(
            [0, 1, 2], epsilons_pct
        )

        my_pal = {"F2": "salmon", "Precision": "lightblue", "Recall": "gray"}
        melt_id = pd.melt(
            df_box_id,
            id_vars=["Scatter", "Epsilon"],
            value_vars=["F2", "Precision", "Recall"],
        )
        melt_mmd = pd.melt(
            df_box_mmd,
            id_vars=["Scatter", "Epsilon"],
            value_vars=["F2", "Precision", "Recall"],
        )
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 7), sharey=True)
        for z in range(len(scatters)):
            z_scat = scatters[z]
            z_sp = scatter_pairs[z]
            z_melt_id = melt_id[melt_id["Scatter"] == z_scat]
            z_melt_mmd = melt_mmd[melt_mmd["Scatter"] == z_scat]
            ax1 = sns.boxplot(
                data=z_melt_id,
                hue="variable",
                x="Epsilon",
                y="value",
                ax=axes[0, z],
                palette=my_pal,
            )
            ax2 = sns.boxplot(
                data=z_melt_mmd,
                hue="variable",
                x="Epsilon",
                y="value",
                ax=axes[1, z],
                palette=my_pal,
            )
            ax1.set_ylim([-0.1, 1.1])
            ax2.set_ylim([-0.1, 1.1])
            if z == 0:
                ax1.set(title=z_sp, xlabel="", xticklabels=[])
                ax2.set(xlabel=xlab)
                ax1.set_ylabel(
                    "$\mathregular{ICS^{MAD}}$",
                    rotation=0,
                    horizontalalignment="right",
                    fontsize="large",
                )
                ax2.set_ylabel(
                    "$\mathregular{ICS^{MMD}}$",
                    rotation=0,
                    horizontalalignment="right",
                    fontsize="large",
                )
            elif z == 1:
                ax1.set(title=z_sp, xlabel="", ylabel="", xticklabels=[])
                ax2.set(ylabel="", xlabel=xlab)
            else:
                ax1.set(title=z_sp, xlabel="", ylabel="", xticklabels=[])
                ax2.set(xlabel=xlab, ylabel="")
        for x in range(2):
            for y in range(3):
                axes[x, y].get_legend().remove()
        axes[1, 1].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=3,
            frameon=False,
        )
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.1)
        plt.savefig(
            "3. Plots\{}_boxplots_{}_results.pdf".format(design, rc),
            format="pdf",
        )

    # Accuracy

    as_id = results["as_id"]
    as_mmd = results["as_mmd"]
    accuracy = pd.concat([as_id, as_mmd], axis=1)
    accuracy.columns = [
        "COV-COV4",
        "MLC-COV",
        "MCD-COV",
        "COV-COV4",
        "MLC-COV",
        "MCD-COV",
    ]
    a5 = accuracy.iloc[0::3, :].mean()
    a10 = accuracy.iloc[1::3, :].mean()
    a15 = accuracy.iloc[2::3, :].mean()
    accuracy = pd.concat([a5, a10, a15], axis=1).T
    accuracy = pd.concat([accuracy.iloc[:, 0:3], accuracy.iloc[:, 3:]])
    accuracy.index = epsilons * 2

    latex_code_mean_accuracy = accuracy.to_latex(
        caption="Accuracy of the top $\varepsilon$ outlying observations of simulation design {} using ID, MMD".format(
            design
        ),
        label="tab:sim_accuracy_{}".format(design),
        index=True,
        position="H",
        float_format="{:.3f}".format,
    )
    latex_top_e_results.append(latex_code_mean_accuracy)

    as_bm = results["as_bm"]
    as_bm.columns = ["MD", "Gp"]
    a5 = as_bm.iloc[0::3, :].mean()
    a10 = as_bm.iloc[1::3, :].mean()
    a15 = as_bm.iloc[2::3, :].mean()
    as_bm = pd.concat([a5, a10, a15], axis=1).T
    as_bm.index = epsilons

    latex_code_mean_accuracy_bm = as_bm.to_latex(
        caption="Accuracy of the top $\varepsilon$ outlying observations of simulation design {} using MD and Gp".format(
            design
        ),
        label="tab:sim_accuracy_bm_{}".format(design),
        index=True,
        position="H",
        float_format="{:.3f}".format,
    )
    latex_top_e_bm_results.append(latex_code_mean_accuracy_bm)

    # Which questions were flagged the most?
    items_flag = results["items_flag"]
    result = []  # per scatter, then per contamination
    for scatter in range(3):
        result_scatter = []
        for start in start_idx:
            temp_e = items_flag.iloc[start::3, scatter]
            frequency = []
            for i in temp_e:
                if i == "NA":
                    pass
                else:
                    items_flagged = i.split()
                    items_flagged = list(map(int, items_flagged))
                    frequency.extend(items_flagged)
            frequency = pd.DataFrame(
                pd.Series(frequency).value_counts()
            ).sort_index(ascending=False)
            result_scatter.append(frequency)
        result.append(result_scatter)
    bar_items_flag = []
    for i in result:
        temp = pd.concat([i[0], i[1]], axis=1)
        temp = pd.concat([temp, i[2]], axis=1)
        temp.columns = epsilons_pct
        bar_items_flag.append(temp)

    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 7), sharey=True)
    for plot in bar_items_flag:
        plot = plot / (items_flag.shape[0] / 3)
        plot = plot.fillna(0)
        if count == 0:
            ax1 = sns.heatmap(
                plot,
                ax=axes[count],
                cmap="OrRd",
                vmin=0,
                vmax=1,
                cbar=False,
                annot=True,
            )
            ax1.set(title=scatter_pairs[count], xlabel="$\epsilon^c$")
            ax1.set_ylabel("Item")
        elif count == 1:
            ax1 = sns.heatmap(
                plot,
                ax=axes[count],
                cmap="OrRd",
                vmin=0,
                vmax=1,
                cbar=False,
                annot=True,
            )
            ax1.set(title=scatter_pairs[count], xlabel="$\epsilon^c$")
        else:
            ax1 = sns.heatmap(
                plot,
                ax=axes[count],
                cmap="OrRd",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Flag rate"},
                annot=True,
            )
            ax1.set(title=scatter_pairs[count], xlabel="$\epsilon^c$")
        count += 1

    plt.tight_layout()
    plt.savefig(
        "3. Plots\{}_barplots_items_flagged.pdf".format(design), format="pdf"
    )

    epsilons_pct = ["0.05", "0.10", "0.15"]
    # Rowwise benchmark results
    ps_row_bm = results["ps_row_bm"].fillna(0)
    rs_row_bm = results["rs_row_bm"].fillna(0)
    mean_bm_results = pd.DataFrame()
    df_box_md = pd.DataFrame()
    df_box_gp = pd.DataFrame()
    for start in start_idx:
        e_p = ps_row_bm.iloc[start::3, :].reset_index(drop=True)
        e_r = rs_row_bm.iloc[start::3, :].reset_index(drop=True)
        e_pr = pd.concat([e_p, e_r], axis=1).dropna()
        e_pr["F2_md"] = (5 * e_pr["ps_row_md"] * e_pr["rs_row_md"]) / (
            4 * e_pr["ps_row_md"] + e_pr["rs_row_md"]
        )
        e_pr["F2_gp"] = (5 * e_pr["ps_row_gp"] * e_pr["rs_row_gp"]) / (
            4 * e_pr["ps_row_gp"] + e_pr["rs_row_gp"]
        )
        e_pr.fillna(0, inplace=True)
        order = [
            "F2_md",
            "ps_row_md",
            "rs_row_md",
            "F2_gp",
            "ps_row_gp",
            "rs_row_gp",
        ]
        e_pr = e_pr[order]

        temp_md = e_pr.iloc[:, 0:3]
        temp_md["Epsilon"] = start
        df_box_md = pd.concat([df_box_md, temp_md])

        temp_gp = e_pr.iloc[:, 3:]
        temp_gp["Epsilon"] = start
        df_box_gp = pd.concat([df_box_gp, temp_gp])

        mean_e_pr = e_pr.mean()
        mean_bm_results = pd.concat([mean_bm_results, mean_e_pr], axis=1)
    mean_bm_results = mean_bm_results.T
    mean_bm_results.columns = [
        "F2",
        "Precision",
        "Recall",
        "F2",
        "Precision",
        "Recall",
    ]
    mean_bm_results = pd.concat(
        [mean_bm_results.iloc[:, 0:3], mean_bm_results.iloc[:, 3:]]
    )
    mean_bm_results.index = epsilons * 2

    latex_code_mean_bm_results = mean_bm_results.to_latex(
        caption="Mean rowwise careless responding detection results of simulation design {} using MD and Gp".format(
            design
        ),
        label="tab:sim_{}_bm_results".format(design),
        index=True,
        position="H",
        float_format="{:.3f}".format,
    )
    latex_mean_bm_results.append(latex_code_mean_bm_results)

    df_box_md["Epsilon"] = df_box_md["Epsilon"].replace(
        [0, 1, 2], epsilons_pct
    )
    df_box_md.columns = ["F2", "Precision", "Recall", "Epsilon"]
    melt_md = pd.melt(
        df_box_md, id_vars="Epsilon", value_vars=["F2", "Precision", "Recall"]
    )

    df_box_gp["Epsilon"] = df_box_gp["Epsilon"].replace(
        [0, 1, 2], epsilons_pct
    )
    df_box_gp.columns = ["F2", "Precision", "Recall", "Epsilon"]
    melt_gp = pd.melt(
        df_box_gp, id_vars="Epsilon", value_vars=["F2", "Precision", "Recall"]
    )
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 7), sharey=True)
    ax1 = sns.boxplot(
        data=melt_md,
        x="Epsilon",
        y="value",
        hue="variable",
        ax=axes[0],
        palette=my_pal,
    )
    ax2 = sns.boxplot(
        data=melt_gp,
        x="Epsilon",
        y="value",
        hue="variable",
        ax=axes[1],
        palette=my_pal,
    )
    ax1.set_ylim([-0.1, 1.1])
    ax2.set_ylim([-0.1, 1.1])
    ax1.set(xlabel="", xticklabels=[])
    ax2.set(xlabel="$\epsilon$")
    ax1.set_ylabel(
        "MD", rotation=0, horizontalalignment="right", fontsize="large"
    )
    ax2.set_ylabel(
        "$\mathregular{G^{p}}$",
        rotation=0,
        horizontalalignment="right",
        fontsize="large",
    )
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[1].legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False
    )
    plt.tight_layout()
    plt.savefig(
        "3. Plots\{}_boxplots_bm_results.pdf".format(design),
        format="pdf",
    )

#%% IPIP

import pyreadr
import numpy as np
import pandas as pd

results = pyreadr.read_r("ipip_results.RData")

# Number of components selected
no_com = results["no_com"]
no_com.index = ["15", "25", "35"]
no_com.columns = ["COV", "MLC", "MCD"]
latex_code_no_com = no_com.to_latex(
    caption="Number of invariant components selected for the IPIP data sets",
    label="tab:ipip_no_com",
    index=True,
    position="H",
    float_format="{:.0f}".format,
)

# Rowwise results
ps_row_id = results["ps_row_id"]
rs_row_id = results["rs_row_id"]
row_id = pd.concat([ps_row_id, rs_row_id], axis=1)
row_id["F2_cov"] = (5 * row_id["ps_row_id_cov"] * row_id["rs_row_id_cov"]) / (
    4 * row_id["ps_row_id_cov"] + row_id["rs_row_id_cov"]
)
row_id["F2_mlc"] = (5 * row_id["ps_row_id_mlc"] * row_id["rs_row_id_mlc"]) / (
    4 * row_id["ps_row_id_mlc"] + row_id["rs_row_id_mlc"]
)
row_id["F2_mcd"] = (5 * row_id["ps_row_id_mcd"] * row_id["rs_row_id_mcd"]) / (
    4 * row_id["ps_row_id_mcd"] + row_id["rs_row_id_mcd"]
)
order = [
    "F2_cov",
    "ps_row_id_cov",
    "rs_row_id_cov",
    "F2_mlc",
    "ps_row_id_mlc",
    "rs_row_id_mlc",
    "F2_mcd",
    "ps_row_id_mcd",
    "rs_row_id_mcd",
]
row_id = row_id[order]
row_id.columns = ["F2", "Precision", "Recall"] * 3


ps_row_mmd = results["ps_row_mmd"]
rs_row_mmd = results["rs_row_mmd"]

row_mmd = pd.concat([ps_row_mmd, rs_row_mmd], axis=1)
row_mmd["F2_cov"] = (
    5 * row_mmd["ps_row_mmd_cov"] * row_mmd["rs_row_mmd_cov"]
) / (4 * row_mmd["ps_row_mmd_cov"] + row_mmd["rs_row_mmd_cov"])
row_mmd["F2_mlc"] = (
    5 * row_mmd["ps_row_mmd_mlc"] * row_mmd["rs_row_mmd_mlc"]
) / (4 * row_mmd["ps_row_mmd_mlc"] + row_mmd["rs_row_mmd_mlc"])
row_mmd["F2_mcd"] = (
    5 * row_mmd["ps_row_mmd_mcd"] * row_mmd["rs_row_mmd_mcd"]
) / (4 * row_mmd["ps_row_mmd_mcd"] + row_mmd["rs_row_mmd_mcd"])
order = [
    "F2_cov",
    "ps_row_mmd_cov",
    "rs_row_mmd_cov",
    "F2_mlc",
    "ps_row_mmd_mlc",
    "rs_row_mmd_mlc",
    "F2_mcd",
    "ps_row_mmd_mcd",
    "rs_row_mmd_mcd",
]
row_mmd = row_mmd[order]
row_mmd.columns = ["F2", "Precision", "Recall"] * 3

row_results = pd.concat([row_id, row_mmd])
row_results.index = ["15", "25", "35"] * 2

latex_code_row_results = row_results.to_latex(
    caption="Rowwise careless responding detection results of IPIP data set using ID and MMD",
    label="tab:ipip_row_results",
    index=True,
    position="H",
    float_format="{:.3f}".format,
)

ps_row_bm = results["ps_row_bm"]
rs_row_bm = results["rs_row_bm"]
row_bm = pd.concat([ps_row_bm, rs_row_bm], axis=1)
row_bm["F2_md"] = (5 * row_bm["ps_row_md"] * row_bm["rs_row_md"]) / (
    4 * row_bm["ps_row_md"] + row_bm["rs_row_md"]
)
row_bm["F2_gp"] = (5 * row_bm["ps_row_gp"] * row_bm["rs_row_gp"]) / (
    4 * row_bm["ps_row_gp"] + row_bm["rs_row_gp"]
)
order = [
    "F2_md",
    "ps_row_md",
    "rs_row_md",
    "F2_gp",
    "ps_row_gp",
    "rs_row_gp",
]
row_bm = row_bm[order]
row_bm.columns = ["F2", "Precision", "Recall"] * 2
row_bm.index = ["15", "25", "35"]
row_bm = pd.concat([row_bm.iloc[:, 0:3], row_bm.iloc[:, 3:]])
bm_results = row_bm

latex_code_bm_results = bm_results.to_latex(
    caption="Rowwise careless responding detection results of IPIP data set using MD and $G^p$",
    label="tab:ipip_bm_results",
    index=True,
    position="H",
    float_format="{:.3f}".format,
)


as_id = results["as_id"]
as_mmd = results["as_mmd"]
accuracy = pd.concat([as_id, as_mmd], axis=1)
accuracy.columns = [
    "COV-COV4",
    "MLC-COV",
    "MCD-COV",
    "COV-COV4",
    "MLC-COV",
    "MCD-COV",
]
accuracy = pd.concat([accuracy.iloc[:, 0:3], accuracy.iloc[:, 3:]])
accuracy.index = ["15", "25", "35"] * 2

latex_code_mean_accuracy = accuracy.to_latex(
    caption="Mean accuracy of the top $\varepsilon$ outlying observations of the IPIP using ID, MMD",
    label="tab:ipip_accuracy",
    index=True,
    position="H",
    float_format="{:.3f}".format,
)

as_bm = results["as_bm"]
as_bm.columns = ["MD", "Gp"]
as_bm.index = ["15", "25", "35"]

latex_code_mean_accuracy_bm = as_bm.to_latex(
    caption="Mean accuracy of the top $\varepsilon$ outlying observations of simulation design {} using MD and Gp",
    label="tab:ipip_accuracy_bm",
    index=True,
    position="H",
    float_format="{:.3f}".format,
)


# Cellwise results
ps_col_id = results["ps_col_id"]
rs_col_id = results["rs_col_id"]
col_id = pd.concat([ps_col_id, rs_col_id], axis=1)
col_id["F2_cov"] = (5 * col_id["ps_col_id_cov"] * col_id["rs_col_id_cov"]) / (
    4 * col_id["ps_col_id_cov"] + col_id["rs_col_id_cov"]
)
col_id["F2_mlc"] = (5 * col_id["ps_col_id_mlc"] * col_id["rs_col_id_mlc"]) / (
    4 * col_id["ps_col_id_mlc"] + col_id["rs_col_id_mlc"]
)
col_id["F2_mcd"] = (5 * col_id["ps_col_id_mcd"] * col_id["rs_col_id_mcd"]) / (
    4 * col_id["ps_col_id_mcd"] + col_id["rs_col_id_mcd"]
)
order = [
    "F2_cov",
    "ps_col_id_cov",
    "rs_col_id_cov",
    "F2_mlc",
    "ps_col_id_mlc",
    "rs_col_id_mlc",
    "F2_mcd",
    "ps_col_id_mcd",
    "rs_col_id_mcd",
]
col_id = col_id[order]
col_id.columns = ["F2", "Precision", "Recall"] * 3


ps_col_mmd = results["ps_col_mmd"]
rs_col_mmd = results["rs_col_mmd"]

col_mmd = pd.concat([ps_col_mmd, rs_col_mmd], axis=1)
col_mmd["F2_cov"] = (
    5 * col_mmd["ps_col_mmd_cov"] * col_mmd["rs_col_mmd_cov"]
) / (4 * col_mmd["ps_col_mmd_cov"] + col_mmd["rs_col_mmd_cov"])
col_mmd["F2_mlc"] = (
    5 * col_mmd["ps_col_mmd_mlc"] * col_mmd["rs_col_mmd_mlc"]
) / (4 * col_mmd["ps_col_mmd_mlc"] + col_mmd["rs_col_mmd_mlc"])
col_mmd["F2_mcd"] = (
    5 * col_mmd["ps_col_mmd_mcd"] * col_mmd["rs_col_mmd_mcd"]
) / (4 * col_mmd["ps_col_mmd_mcd"] + col_mmd["rs_col_mmd_mcd"])
order = [
    "F2_cov",
    "ps_col_mmd_cov",
    "rs_col_mmd_cov",
    "F2_mlc",
    "ps_col_mmd_mlc",
    "rs_col_mmd_mlc",
    "F2_mcd",
    "ps_col_mmd_mcd",
    "rs_col_mmd_mcd",
]
col_mmd = col_mmd[order]
col_mmd.columns = ["F2", "Precision", "Recall"] * 3

col_results = pd.concat([col_id, col_mmd])
col_results.index = ["15", "25", "35"] * 2
col_results.fillna(0, inplace=True)

latex_code_col_results = col_results.to_latex(
    caption="Cellwise careless responding detection results of IPIP data set using ID and MMD",
    label="tab:ipip_col_results",
    index=True,
    position="H",
    float_format="{:.3f}".format,
)

group = results["group"]
group.columns = [
    "COV-COV4",
    "MLC-COV",
    "MCD-COV",
    "COV-COV4",
    "MLC-COV",
    "MCD-COV",
]
group = pd.concat([group.iloc[:, 0:3], group.iloc[:, 3:]])


latex_code_group = group.to_latex(
    caption="Recall of groups 3 and 4 of the IPIP data set using ID and MMD",
    label="tab:ipip_group_recall",
    index=True,
    position="H",
    float_format="{:.3f}".format,
)
