"""Build paired, presentation-oriented TeX from one verified numeric JSON file.

This writes sources only. Compile and visually review the complete pair before
publishing PDFs. No model fitting or frozen-record mutation happens here.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

METHODS = [
    ("global", "全局搜索", "Global search"),
    ("forward", "前向选择", "Forward selection"),
    ("forward_swap", "前向加交换", "Forward with swaps"),
    ("marginal_mmd", "逐变量 MMD", "Marginal MMD"),
    ("l1_logistic", "L1 逻辑回归", "L1 logistic"),
    ("mrmr", "mRMR", "mRMR"),
    ("joint_mmd", "联合 MMD", "Joint MMD"),
    ("random", "均匀随机", "Uniform random"),
]


def bi(zh, en):
    return {"zh": zh, "en": en}


def esc(value):
    text = str(value)
    special = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(special.get(char, char) for char in text)


def local(value, language):
    if isinstance(value, dict) and language in value:
        return value[language]
    return value


def number(value, digits=1, scale=1):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Numeric cells must be finite numbers")
    return f"{scale * value:.{digits}f}"


def cell(value, language):
    if value is None:
        return r"\pendingcell"
    if isinstance(value, dict) and "mean" in value:
        digits, scale = value.get("digits", 1), value.get("scale", 1)
        mean = number(value["mean"], digits, scale)
        if value.get("sd") is not None:
            mean += r" $\pm$ " + number(value["sd"], digits, scale)
        if value.get("ci95") is not None:
            lo, hi = value["ci95"]
            mean += " [" + number(lo, digits, scale) + ", " + number(hi, digits, scale) + "]"
        return mean
    return esc(local(value, language))


def default_data():
    coverage = [
        ("main", bi("E1 合成选择", "E1 synthetic selection"), 600, 1800),
        ("anchor", bi("原论文混合模型", "Original mixture anchor"), 500, 500),
        ("public", bi("E1 公开数据", "E1 public data"), 100, 210),
        ("shift", bi("E2 分布变化", "E2 test-distribution change"), 120, 120),
        ("small", bi("E3 小问题核查", "E3 exact small problems"), 40, 40),
        ("scaling", bi("E3 规模与代价", "E3 scaling and cost"), 200, 200),
        ("ablation", bi("小规模消融", "Small ablations"), 20, 20),
    ]
    empty_model_rows = [[bi(zh, en), None, None, None, None] for _, zh, en in METHODS]
    return {
        "schema_version": 1,
        "state": "not_run",
        "as_of": "2026-10-18",
        "coverage": [
            {
                "suite": code,
                "label": label,
                "planned_data": count,
                "planned_budget_tasks": tasks,
                "observed_data": None,
                "verified_data": None,
                "failed_data": None,
                "timeout_method_runs": None,
            }
            for code, label, count, tasks in coverage
        ],
        "findings": [],
        "figures": {
            "main": {"zh": None, "en": None},
            "public": {"zh": None, "en": None},
            "shift": {"zh": None, "en": None},
            "search": {"zh": None, "en": None},
            "anchor": {"zh": None, "en": None},
        },
        "tables": {
            "main": {
                "columns": [
                    bi("方法", "Method"),
                    bi("均值", "Mean"),
                    bi("波动", "Variance"),
                    bi("相关", "Correlation"),
                    bi("混合", "Mixture"),
                ],
                "rows": [list(row) for row in empty_model_rows],
                "note": bi(
                    "中档 D=50、n=20、s=d=4。数值为概率平衡错误率百分比的均值 ± 标准差；每格应有三十套独立数据。",
                    "Medium scale D=50, n=20, s=d=4. Cells give probabilistic balanced error in percent, mean ± SD; each cell should contain thirty independent datasets.",
                ),
            },
            "generality": {
                "columns": [
                    bi("条件", "Condition"),
                    bi("全局错误率", "Global error"),
                    bi("前向加交换", "Forward + swaps"),
                    bi("联合 MMD", "Joint MMD"),
                ],
                "rows": [],
                "note": bi(
                    "数值为测试概率错误率（%）的均值 ± 标准差；每行单独汇总并列出有效重复数。其他方法和预算见配套数据。",
                    "Test probability BER in percent, mean ± sample SD; each row is summarized separately with its observed repeat count. Other methods and budgets remain in the accompanying data.",
                ),
            },
            "public": {
                "columns": [bi("方法", "Method"), "WDBC", "Ionosphere", "Sonar", "HAR"],
                "rows": [list(row) for row in empty_model_rows],
                "note": bi(
                    "前三组为 n=20、d=4 的十个共享划分，报告均值 ± 划分标准差。HAR 为 n=40、d=16，先在每名测试者内平均十次训练结果，再对九个人等权；其区间按人重采样。",
                    "For the first three datasets, n=20 and d=4 with ten shared splits; report mean ± split SD. HAR uses n=40, d=16: average ten training repetitions within each test person, then weight the nine people equally; resample people for its interval.",
                ),
            },
            "anchor": {
                "columns": [
                    bi("全变量检验", "All-variable test"),
                    bi("概率错误率", "Probabilistic error"),
                    bi("阈值错误率", "Threshold error"),
                    bi("有效重复", "Observed repeats"),
                ],
                "rows": [[bi("100 维混合模型", "100-dimensional mixture"), None, None, None]],
                "note": bi(
                    "与原文相同的生成机制；本文另设验证样本并使用当前评分构造，因此不与原文表格直接对数值。",
                    "The data-generating mechanism matches the original study. Our extra validation set and current score construction differ, so this is not a direct numerical replication of its table.",
                ),
            },
            "shift": {
                "columns": [
                    bi("模型 / 变化", "Model / change"),
                    bi("全局增量", "Global increase"),
                    bi("前向加交换", "Forward + swaps"),
                    bi("联合 MMD", "Joint MMD"),
                ],
                "rows": [],
                "note": bi(
                    "增量为最大变化时错误率减去原始错误率，单位是百分点。每套数据内先作差，再汇总三十套数据。",
                    "Increase means error at the largest change minus original error, in percentage points. Compute a difference within each dataset before aggregating thirty datasets.",
                ),
            },
            "radius": {
                "columns": [
                    bi("模型 / 方法", "Model / method"),
                    bi("半径 ×0.5", "Radius ×0.5"),
                    bi("半径 ×1", "Radius ×1"),
                    bi("半径 ×2", "Radius ×2"),
                ],
                "rows": [],
                "note": bi(
                    "均值与混合各十套新数据。数值为测试概率错误率百分比；零评分兜底次数另报，不能从错误率 50% 推断评分恒定。",
                    "Ten new datasets each for mean and mixture. Values are probabilistic test error in percent; report zero-score fallback counts separately. Error of 50% alone does not establish a constant score.",
                ),
            },
            "small": {
                "columns": [
                    bi("方法", "Method"),
                    bi("高出最优", "Excess risk"),
                    bi("时间 / 秒", "Time / s"),
                    bi("拟合次数", "Fits"),
                ],
                "rows": [
                    [bi("穷举", "Enumeration"), None, None, None],
                    [bi("全局搜索", "Global search"), None, None, None],
                    [bi("前向选择", "Forward"), None, None, None],
                    [bi("前向加交换", "Forward + swaps"), None, None, None],
                ],
                "note": bi(
                    "四种模型、两种训练样本量，每格按四十套独立数据汇总；风险的尺度是两类损失之和，不是错误率百分比。",
                    "Four models and two training sizes, summarized over forty independent datasets per method. Risk is the sum of the two class losses, rather than an error percentage.",
                ),
            },
            "checks": {
                "columns": [
                    bi("核查", "Check"),
                    bi("完成 / 应做", "Completed / planned"),
                    bi("失败或违反", "Failures or violations"),
                ],
                "rows": [
                    [bi("完整穷举实例", "Complete enumeration instances"), None, None],
                    [bi("独立固定选集 LP", "Independent fixed-subset LPs"), None, None],
                    [bi("全部下界割检查", "All lower-bound cut checks"), None, None],
                    [bi("区间覆盖最优值", "Intervals covering optimum"), None, None],
                ],
                "note": bi(
                    "通过表示在已运行实例和既定容差下未发现数值反例；不能代替理论证明。",
                    "Passing means no numerical counterexample was found on these instances at the stated tolerance; it does not replace a proof.",
                ),
            },
            "initialization": {
                "columns": [
                    bi("初始化", "Initialization"),
                    bi("风险", "Risk"),
                    bi("时间 / 秒", "Time / s"),
                    bi("最终差距", "Final gap"),
                ],
                "rows": [
                    [bi("逐变量 MMD", "Marginal MMD"), None, None, None],
                    [bi("一次随机选集", "One random subset"), None, None, None],
                ],
                "note": bi(
                    "使用与半径消融相同的二十套数据、原选定半径；随机和 MMD 逐实例配对。",
                    "Use the same twenty datasets as the radius ablation at the originally selected radius; pair random and MMD starts by dataset.",
                ),
            },
            "execution": {
                "columns": [bi("项目", "Item"), bi("实际运行信息", "Actual execution information")],
                "rows": [
                    [bi("CPU / 内存", "CPU / memory"), None],
                    [bi("算法 / 求解器版本", "Algorithm / solver version"), None],
                    [
                        bi("每次搜索时间限制", "Search time limit"),
                        bi("通常 300 秒，HAR 600 秒", "Normally 300 s, HAR 600 s"),
                    ],
                    [bi("求解器线程数", "Solver threads"), 1],
                    [
                        bi("外层 / 内层绝对精度", "Outer / inner absolute tolerances"),
                        bi(
                            "通常 0.001 / 0.0001；小问题另收紧",
                            "Normally 0.001 / 0.0001; tightened for exact small problems",
                        ),
                    ],
                    [bi("数值失败 / 超时", "Numerical failures / timeouts"), None],
                ],
                "note": "",
            },
        },
        "commentary": {
            name: None
            for name in (
                "main",
                "generality",
                "public",
                "anchor",
                "shift",
                "radius",
                "small",
                "checks",
                "scaling",
                "initialization",
                "conclusions",
                "limitations",
            )
        },
        "readable_data_path": None,
    }


PREAMBLE = r"""% !TEX program = xelatex
\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage{fontspec}
FONT
\usepackage{amsmath,mathtools,unicode-math}
\setmathfont{Cambria Math}
\usepackage[expansion=false,protrusion=false]{microtype}
\usepackage{booktabs,array,tabularx,enumitem,xcolor,graphicx}
\usepackage[hidelinks]{hyperref}
\definecolor{ink}{HTML}{263B49}
\definecolor{pending}{HTML}{995A21}
\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5.5pt}
\setlength{\emergencystretch}{2em}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.18}
\linespread{1.06}
\setlist[itemize]{leftmargin=1.5em,itemsep=3pt,topsep=3pt,parsep=0pt}
\setlength{\abovedisplayskip}{4pt}
\setlength{\belowdisplayskip}{4pt}
\clubpenalty=10000\widowpenalty=10000
\newcommand{\topic}[1]{\par\vspace{8pt}{\large\bfseries\color{ink}#1}\par\vspace{2pt}}
\newcommand{\pendingcell}{\textcolor{pending}{---}}
\newcommand{\unfilled}[1]{\par{\color{pending}\small #1}\par}
\newcommand{\noteline}[1]{\par{\small #1}\par}
\newcounter{reportfigure}
\newcommand{\reportcaption}[1]{\refstepcounter{reportfigure}\noteline{\textbf{FIGURE_PREFIX~\arabic{reportfigure}.} #1}}
\begin{document}
"""


class Report:
    def __init__(self, data, language):
        self.data, self.lang = data, language

    def t(self, zh, en):
        return zh if self.lang == "zh" else en

    def paragraph(self, zh, en):
        return self.t(zh, en) + "\n\n"

    def heading(self, zh, en):
        return "\\topic{" + self.t(zh, en) + "}\n"

    def commentary(self, name):
        value = self.data.get("commentary", {}).get(name)
        if value:
            return esc(local(value, self.lang)) + "\n\n"
        return (
            "\\unfilled{"
            + self.t(
                "结果解读尚未填入。需依据本节的真实数据写出比较、差值与边界。",
                "Interpretation is unfilled. Add comparisons, differences and limits supported by this section's observed data.",
            )
            + "}\n"
        )

    def table(self, name, rows=None):
        table = self.data["tables"][name]
        cols = table["columns"]
        rows = table["rows"] if rows is None else rows
        if not rows:
            rows = [[None] * len(cols)]
        spec = "@{}P{38mm}" + "Y" * (len(cols) - 1) + "@{}"
        lines = [
            "{\\small",
            "\\begin{tabularx}{\\linewidth}{" + spec + "}\\toprule",
            " & ".join(esc(local(col, self.lang)) for col in cols) + r"\\\midrule",
        ]
        for row in rows:
            if len(row) != len(cols):
                raise ValueError(f"Table {name}: wrong cell count")
            lines.append(" & ".join(cell(value, self.lang) for value in row) + r"\\")
        lines.extend(
            [
                r"\bottomrule\end{tabularx}}",
                "\\noteline{" + esc(local(table.get("note", ""), self.lang)) + "}",
            ]
        )
        return "\n".join(lines) + "\n\n"

    def figure(self, name, zh, en, height="61mm"):
        path = self.data.get("figures", {}).get(name, {}).get(self.lang)
        if path:
            path = Path(path).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            body = (
                r"\includegraphics[width=\linewidth,height="
                + height
                + r",keepaspectratio]{\detokenize{"
                + path.as_posix()
                + "}}"
            )
        else:
            body = (
                r"\fbox{\parbox[c]["
                + height
                + r"][c]{0.94\linewidth}{\centering\color{pending}"
                + self.t(
                    "此处预留真实结果图；尚未填入数据。",
                    "Observed-result figure is unfilled; no data have been inserted.",
                )
                + "}}"
            )
        caption = re.sub(r"^(?:图\s*\d+|Figure\s+\d+)[.．]\s*", "", self.t(zh, en))
        return "\\begin{center}\n" + body + "\n\\end{center}\n\\reportcaption{" + caption + "}\n\n"

    def build_full(self):
        language = self.lang
        font = (
            (
                r'\XeTeXlinebreaklocale "zh"'
                + "\n"
                + r"\XeTeXlinebreakskip=0pt plus 1pt"
                + "\n"
                + r"\setmainfont{SimSun}[BoldFont=SimHei,ItalicFont=SimSun]"
            )
            if language == "zh"
            else r"\setmainfont{Times New Roman}"
        )
        out = [PREAMBLE.replace("FONT", font).replace("FIGURE_PREFIX", self.t("图", "Figure"))]
        add = out.append
        title = self.t(
            "鲁棒变量选择：实验结果与解读", "Robust variable selection: results and interpretation"
        )
        state = {
            "not_run": self.t("尚无已核查结果", "No verified results yet"),
            "in_progress": self.t("正式实验运行中", "Formal experiments in progress"),
            "partial": self.t(
                "部分结果，实验尚未全部完成", "Partial results; experiments incomplete"
            ),
            "complete": self.t("正式实验已完成", "Formal experiments completed"),
        }[self.data["state"]]
        add(
            "\\begin{center}{\\LARGE\\bfseries "
            + title
            + "}\\par\\vspace{5pt}\n{\\small "
            + esc(self.data["as_of"])
            + "\\quad "
            + state
            + "}\\end{center}\n"
        )
        add(self.heading("我们在检验什么", "What are we testing?"))
        add(
            self.paragraph(
                "实际只能保留少量测量变量。我们检验它们能保留多少类别信息、分布变化后是否仍有效，以及全局搜索的解质量能否补偿计算成本。",
                "We test how few measurements retain class information, whether they remain useful after distribution change, and whether global search improves solutions enough to justify its cost.",
            )
        )
        add(
            self.paragraph(
                "各方法选同样数量的变量，并拟合同一种鲁棒评分。鲁棒风险是允许两类训练分布在既定距离内移动时的最大损失。",
                "Methods select the same number of variables and fit the same robust score. Robust risk is the largest loss when training distributions may move within the allowed distance.",
            )
        )
        add(self.heading("报告完成到哪里", "What has actually been completed?"))
        coverage = self.data["coverage"]
        add(r"{\small\begin{tabularx}{\linewidth}{@{}P{48mm}YYYY@{}}\toprule" + "\n")
        add(
            self.t(
                "实验 & 计划数据 & 已有记录 & 核查完成 & 失败记录",
                "Experiment & Planned data & Recorded & Verified & Failed records",
            )
            + r"\\\midrule"
            + "\n"
        )
        for row in coverage:
            add(
                " & ".join(
                    [
                        esc(local(row["label"], language)),
                        str(row["planned_data"]),
                        cell(row["observed_data"], language),
                        cell(row["verified_data"], language),
                        cell(row["failed_data"], language),
                    ]
                )
                + r"\\"
                + "\n"
            )
        add(r"\bottomrule\end{tabularx}}" + "\n")
        add(
            self.paragraph(
                "E2 复用 120 套 E1 训练结果，重复计时也不增加独立数据。横线表示未填入。超时、失败和零评分兜底保留在统计分母内。",
                "E2 reuses 120 E1 fits; timing repeats add no independent data. Dashes mean unfilled. Timeouts, failures and zero-score fallbacks remain in their denominators.",
            )
        )
        add(self.heading("先统一读图方式", "How to read the results"))
        add(
            self.paragraph(
                "两类错误各占一半。评分 $f(x)\\in[-1,1]$ 对应接受类别零的概率 $p_0(x)=(1+f(x))/2$，主要指标为",
                "Weight class errors equally. Score $f(x)\\in[-1,1]$ sets randomized decision probability $p_0(x)=(1+f(x))/2$ of accepting class zero. The primary metric is",
            )
        )
        add(
            r"\[\mathrm{BER}_{\rm prob}=\tfrac12\left[\operatorname{mean}_{Y=0}(1-p_0(X))+\operatorname{mean}_{Y=1}p_0(X)\right].\]"
            + "\n"
        )
        add(
            self.paragraph(
                "错误率越低越好，$50\\%$ 表示没有类别判别信息的恒定预测水平。另报按评分正负直接判类的阈值错误率。合成数据中，若选中 $h$ 个真信号，精确率为 $h/d$，召回率为 $h/s$；召回高不一定代表变量用得节省。",
                "Lower error is better; a constant prediction has $50\\%$ balanced error. We also report error from classifying by the score's sign. In synthetic data, selecting $h$ true signals gives precision $h/d$ and recall $h/s$; high recall alone does not establish economical selection.",
            )
        )
        if self.data.get("findings"):
            add(self.heading("目前数据支持的结论", "Findings supported by the observed data"))
            add(
                "\\begin{itemize}\n"
                + "\n".join(
                    "\\item " + esc(local(value, language)) for value in self.data["findings"]
                )
                + "\n\\end{itemize}\n"
            )
        else:
            add(
                "\\unfilled{"
                + self.t(
                    "主要结论尚未填入；不根据设计预判方法排名。",
                    "Main findings are unfilled; the design does not predetermine a method ranking.",
                )
                + "}\n"
            )

        add(r"\clearpage" + "\n")
        add(self.heading("数据怎样生成与划分", "How data are generated and split"))
        add(
            self.paragraph(
                "每一行是一条带类别标签 $k=0,1$ 的记录，每一列是一个数值变量。$D$ 是候选变量总数，$n$ 是每类训练样本数，$s$ 是真正携带类别信息的变量数，$d$ 是允许选取的变量数。例如 $D=50,n=20,s=4$ 的训练表有四十行、五十列，四列信号和四十六列噪声。",
                "Each row is a record labelled $k=0,1$, and each column is a numerical variable. $D$ is the candidate count, $n$ the training count per class, $s$ the true signal count, and $d$ the allowed selected count. For example, $D=50,n=20,s=4$ gives forty training rows and fifty columns: four signals and forty-six noise variables.",
            )
        )
        add(
            self.paragraph(
                "合成数据先按下表为每行抽取 $s$ 个信号，再补 $D-s$ 个独立标准正态噪声。下表的 $Z_j$ 每行、每列独立重新抽取；混合模型的符号 $B=\\pm1$ 每行各以一半概率抽取。每次重复随机打乱列位置，同一排列用于各数据集和各方法。",
                "Generate each synthetic row's $s$ signals using the rules below, then append $D-s$ independent standard normal noise variables. Draw every $Z_j$ independently across rows and coordinates; in the mixture model, draw the row-specific sign $B=\\pm1$ with equal probability. Randomly permute columns in each repetition, sharing that permutation across splits and methods.",
            )
        )
        add(r"{\small\begin{tabularx}{\linewidth}{@{}P{23mm}X@{}}\toprule" + "\n")
        signals = [
            (bi("均值不同", "Mean difference"), r"$X_j=(2k-1)a+Z_j$."),
            (
                bi("波动不同", "Variance difference"),
                self.t(
                    r"类别零 $X_j=Z_j$；类别一 $X_j=rZ_j$。",
                    r"Class zero: $X_j=Z_j$; class one: $X_j=rZ_j$.",
                ),
            ),
            (
                bi("相关不同", "Correlation difference"),
                r"$X_{2h-1}=Z_{2h-1}$, $X_{2h}=(1-2k)\rho Z_{2h-1}+\sqrt{1-\rho^2}Z_{2h}$.",
            ),
            (
                bi("混合不同", "Mixture difference"),
                self.t(
                    r"$X_{1:s}=Ba v_k+Z$；$v_0$ 全为一，$v_1$ 前半为一、后半为负一。",
                    r"$X_{1:s}=Ba v_k+Z$; $v_0$ is all ones, and $v_1$ has a positive first half and negative second half.",
                ),
            ),
        ]
        for name, formula in signals:
            add(esc(local(name, language)) + " & " + formula + r"\\" + "\n")
        add(r"\bottomrule\end{tabularx}}" + "\n")
        add(
            self.paragraph(
                "四种模型都在运行方法前校准到同一难度：知道真实分布、观察全部信号的理想分类器错误率为 $20\\%$。相关与混合模型的单列分布在两类中相同，用来检查变量组合。各重复独立抽取每类二十条验证样本、一万条测试样本；训练、验证、测试不互相复制。",
                "Before fitting methods, calibrate every model to the same difficulty: an ideal classifier knowing the true distributions and observing all signals has $20\\%$ error. In correlation and mixture data, individual columns have identical class distributions, so discrimination requires combinations. Each repetition independently draws twenty validation and ten thousand test samples per class, without copying samples across roles.",
            )
        )
        add(
            self.paragraph(
                "主实验固定 $s=4$，三档 $(D,n)=(20,10),(50,20),(100,50)$ 都比较 $d=2,4,8$；中档另设 $s=2$、$d=1,2,4$ 与 $s=8$、$d=4,8,16$。四种模型、五个数据配置各重复三十次，共六百套数据、一千八百个预算任务。不同预算复用同一套数据。",
                "The main experiment uses $s=4$ at three scales, $(D,n)=(20,10),(50,20),(100,50)$, each with $d=2,4,8$. At the medium scale also use $s=2$ with $d=1,2,4$, and $s=8$ with $d=4,8,16$. Four models and five data configurations, with thirty repetitions each, give six hundred datasets and eighteen hundred budget tasks. Budgets share the same data.",
            )
        )
        add(
            self.heading(
                "四组公开数据补充真实测量条件",
                "Four public datasets add real measurement conditions",
            )
        )
        add(
            r"{\small\begin{tabularx}{\linewidth}{@{}P{25mm}P{34mm}X@{}}\toprule"
            + "\n"
            + self.t("数据 & 原始规模 & 区分什么", "Dataset & Original size & Classification task")
            + r"\\\midrule"
            + "\n"
        )
        public = [
            ("WDBC", "$569\\times30$", bi("细胞良性与恶性", "Benign versus malignant cells")),
            ("Ionosphere", "$351\\times34$", bi("雷达好坏回波", "Good versus bad radar returns")),
            ("Sonar", "$208\\times60$", bi("声呐中的岩石与金属", "Rock versus metal in sonar")),
            (
                "UCI HAR",
                "$10{,}299\\times561$",
                bi(
                    "手机测量的坐与站，三十名受试者",
                    "Sitting versus standing from smartphone measurements; thirty people",
                ),
            ),
        ]
        for name, count, purpose in public:
            add(name + " & " + count + " & " + esc(local(purpose, language)) + r"\\" + "\n")
        add(r"\bottomrule\end{tabularx}}" + "\n")
        add(
            self.paragraph(
                "前三组每类按 $60\\%$、$20\\%$ 与其余分为训练池、验证、测试，训练与验证数向下取整；抽取每类 $10,20,40$ 个训练样本，较小训练集包含在较大集中。每组十个共享划分；预算四比较样本量，每类二十训练样本时比较预算 $1,2,4,8$。HAR 使用官方二十一名训练者和九名测试者；训练侧每次留六人验证，余下十五人均匀抽每类四十窗口，重复十次、预算 $8,16,32$。同人同活动连续区段隔一个保留一个窗口，评价先按人等权。",
                "For the first three datasets, split each class into a $60\\%$ training pool, $20\\%$ validation and the remainder for testing, rounding training and validation down. Draw $10,20,40$ training samples per class with nested sets. Use ten shared splits; compare sample sizes at budget four and budgets $1,2,4,8$ at twenty training samples per class. HAR retains the official twenty-one training people and nine test people. In each of ten repetitions, reserve six training-side people for validation and sample forty windows per class evenly across the other fifteen; use budgets $8,16,32$. Keep alternate windows within continuous person/activity segments and weight test people equally.",
            )
        )

        add(r"\clearpage" + "\n")
        add(
            self.heading(
                "比较哪些方法，怎样保持公平",
                "Which methods are compared, and how is the comparison kept fair?",
            )
        )
        add(
            self.paragraph(
                "先区分两件事：选择变量的目标好不好，以及优化同一个目标时怎样搜索。前四种方法优化同一个鲁棒风险，后五种提供不同的选变量依据。全部方法都只用训练数据选变量，并在同一预算下重新拟合同一个鲁棒评分。",
                "Separate the selection criterion from the search procedure. The first four procedures below optimize the same robust risk; the remaining five provide different selection criteria. Every method selects variables using training data and refits the same robust score at the same budget.",
            )
        )
        add(
            r"{\small\begin{tabularx}{\linewidth}{@{}P{32mm}X@{}}\toprule"
            + "\n"
            + self.t(
                "方法 & 它做什么，以及为什么保留", "Method & What it does and why it is included"
            )
            + r"\\\midrule"
            + "\n"
        )
        methods = [
            (
                bi("穷举", "Enumeration"),
                bi(
                    "逐一拟合所有同预算选集，只用于小问题，提供最优值参照。",
                    "Fit every equal-budget subset; used only on small problems to obtain an optimal-value reference.",
                ),
            ),
            (
                bi("全局搜索", "Global search"),
                bi(
                    "当前算法。用已评估选集产生的风险下界缩小搜索范围，返回可行解及还未排除的改进空间。",
                    "Our algorithm. Use risk lower bounds from evaluated subsets to reduce the search, returning a feasible solution and the remaining possible improvement.",
                ),
            ),
            (
                bi("前向 / 前向加交换", "Forward / forward with swaps"),
                bi(
                    "每次加入使风险最低的一个变量；后者再尝试一换一改进。比较较局部的搜索是否已足够。",
                    "Add the variable yielding the lowest risk at each step; the latter then tries one-for-one improvements. Test whether a more local search is sufficient.",
                ),
            ),
            (
                bi("逐变量 MMD", "Marginal MMD"),
                bi(
                    "每列分别比较两类的分布差异，再按差异排序。MMD（最大均值差异）比较类内与类间的样本相似度，能看到均值以外的单列变化。",
                    "Rank columns by their class-distribution difference. MMD (maximum mean discrepancy) compares within-class and between-class similarities, detecting more than a change in the mean.",
                ),
            ),
            (
                bi("L1 逻辑回归", "L1 logistic regression"),
                bi(
                    "拟合带系数惩罚的线性分类器，选择绝对系数最大的变量；代表常见稀疏线性选择。",
                    "Fit a penalized linear classifier and select the largest absolute coefficients, representing standard sparse linear selection.",
                ),
            ),
            (
                "mRMR",
                bi(
                    "优先选择与类别有关、又不重复已选变量信息的列；用互信息衡量依赖，训练三分位点离散成三档。",
                    "Prefer label-related variables without duplicating selected information; measure dependence by mutual information after binning at training tertiles.",
                ),
            ),
            (
                bi("联合 MMD", "Joint MMD"),
                bi(
                    "将多个变量作为整体比较两类，用前向和交换增大差异；检查另一种可利用变量关系的选择依据。",
                    "Compare class distributions on a group of variables, using forward steps and swaps to increase the difference; a joint criterion that can use dependencies.",
                ),
            ),
            (
                bi("均匀随机", "Uniform random"),
                bi(
                    "按固定种子随机抽同样数量变量，提供无选择依据的参照。",
                    "Draw an equal-size subset with a saved seed, providing an uninformed selection reference.",
                ),
            ),
        ]
        for name, reason in methods:
            add(esc(local(name, language)) + " & " + esc(local(reason, language)) + r"\\" + "\n")
        add(r"\bottomrule\end{tabularx}}" + "\n")
        add(
            self.paragraph(
                "两种 MMD 使用高斯相似度，带宽为当前选集训练样本正距离的中位数。分数等于两类平均类内相似度之和，减去两倍平均类间相似度，类内不含样本自身。联合 MMD 和鲁棒前向加交换最多接受十次改进。真实信号全部列与全变量检验仅作为解释参照，不参加同预算排名。",
                "Both MMD methods use Gaussian similarity with bandwidth equal to the median positive training distance on the current subset. Their score adds the two within-class mean similarities and subtracts twice the between-class mean, excluding within-class self-pairs. Joint MMD and robust forward with swaps accept at most ten improvements. All true signals and all variables are interpretive references, excluded from equal-budget rankings.",
            )
        )
        add(self.heading("调参不接触测试数据", "Parameter selection never accesses test data"))
        add(
            self.paragraph(
                "只用合并训练样本的均值和标准差标准化变量，删除训练中不变化的列。鲁棒半径表示允许训练分布移动多远。以每类训练两两绝对距离之和的中位数除以保留维数为尺度，比较其 $1/8,1/4,1/2,1,2$ 倍；在参考预算四（HAR 十六）下，由逐变量 MMD、mRMR、联合 MMD 的平均验证错误率选择共同半径，随后所有方法和预算固定它。L1 参数从 $0.01,0.1,1,10,100$ 中验证选择；平局取更强惩罚、较大半径和较小变量索引。",
                "Standardize using pooled training means and standard deviations only, dropping training-constant columns. The robust radius sets the allowed distribution movement. Its scale is the classwise median pairwise sum of absolute training differences divided by retained dimension. Compare multipliers $1/8,1/4,1/2,1,2$; at reference budget four (sixteen for HAR), select one common radius by mean validation error from marginal MMD, mRMR and joint MMD. Fix it for every method and budget. Validate L1 parameters among $0.01,0.1,1,10,100$; ties favor stronger regularization, a larger radius and a smaller column index.",
            )
        )
        add(
            self.paragraph(
                "合成结果按三十套独立数据汇总；方法差值在同一套数据内相减，再用二千次配对重采样得到 $95\\%$ 区间。表格公开数据只将十个划分的变化解释为划分敏感性。HAR 先在测试者内平均训练重复，再对九个人等权与重采样，不能把窗口或重复当成新受试者。",
                "Summarize synthetic results over thirty independent datasets. Compute method differences within each dataset, then use two thousand paired bootstrap resamples for $95\\%$ intervals. The ten tabular splits measure split sensitivity, rather than independent test populations. For HAR, average training repetitions within test person, then equally weight and resample the nine people; windows and repetitions do not become new people.",
            )
        )

        add(r"\clearpage" + "\n")
        add(
            self.heading(
                "E1：少量变量保留多少类别信息",
                "E1: How much class information do a few variables retain?",
            )
        )
        add(
            self.paragraph(
                "先看四种信息来源下预算变化是否有效，再检查结论是否依赖某一档规模或真实信号数。主图固定中档和四个信号，避免把数据难度变化混进一条曲线。",
                "First ask whether additional measurement budget helps across four sources of class information, then check dependence on scale and true signal count. The main plot fixes medium scale and four signals, avoiding a curve that mixes changing data difficulty.",
            )
        )
        add(
            self.figure(
                "main",
                "图 1. 中档 D=50、每类训练 n=20、信号 s=4，预算 d=2,4,8。主图固定展示全局、前向加交换、联合 MMD、随机选择；其余方法保留在数值表中。点为已完成数据的平均概率错误率，误差线为数据间标准差。虚线 50% 是无信息参照，点线 20% 是知道真实分布且观察全部信号的理想错误率。计划每模型三十套数据，实际完成数见本页。",
                "Figure 1. Medium scale D=50, n=20 training samples per class, s=4 signals and budgets d=2,4,8. Curves consistently show global, forward with swaps, joint MMD and random selection; numeric tables retain the other methods. Points are observed mean probabilistic errors and bars show across-dataset SD. The dashed 50% line is the constant-prediction reference; the dotted 20% line is ideal error with known distributions and all signals. Thirty datasets are planned per model; observed counts are stated on this page.",
                "63mm",
            )
        )
        add(self.table("main"))
        if self.data.get("display_counts", {}).get("main"):
            add(esc(local(self.data["display_counts"]["main"], self.lang)) + "\n\n")
        add(self.commentary("main"))
        add(
            self.heading(
                "扩大规模、改变信号数后是否保持",
                "Does the pattern persist across scales and signal counts?",
            )
        )
        add(self.table("generality"))
        add(self.commentary("generality"))
        add(
            self.paragraph(
                "信号恢复和测试错误率需要一起读。混合模型可能只凭跨两个半区的一对变量就能判别，未恢复全部信号并不等于选集无用。两信号相关或混合模型在预算一时没有单列类别信息，这一限制不能被算法消除。",
                "Read signal recovery together with test error. A mixture may be discriminated by one pair spanning its two halves, so incomplete support recovery need not imply an unhelpful subset. With two-signal correlation or mixture data and budget one, no individual variable carries class information; an algorithm cannot remove this restriction.",
            )
        )

        add(r"\clearpage" + "\n")
        add(self.heading("E1：公开数据中的表现", "E1: Performance on public data"))
        add(
            self.paragraph(
                "前三组检查小样本测量数据，HAR 检查从已见的人推广到未见的人。图中分别改变样本量和观测预算，不把两种原因合为同一个效果。",
                "The first three datasets test small labelled measurement samples; HAR tests transfer from seen to unseen people. Vary training size and measurement budget separately, so their effects are not conflated.",
            )
        )
        add(
            self.figure(
                "public",
                "图 2. 中心条件的全部八种选择方法：WDBC、Ionosphere、Sonar 固定每类训练 n=20、预算 d=4；HAR 固定 n=40、d=16。点为平均概率错误率，横线为标准差；前三组按共享划分统计，HAR 按九名测试者统计。此图不代表样本量或预算扫描结果。",
                "Figure 2. All eight selection methods at the central conditions: n=20 training samples per class and budget d=4 for WDBC, Ionosphere and Sonar; n=40 and d=16 for HAR. Points show mean probabilistic error and horizontal bars SD across shared splits for the first three datasets, or across nine test people for HAR. This view does not display sample-size or budget scans.",
                "56mm",
            )
        )
        add(self.table("public"))
        add(self.commentary("public"))
        add(self.heading("与原论文混合数据的对应", "Connection to the original mixture experiment"))
        add(
            self.paragraph(
                "另行使用原论文的一百维混合生成式：$X=0.4Be+Z$ 与 $X=0.4Bf+Z$。每条记录独立抽 $B=\\pm1$，各占一半；$e$ 全为一，$f$ 前五十项为一、后五十项为负一，$Z$ 为标准正态噪声。每类训练十条、验证二十条、测试一千条，重复五百次，只拟合全变量检验；没有唯一的四变量真支持。",
                "A separate anchor uses the original 100-dimensional mixtures, $X=0.4Be+Z$ and $X=0.4Bf+Z$. Independently draw each row's sign $B=\\pm1$ with equal probability; $e$ is all ones, $f$ has fifty positive and fifty negative entries, and $Z$ is standard normal noise. Use ten training, twenty validation and one thousand test samples per class, five hundred repetitions and all variables. There is no unique four-variable true support.",
            )
        )
        add(
            self.paragraph(
                "训练鲁棒风险的一半，是所设分布范围内最坏错误率的上界；测试错误率来自独立新样本。阈值检验将非负评分判为类别零、负评分判为类别一；同时报告各类错误，避免平均值掩盖单类失败。",
                "Half the fitted robust risk bounds the worst balanced error over the specified distribution sets; test error is measured on independent observations. The threshold test assigns nonnegative scores to class zero and negative scores to class one. Classwise errors expose failures hidden by an average.",
            )
        )
        add(self.table("anchor"))
        add(self.commentary("anchor"))

        add(r"\clearpage" + "\n")
        add(
            self.heading(
                "E2：测试分布变化后是否仍有用",
                "E2: Do the selected variables remain useful after a change?",
            )
        )
        add(
            self.paragraph(
                "训练结果、选集和参数全部固定，只改变测试样本。在中档四信号、预算四的每种模型三十套数据上，分别平移信号坐标或加入正态测量噪声。用总绝对位移的期望 $\\delta$ 衡量变化，比较 $\\delta/\\bar\\theta=0,1/2,1,2$；$\\bar\\theta$ 是两类训练半径的平均值。两种变化的期望绝对位移匹配，各方法接收完全相同的样本。",
                "Fix every fitted model, selected subset and parameter, changing test samples only. For each model, use thirty medium-scale datasets with four signals and budget four, translating signal coordinates or adding normal measurement noise. Measure change by expected total absolute displacement $\\delta$, using $\\delta/\\bar\\theta=0,1/2,1,2$, where $\\bar\\theta$ averages the two training radii. Match expected displacement between transformations and give every method the same changed samples.",
            )
        )
        add(
            self.figure(
                "shift",
                "图 3. 固定 E1 评分后，测试样本整体平移或增加噪声时的平均概率错误率。各方法共享配对样本；只汇总已有对应 E1 训练记录的数据，计划每模型三十套。主图与选择效果主比较使用相同四种方法，不绘制误差线。曲线增量描述所设变化下的退化，不是任意测试变化下的保证。",
                "Figure 3. Mean probabilistic error after translating test samples or adding noise, with E1 scores fixed. Methods share paired samples; only datasets with matching E1 training records contribute, out of thirty planned per model. Curves use the same four methods as the main selection comparison, without error bars. Their increase describes degradation under the specified changes, rather than a guarantee for arbitrary test shifts.",
                "62mm",
            )
        )
        add(self.table("shift"))
        add(self.commentary("shift"))
        add(
            self.heading(
                "小消融：保护更强是否一定更好",
                "Small ablation: is stronger protection always better?",
            )
        )
        add(
            self.paragraph(
                "另取二十套中档数据，均值、混合模型各十套，比较选定半径的二分之一、一倍、两倍。固定数据、预算和其余设置，不按测试结果重新选半径。",
                "Use twenty additional medium-scale datasets, ten each for mean and mixture, comparing half, one and twice the selected radius. Hold data, budget and all other settings fixed; do not select a new radius from test results.",
            )
        )
        add(self.table("radius"))
        add(self.commentary("radius"))

        add(r"\clearpage" + "\n")
        add(
            self.heading(
                "E3：搜索质量与证明成本", "E3: Search quality and the cost of certification"
            )
        )
        add(
            self.paragraph(
                "先在可完整穷举的小问题上检查解，再在较大问题上衡量时间成本。可行解的训练风险上界记为 $U$，对所有选集有效的下界记为 $L$；差距 $U-L$ 是仍未排除的改进空间。找到一个好解与证明几乎不能更好，是两件不同的事。",
                "First check solutions where full enumeration is possible, then measure search cost on larger problems. Let $U$ be a feasible solution's training-risk upper bound and $L$ a valid lower bound over all subsets. The gap $U-L$ is improvement not yet ruled out. Finding a good solution and establishing that little improvement remains are distinct tasks.",
            )
        )
        add(
            self.paragraph(
                "为单独比较搜索，半径固定为各类训练样本两两 $\\ell_1$ 距离中位数除以 $4D$（$\\ell_1$ 是坐标绝对差之和），核查容差为 $10^{-8}$。",
                "To compare search separately, fix each class's radius at its median pairwise training $\\ell_1$ distance divided by $4D$ ($\\ell_1$ sums absolute coordinate differences), with verification tolerance $10^{-8}$.",
            )
        )
        add(
            self.figure(
                "search",
                "图 4. 左侧为各模型、训练样本量条件下五次重复的平均超额风险，每方法八个点，颜色区分模型。独立运输线性规划提供最优参照，穷举与全局在零附近重叠。右侧为平均拟合次数。下表时间来自两个并发任务，不代表独占硬件速度。",
                "Figure 4. Condition means over five repeats; colors/shapes identify models, open/filled points denote n=5/10, and black ticks mark the overall mean. Left: excess risk above the independent optimum. Right: fixed-subset fits. Table timings reflect two concurrent jobs, not isolated hardware.",
                "63mm",
            )
        )
        add(self.table("small"))
        add(self.table("checks"))
        add(self.commentary("small"))
        add(self.commentary("checks"))
        add(
            self.paragraph(
                "小问题为 $D=10,s=4,d=3$，每类训练五或十条，四模型各五次，共四十套；每套枚举 120 个三列选集，独立核查全部 176 个至多三列选集。规模实验以 $D=50,n=20,d=4,s=4$ 为中心，分别扫描 $D=20,50,100,200$、$n=10,20,50,100$ 或 $d=2,4,8,12$。均值与混合各十个唯一条件、每条件十次，共二百套；二十套中心各计时三次，其余一次。每套先汇总重复计时，再汇总独立数据。",
                "Small problems use $D=10,s=4,d=3$, five or ten training samples per class and five repeats of each of four models, giving forty datasets. Enumerate 120 size-three subsets and independently check all 176 subsets of size at most three per dataset. Scaling centres on $D=50,n=20,d=4,s=4$, scanning $D=20,50,100,200$, $n=10,20,50,100$ or $d=2,4,8,12$. Mean and mixture each have ten unique conditions and ten repeats, giving two hundred datasets. Time the twenty centre instances three times, all others once; collapse timing repeats before aggregating datasets.",
            )
        )

        add(r"\clearpage" + "\n")
        add(self.heading("多花的计算换来了什么", "What does the additional computation buy?"))
        add(self.commentary("scaling"))
        add(
            self.paragraph(
                "全局搜索在 10、60、300 秒的证明完成比例，应以全部计划实例为分母，缺失或未达到不能算作零秒完成。贪心方法只有可行风险，不能给它附加全局最优证书。最先达到最终最好解的时间与达到目标差距的时间应同时报告，避免把证明开销误读为找解开销。",
                "At 10,60,300 seconds, report the fraction certified by global search with all planned instances in the denominator. Missing or unreached certificates are not zero-second successes. Greedy methods provide feasible risk without a global optimality certificate. Report both first arrival at the final best risk and time to the target gap, separating solution discovery from proof cost.",
            )
        )
        add(
            self.heading(
                "小消融：初始化是否影响结果", "Small ablation: does initialization matter?"
            )
        )
        add(
            self.paragraph(
                "复用半径消融的二十套数据，在原选定半径下，将全局搜索的逐变量 MMD 初始化换为一次随机选集。只改变起点，观察最终风险、时间与剩余差距。",
                "Reuse the twenty radius-ablation datasets at the originally selected radius, replacing marginal-MMD initialization with one random subset. Change only the starting subset and compare final risk, runtime and remaining gap.",
            )
        )
        add(self.table("initialization"))
        add(self.commentary("initialization"))
        add(
            self.heading(
                "现在能说什么，还有哪些限制", "What can be concluded, and what remains limited?"
            )
        )
        add(self.commentary("conclusions"))
        add(self.commentary("limitations"))
        add(
            self.paragraph(
                "主实验回答的是小样本下、指定预算和共同预测规则中的选择效果；公开数据的十次划分重复使用同一数据集，不能解释成十个独立人群。分布变化只覆盖本报告列出的平移和噪声。预测优势、信号恢复、训练目标与求解证明必须分别判断，不能互相替代。",
                "The main experiment evaluates selection with small labelled samples, a stated budget and a common predictor. Ten public-data splits reuse a dataset rather than representing ten independent populations. Shift tests cover the specified translations and noise only. Predictive gains, signal recovery, training objective and optimization certification must be assessed separately.",
            )
        )
        add(self.heading("实际运行约定", "Actual execution conditions"))
        add(self.table("execution"))
        add(
            self.paragraph(
                "所有方法计入初始化、候选比较、交换和最终拟合；共同半径选择耗时单列，并计入总耗时。失败保留零评分兜底的预测，超时保留已核查的最好可行解。开发检查独立于正式数据，不并入本报告的方法效果。",
                "Charge initialization, candidate comparisons, swaps and final fitting to each method. List common radius-selection time separately and include it in total cost. Retain zero-score fallback predictions for failures and the best verified feasible solution at timeout. Development checks use separate data and are excluded from method-effect results.",
            )
        )
        if self.data.get("readable_data_path"):
            add(
                "\\noteline{"
                + self.t("完整数值表：", "Complete numeric tables: ")
                + r"\path{"
                + str(self.data["readable_data_path"]).replace("\\", "/")
                + "}}\n"
            )
        add("\\end{document}\n")
        return "".join(out)

    def build(self):
        if self.lang == "zh":
            from explain_experiments_zh import build_chinese_report

            return build_chinese_report(self)
        full = self.build_full()
        if self.data["state"] == "complete" or self.data.get("layout") == "full-design":
            return full
        sections = set(self.data.get("available_sections", []))
        for name in ("main", "public"):
            # Early main records may cover only the small D=20 condition,
            # whereas the predeclared presentation plot shows medium D=50.
            # Keep their coverage count without adding an empty result page.
            has_cells = any(
                value is not None for row in self.data["tables"][name]["rows"] for value in row[1:]
            )
            has_figure = self.data.get("figures", {}).get(name, {}).get(self.lang)
            if not has_cells and not has_figure:
                sections.discard(name)
        pages = full.split("\\clearpage\n")
        if len(pages) != 8:
            raise ValueError("Partial-report page boundaries changed; review the template")
        overview = pages[0].replace(
            self.t(
                "鲁棒变量选择：实验结果与解读",
                "Robust variable selection: results and interpretation",
            ),
            self.t("实验记录：首轮结果", "Experimental record: first results"),
        )
        opening_note = self.paragraph(
            "本版只记录已有结果，保留未完成计划。$D$ 是候选变量数，$n$ 是每类训练数，$s$ 是信号数，$d$ 是所选变量数。",
            "This edition reports observed results and retains unfinished plans. $D$ counts candidates, $n$ training samples per class, $s$ signals and $d$ selected variables.",
        )
        marker = self.heading("我们在检验什么", "What are we testing?")
        overview = overview.replace(marker, opening_note + marker, 1)
        selected = [overview, pages[1], pages[2]]
        if "small" in sections or "scaling" in sections:
            page = pages[6]
            if "scaling" not in sections:
                page = page.replace(
                    self.paragraph(
                        "先在可完整穷举的小问题上检查解，再在较大问题上衡量时间成本。可行解的训练风险上界记为 $U$，对所有选集有效的下界记为 $L$；差距 $U-L$ 是仍未排除的改进空间。找到一个好解与证明几乎不能更好，是两件不同的事。",
                        "First check solutions where full enumeration is possible, then measure search cost on larger problems. Let $U$ be a feasible solution's training-risk upper bound and $L$ a valid lower bound over all subsets. The gap $U-L$ is improvement not yet ruled out. Finding a good solution and establishing that little improvement remains are distinct tasks.",
                    ),
                    self.paragraph(
                        "本页用 $D=10,s=4,d=3$、每类训练五或十条，四模型各五次，共四十套数据。穷举每套 120 个三列选集；独立运输 LP 核查全部 176 个至多三列选集。前向每次加一列，交换再尝试一列替换。可行风险上界 $U$ 与全局下界 $L$ 之差 $U-L$，表示仍可能改进多少。",
                        "Use $D=10,s=4,d=3$ and five or ten training samples per class: four models with five repeats give forty datasets. Enumerate 120 size-three subsets; independent transport LPs check all 176 subsets of size at most three. Forward adds one variable at a time; swaps try replacements. Feasible upper bound $U$ and global lower bound $L$ leave gap $U-L$ for possible improvement.",
                    ),
                )
                # Keep the exact small design here; the scaling grid is restated
                # on the pending-plan page rather than implying an observed result.
                prefix = self.t("规模实验以", "Scaling centres on")
                position = page.find(prefix)
                if position >= 0:
                    page = page[:position].rstrip() + "\n"
                repeated_setup = page.rfind(self.t("小问题为 $D=10", "Small problems use $D=10"))
                if repeated_setup >= 0:
                    page = page[:repeated_setup].rstrip() + "\n"
            selected.append(page)
        if "anchor" in sections:
            anchor_title = self.heading(
                "与原论文混合数据的对应", "Connection to the original mixture experiment"
            )
            anchor = pages[4][pages[4].index(anchor_title) :]
            marker = self.table("anchor")
            if self.data.get("figures", {}).get("anchor", {}).get(self.lang):
                anchor = anchor.replace(
                    marker,
                    self.figure(
                        "anchor",
                        "原模型的训练风险与测试错误。阈值预测偏向同一类别，两类各自的错误率揭示这一失败；低训练风险没有转为低测试错误。",
                        "Original mixture: training risk and test error. Classwise errors reveal threshold predictions favoring one class; low training risk has not translated into low test error.",
                        "64mm",
                    )
                    + marker,
                    1,
                )
            selected.append(anchor)
        if "main" in sections:
            selected.append(pages[3])
        if "public" in sections:
            public_page = pages[4]
            anchor_title = self.heading(
                "与原论文混合数据的对应", "Connection to the original mixture experiment"
            )
            selected.append(public_page[: public_page.index(anchor_title)])
        if "shift" in sections:
            shift_page = pages[5]
            if "ablation" not in sections:
                ablation_title = self.heading(
                    "小消融：保护更强是否一定更好",
                    "Small ablation: is stronger protection always better?",
                )
                shift_page = shift_page[: shift_page.index(ablation_title)]
            selected.append(shift_page)
        # The brief English report describes data and methods before full results.
        ending = self.heading("其余实验的计划与进度", "Remaining experiments: plan and progress")
        ending += self.paragraph(
            "E1 合成主比较保留六百套独立数据、每套三个预算，覆盖四种信号、三档规模和不同信号数；公开数据保留三种表格数据和按人划分的 HAR。E2 在 E1 中档四信号、预算四的训练结果上，只改变测试分布，分别加入平移和测量噪声；对应 E1 记录尚未完成时，不重新训练来补这一实验。",
            "E2 reuses 120 medium-scale E1 fits at s=d=4, keeping selection and prediction fixed. Translate test signals or add Gaussian noise with matched expected total absolute displacement $\\delta$; use $\\delta/\\bar\\theta=0,0.5,1,2$, where $\\bar\\theta$ averages the class radii. Unfinished source fits are not replaced by retraining.",
        )
        ending += self.paragraph(
            "E3 规模实验以 $D=50,n=20,d=4,s=4$ 为中心，分别扫描 $D=20,50,100,200$、$n=10,20,50,100$、$d=2,4,8,12$，每次只改一项。均值、混合两模型共二百套独立数据，二十套中心各计时三次，其余一次。另有二十套新数据检查半径二分之一、一倍、两倍，以及一次随机初始化。这些重复计时或消融设置均不增加独立数据数。",
            "E3 scaling centres on $D=50,n=20,d=4,s=4$, scanning $D=20,50,100,200$, $n=10,20,50,100$ or $d=2,4,8,12$, one factor at a time. Mean and mixture give two hundred independent datasets; time twenty centre instances three times and the others once. Twenty additional datasets examine half, one and twice the radius, plus one random initialization. Timing and ablation repeats do not increase independent-data counts.",
        )
        ending += self.heading("当前结论与边界", "Current conclusions and limits")
        early_rows = self.data["tables"]["generality"]["rows"]
        if "main" not in sections and early_rows:
            ending += self.paragraph(
                "下面仅列已有的预算等于信号数的主实验观测；重复数未满，不能支持总体排名。",
                "Early E1 observations at budget equal to signal count are shown below. Repetitions are incomplete; these do not support an overall ranking.",
            )
            ending += self.table("generality") + self.commentary("generality")
        ending += self.commentary("conclusions") + self.commentary("limitations")
        ending += self.paragraph(
            "结果来自10月18日冻结的实现；新版理论补充的变量影响与搜索次数上界尚未专门做实验核查。",
            "Results use the implementation frozen on October 18. The newer analysis of coordinate influence and iteration bounds has not been experimentally evaluated here.",
        )
        ending += self.heading("实际运行条件", "Actual execution conditions") + self.table(
            "execution"
        )
        ending += self.paragraph(
            "共同半径选择耗时单列，并计入总耗时。超时保留已核查的最好可行结果，失败保留零评分兜底及其错误率。首页的核查完成表示记录通过完整性检查与预定实例检查，并不表示所有搜索都有全局最优证明。开发数据不并入正式方法结果。",
            "Search and radius-selection costs are recorded separately and jointly. Timeouts retain feasible incumbents; failed fits retain zero-score predictions. Development data are excluded.",
        )
        ending += "\\end{document}\n"
        selected.append(ending)
        return "\\clearpage\n".join(selected)


def pooled_metric(groups, metric, *, percent=False):
    """Pool per-condition sample summaries without treating SD as an SE."""
    measures = [
        group["metrics"][metric]
        for group in groups
        if group.get("metrics", {}).get(metric, {}).get("n", 0)
        and group["metrics"][metric].get("mean") is not None
    ]
    if not measures:
        return None
    count = sum(item["n"] for item in measures)
    mean = sum(item["n"] * item["mean"] for item in measures) / count
    ss = sum(
        (item["n"] - 1) * (item.get("sd") or 0) ** 2 + item["n"] * (item["mean"] - mean) ** 2
        for item in measures
    )
    return {
        "mean": mean,
        "sd": math.sqrt(max(0, ss / (count - 1))) if count > 1 else None,
        "n": count,
        "scale": 100 if percent else 1,
        "digits": 1 if percent else 4,
    }


def from_summary(summary_path):
    """Adapt the repository's tested summary schema to the compact view.

    Exact small-problem audit totals must still be supplied from the raw
    verification objects; this adapter deliberately leaves those cells unfilled.
    It never promotes a partial suite or an imputed metric to completed status.
    """
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    data = default_data()
    data["summary_source"] = str(Path(summary_path).resolve())
    data["state"] = "partial" if summary.get("groups") else "in_progress"
    data["findings"] = []
    data["available_sections"] = sorted(
        {item["group"]["suite"] for item in summary.get("groups", []) if item.get("n_instances", 0)}
    )
    by_suite = defaultdict(list)
    for item in summary.get("protocol_coverage", []):
        manifest = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        by_suite[manifest["suite"]].append(item)
    failure_counts = defaultdict(int)
    for failed in summary.get("failed_instances", []):
        spec = failed["spec"]
        suite = spec.get("metadata", {}).get("suite", spec["kind"])
        failure_counts[suite] += 1
    for row in data["coverage"]:
        items = by_suite.get(row["suite"], [])
        # No protocol means genuinely not started, rather than unknown results.
        observed = sum(item["valid_record_instances"] for item in items)
        row["observed_data"] = observed
        row["failed_data"] = failure_counts[row["suite"]]
        row["verified_data"] = observed - row["failed_data"]
        row["frozen_instances"] = sum(item["planned_instances"] for item in items)
    groups = summary.get("groups", [])

    def choose(**criteria):
        return [
            item
            for item in groups
            if all(item["group"].get(key) == value for key, value in criteria.items())
        ]

    for index, (method, _, _) in enumerate(METHODS):
        for model_index, model in enumerate(("mean", "variance", "correlation", "mixture"), 1):
            selected = choose(
                suite="main", D=50, n=20, s=4, budget=4, method=method, role="comparison"
            )
            selected = [item for item in selected if item["group"]["model"] == model]
            data["tables"]["main"]["rows"][index][model_index] = pooled_metric(
                selected, "ber_prob", percent=True
            )
        for dataset_index, model in enumerate(("wdbc", "ionosphere", "sonar", "har"), 1):
            selected = choose(
                suite="public",
                model=model,
                n=40 if model == "har" else 20,
                budget=16 if model == "har" else 4,
                method=method,
                role="comparison",
            )
            if model == "har":
                # All nine people recur across manifests; combining summary
                # means would inflate their count. Require one unambiguous group.
                if len(selected) == 1:
                    measure = selected[0]["metrics"].get("ber_prob", {})
                    if measure.get("mean") is not None:
                        data["tables"]["public"]["rows"][index][dataset_index] = {
                            "mean": measure["mean"],
                            "ci95": measure.get("ci95"),
                            "scale": 100,
                            "digits": 1,
                            "n": measure["n"],
                        }
            else:
                data["tables"]["public"]["rows"][index][dataset_index] = pooled_metric(
                    selected, "ber_prob", percent=True
                )
    anchor = choose(suite="anchor", role="reference")
    data["tables"]["anchor"]["rows"][0][1] = pooled_metric(anchor, "ber_prob", percent=True)
    data["tables"]["anchor"]["rows"][0][2] = pooled_metric(anchor, "ber_hard", percent=True)
    data["tables"]["anchor"]["rows"][0][3] = sum(item["n_instances"] for item in anchor)
    for index, method in enumerate(("enumeration", "global", "forward", "forward_swap")):
        selected = choose(suite="small", method=method, role="comparison")
        for column, metric in enumerate(("excess_risk", "runtime", "fit_count"), 1):
            data["tables"]["small"]["rows"][index][column] = pooled_metric(selected, metric)
    data["tables"]["generality"]["rows"] = []
    data["readable_data_path"] = str(Path(summary_path).parent / "groups.csv")
    central_counts = {}
    for model in ("mean", "variance", "correlation", "mixture"):
        central_counts[model] = sum(
            item["n_instances"]
            for item in choose(
                suite="main",
                model=model,
                D=50,
                n=20,
                s=4,
                budget=4,
                method="global",
                role="comparison",
            )
        )
    data["display_counts"] = {
        "main": bi(
            "本版中档四信号已记录数据数：均值 {mean}/30、波动 {variance}/30、相关 {correlation}/30、混合 {mixture}/30。未记录的条件不补值。".format(
                **central_counts
            ),
            "Observed medium-scale four-signal datasets: mean {mean}/30, variance {variance}/30, correlation {correlation}/30, mixture {mixture}/30. Unrecorded conditions are not imputed.".format(
                **central_counts
            ),
        )
    }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--write-example", type=Path)
    parser.add_argument(
        "--from-summary",
        type=Path,
        help="Read the repository analysis summary and write compact JSON to input",
    )
    parser.add_argument("--allow-unfilled", action="store_true")
    args = parser.parse_args()
    if args.from_summary:
        if args.input is None:
            parser.error("Provide an output JSON path as input when using --from-summary")
        args.input.parent.mkdir(parents=True, exist_ok=True)
        args.input.write_text(
            json.dumps(from_summary(args.from_summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if args.output_dir is None:
            print(
                "Wrote compact presentation JSON; narratives and exact-check counts remain unfilled."
            )
            return
    if args.write_example:
        args.write_example.parent.mkdir(parents=True, exist_ok=True)
        args.write_example.write_text(
            json.dumps(default_data(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return
    if args.input is None or args.output_dir is None:
        parser.error("input and --output-dir are required unless using --write-example")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if data["state"] not in {"not_run", "in_progress", "partial", "complete"}:
        raise ValueError("Unknown report completion state")
    if data["state"] != "complete" and not args.allow_unfilled:
        raise ValueError("Incomplete reports require --allow-unfilled and retain visible status")
    if data["state"] == "complete":
        missing = [name for name, value in data["commentary"].items() if not value]
        missing.extend(
            "figure:" + name
            for name, value in data["figures"].items()
            if not all(value.get(lang) for lang in ("zh", "en"))
        )
        missing.extend(
            "coverage:" + row["suite"]
            for row in data["coverage"]
            if any(
                row.get(key) is None for key in ("observed_data", "verified_data", "failed_data")
            )
        )
        if missing:
            raise ValueError("A completed report has unfilled content: " + ", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for language, filename in (("en", "experiment_report.tex"), ("zh", "experiment_report_zh.tex")):
        (args.output_dir / filename).write_text(
            Report(data, language).build(), encoding="utf-8", newline="\n"
        )
    print("Wrote paired TeX sources; PDFs have not been compiled or reviewed.")


if __name__ == "__main__":
    main()
