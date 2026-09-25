import re
from argparse import ArgumentParser, FileType
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from itertools import chain, product
from typing import Collection, Dict, List, Optional

import numpy as np
import seaborn as sns
from flitsr.calculations.perms import Calc
from flitsr.merge import Merge  # type:ignore
from flitsr.suspicious import Suspicious
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


def rec_dd(n: Optional[int] = None):
    if (n == 1):
        return dict()
    return defaultdict(partial(rec_dd, None if n is None else (n-1)))


class Type(StrEnum):
    FULL = 'full sampled'
    BASE = ''
    PART = '100 partial sampled'
    STMN = 'steimann'
    BEST = 'best case'
    WRST = 'worst case'
    AVRG = 'average case'


class Result(StrEnum):
    RAW = 'raw'
    ABS_DIFF = 'diff'
    REL_DIFF = 'diff %'


class Measure(StrEnum):
    RESULT = 'result'
    RUNTIME = 'runtime'


@dataclass
class RawData:
    category: str
    calc: Calc
    type: Type
    result: np.ndarray
    runtime: np.ndarray

    def serialize(self):
        # transform to dict format
        dict_data = asdict(self)
        # change np.ndarray to list
        dict_data['result'] = dict_data['result'].tolist()
        dict_data['runtime'] = dict_data['runtime'].tolist()
        return dict_data

    @staticmethod
    def deserialize(data: Dict):
        rd = RawData(**data)
        rd.result = np.asarray(rd.result)
        rd.runtime = np.asarray(rd.runtime)
        return rd


nums = {Calc.WEFFORT: [1, 2, 3, 4, 5], Calc.RECALL: [1, 5, 10, 20]}
types: Dict[Calc, List[Type]] = {Calc.WEFFORT: list(Type), Calc.RECALL:
                                 [t for t in Type if t is not Type.STMN]}
typ_names = {Type.FULL: 'Enum. (10!)', Type.BASE: 'Framework',
             Type.PART: 'Sampled (100)', Type.STMN: 'Högerle [24]',
             Type.BEST: 'Best case', Type.WRST: 'Worst case',
             Type.AVRG: 'Average case'}
type_order = [Type.FULL, Type.BASE, Type.PART, Type.BEST, Type.WRST, Type.AVRG,
              Type.STMN]
calc_names = {Calc.WEFFORT: ('wasted effort', 'effort'),
              Calc.RECALL: ('recall', 'cutoff')}


def name(type_, calc, fault_num=None, runtime: Measure = Measure.RESULT):
    cname = calc_names[calc][0]
    if (type_ is Type.BASE):
        name = f'{cname}'
    else:
        name = f'{type_} {cname}'
    if (runtime is Measure.RUNTIME):
        name += ' runtime'
    if (fault_num is not None):
        if (calc is Calc.RECALL):
            name += f' at {fault_num}'
        elif (calc is Calc.WEFFORT):
            name += f' ({fault_num})'
        else:
            raise ValueError()
    return name


def bland_altman_plot(data1, data2, *args, **kwargs):
    # data1 = np.log1p(np.asarray(data1))
    # data2 = np.log1p(np.asarray(data2))
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    abs_diff = (data1 - data2)
    # Difference between data1 and data2
    diff = np.divide(abs_diff, mean, out=np.zeros_like(abs_diff),
                     where=mean != 0)
    md = np.mean(diff)                   # Mean of the difference
    sd = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(data1, diff, s=4, *args, **kwargs)
    # xs = np.geomspace(1e-10, np.max(mean), 100)
    # plt.plot(xs, xs * 0.05)
    # plt.plot(xs, xs * -0.05)
    plt.axhline(md, linestyle='-', *args, **kwargs)
    plt.axhline(md + 1.96*sd, linestyle='--', *args, **kwargs)
    plt.axhline(md - 1.96*sd, linestyle='--', *args, **kwargs)


RawResults = Dict[str, Dict[Calc, Dict[Type, RawData]]]


def get_raw_results(merge: Merge, metrics: Collection[str], modes: Collection[str],
                    calcs: Collection[Calc]) -> RawResults:
    """
    Return a list of the raw results for the given metric, mode and
    calculation.
    """

    raw_results: RawResults = rec_dd(3)

    # Compress lists of numbers for each stopping point into one (raw) list
    for metric, mode, calc in product(metrics, modes, calcs):
        try:
            avgs = merge.avgs[mode][metric]
        except KeyError:
            continue
        calcs = list(avgs.keys())
        # get stopping criteria nums (based on full enumeration)
        r = re.compile(re.escape(name(Type.FULL, calc, "<rpl>"))
                         .replace("<rpl>", "([0-9]+)"))
        matches = filter(None, map(r.match, calcs))
        nums = sorted([m.group(1) for m in matches])
        for type_ in types[calc]:  # FULL, BASE, PART, (STMN)
            # check if this type_ is available (if not, skip it)
            r = re.compile(re.escape(name(type_, calc, "<rpl>"))
                           .replace("<rpl>", "([0-9]+)"))
            if (not any(r.match(c) for c in calcs)):
                continue
            rs: Dict[str, np.ndarray] = {}
            for measure in Measure:  # RESULT, RUNTIME
                chn = chain(*(avgs[name(type_, calc, i, measure)].all
                              for i in nums))
                rs[measure] = np.asarray(list(chn))
            rd = RawData(metric, calc, type_, **rs)
            raw_results[metric][calc][type_] = rd
    return raw_results


def diff(golden, data):
    mean = np.mean([golden, data], axis=0)
    abs_diff = (data - golden)
    # Difference between data1 and data2
    diff = np.divide(abs_diff, mean, out=abs_diff,
                     where=mean > 1e-1)
    diff *= 100
    return diff


def plot_violins(raw_results: RawResults, categories: Collection[str],
                 calcs: Collection[Calc]):
    sns.set_theme()
    sns.set_style("whitegrid")
    # colors = ['cyan', 'blue', 'red', 'green', 'magenta', 'yellow']
    cs = len(calcs)
    ms = len(categories)
    plt.rc('font', weight='bold')
    fig, axs = plt.subplots(nrows=cs, ncols=ms, figsize=(9, 3), sharey=True,
                            layout='constrained', squeeze=False)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
    for c, calc in enumerate(calcs):
        for m, category in enumerate(categories):
            cax: plt.Axes = axs[c][m]
            cur = raw_results[category][calc]
            golden = cur[Type.FULL].result
            dataset = []
            all_types = sorted(cur.keys(), key=lambda t: type_order.index(t))
            all_types.remove(Type.FULL)
            all_names = [typ_names[t] for t in all_types]
            for type_ in all_types:
                data = cur[type_].result
                dataset.append(diff(golden, data))
            cax.set_yscale('symlog', linthresh=0.2)
            parts = sns.violinplot(data=dataset, ax=cax, cut=0,
                                   density_norm='area', linewidth=1,
                                   inner_kws={'box_width': 6, 'whis_width': 3})
            # sns.boxplot(data=dataset, saturation=0.5, width=0.1, whis=[0,100],
            #             boxprops={'zorder': 2, 'facecolor': 'k'}, ax=cax, showfliers=False,
            #             medianprops={'color': 'w'})

            # , facecolor=list(zip(colors, [0.4]*len(colors))), linecolor='black', widths=0.7, showmeans=True, log_scale=True)
            # for pc in parts['bodies']:
            #     pc.set_edgecolor('black')
            #     pc.set_linewidth(.5)
            cax.set_xticks(np.arange(0, len(all_types)), labels=all_names,
                           rotation=45, ha='right')
            cax.set_title(f"{category} {calc_names[calc][1]}".capitalize(),
                          fontsize=14, fontweight='bold')
            # cax.grid()
            cax.yaxis.set_tick_params(labelleft=True)
            formatter = FuncFormatter(lambda y, _: '{:g}'.format(y))
            cax.yaxis.set_major_formatter(formatter)
            cax.yaxis.set_ticks([-200, -10, -1, -0.1, 0, 0.1, 1, 10, 200])
            cax.yaxis.set_tick_params(which='both', bottom=True)
            # cax.yaxis.set_minor_locator(AutoMinorLocator())
            cax.yaxis.get_minor_locator().set_params(numticks=np.inf, subs=range(1, 10))
    plt.show()


def results_outputs(raw_results: RawResults, categories: List[str],
                    calcs: List[Calc]):
    # code to get the max difference of a particular technique
    # t = raw_results['tarantula']['base'][Calc.RECALL]
    # diffs = diff(t[Type.FULL].result, t[Type.PART].result)
    # zp = zip(diffs, t[Type.PART].result, t[Type.FULL].result,
    #          t[Type.BASE].result)
    # print(sorted(zp, key=lambda x: x[0], reverse=True)[:10])

    # produce table for the run-times
    for calc in calcs:
        printed_header = False
        for category in categories:
            cur = raw_results[category][calc]
            sort_types = sorted(cur.keys(), key=lambda t: type_order.index(t))
            if (not printed_header):
                print('', *[typ_names[t] for t in sort_types], sep=' & ',
                      end='\\\\\n')
                printed_header = True
            print(category.capitalize(), *[f'{np.mean(cur[t].runtime):.4f}'
                  for t in sort_types], sep=' & ', end='\\\\\n')

    # plot the violin plots
    plot_violins(raw_results, categories, calcs)


class TieInfo(StrEnum):
    FPT = 'faults per critical tie',
    LPT = 'fault locs per critical tie',
    TS = 'critical tie size',
    NT = 'number of critical ties'


if __name__ == "__main__":
    # get cmd-line args
    parser = ArgumentParser()
    parser.add_argument('-r', '--recurse', action='store_true',
                        help='By default this script will only look for '
                        'results files in the currently directory. With this '
                        'option, it will recursively look in all subdirectories '
                        'from the current directory. Useful for merging results '
                        'from multiple projects.')
    # parser.add_argument('-o', '--output-file', action='store',
    #                     type=FileType('w'))
    parser.add_argument('-m', '--metrics', action='extend', nargs='+',
                        help='Specify the metrics to merge results for, may '
                        'be specified multiple times.')
    args = parser.parse_args()

    # merge results
    merge = Merge()
    merge.read_results(args.recurse)
    if (hasattr(args, "metrics") and args.metrics is not None
            and len(args.metrics) > 0):
        metrics = [m for m in args.metrics if m in merge.metrics]
    else:
        metrics = sorted(merge.metrics)
    modes = sorted(merge.modes)

    all_calcs: List[Calc] = [Calc.WEFFORT, Calc.RECALL]

    # get all of the raw results
    raw_results: RawResults = get_raw_results(merge, metrics, modes, all_calcs)

    results_outputs(raw_results, metrics, all_calcs)

    # produce table for tie info
    tieinfo_out = {TieInfo.FPT: '$\\obar{f}$', TieInfo.LPT: '$\\obar{l}$',
                   TieInfo.TS: '$\\obar{n}$', TieInfo.NT: '$\\# s$'}

    print('', *tieinfo_out.values(), sep=' & ', end='\\\\\n')
    mode = 'base'
    for metric in metrics:
        avgs = merge.avgs[mode][metric]
        print(metric.capitalize(), *[f'{avgs[ti].eval():.2f}' for ti in TieInfo],
              sep=' & ', end='\\\\\n')
