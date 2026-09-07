from datetime import timedelta
from argparse import ArgumentParser, ArgumentTypeError
from os import path as osp
import time
from typing import Dict, List, Tuple, Callable
from itertools import chain
from collections import Counter
import numpy as np
from flitsr.calculations.perms import exact_method, Calc
from flitsr.calculations import BUModel
from flitsr.spectrum import Spectrum
from flitsr.tie import Tie, _CollapsableFault
from flitsr_calculations.merge_custom import Type, types, RawResults, \
        RawData, rec_dd, plot_violins, calc_names, type_order, typ_names
from experiment_helper import ExpConfig, Exp, read_exp_file, intRange
# from matplotlib import pyplot as plt


def compute(config: ExpConfig, exp: Exp, calc: Calc, bu: BUModel,
            func: Callable[[ExpConfig, Exp, Calc, BUModel], float]) \
                    -> Tuple[float, timedelta]:
    start_time = time.time()
    result = func(config, exp, calc, bu)
    end_time = time.time()
    time_diff = timedelta(seconds=(end_time - start_time))
    return result, time_diff


def time_diff(t1: timedelta, t2: timedelta, perc=True) -> float:
    if (not perc):
        return (t1 - t2).total_seconds()
    if (t1 > t2):
        return min(100, ((t1 - t2)/t2)*100)
    else:
        return (- (t2 - t1)/t2)*100


def val_diff(v1, v2, perc=True):
    if (not perc):
        return abs(v1 - v2)
    if (v1 == 0 and v2 == 0):
        return 0.0
    elif (v1 == 0 or v2 == 0):
        return 100.0
    elif (v1 < 1e-3 and v2 < 1e-3):
        return 0.0
    else:
        return abs(v1 - v2)*100/v2


def _best_case_w(config: ExpConfig, exp: Exp) -> float:
    return 0.0


def _worst_case_w(config: ExpConfig, exp: Exp) -> float:
    """worst case -> look at all non-faulty elements first"""
    num_fault_locs = len(exp.faults)
    return config.m - num_fault_locs


def _avg_case_w(config: ExpConfig, exp: Exp) -> float:
    """average case -> look at half the non-faulty elements first"""
    return (_worst_case_w(config, exp))/2


def _best_worst_faults(faults, k: int, bu: BUModel, rev=True) -> float:
    """
    Compute how many faults are found when inspecting only faulty locations.
    When `rev` is true, inspect elements with the most faults first,
    otherwise inspect elements with the most faults last.
    """
    # inspect the elements with the most faults first/last
    flts_by_loc = [fs for fs in faults.values()]
    flts_by_loc = sorted(flts_by_loc, key=lambda x: len(x), reverse=rev)
    # count how many times each fault is found in the first k elements
    counts = Counter(chain.from_iterable(flts_by_loc[:k]))
    # compare to num locs needed to identify each fault
    expected = bu.get_dict(faults, by_loc=True)
    num_found = 0
    for fault in counts.keys():
        if (counts[fault] >= expected[fault]):
            num_found += 1
    return num_found


def _best_case_r(config: ExpConfig, exp: Exp, bu: BUModel) -> float:
    """ Recall at the k-th element in the tie using best case strategy. """
    return _best_worst_faults(exp.faults, config.q, bu, rev=True)


def _worst_case_r(config: ExpConfig, exp: Exp, bu: BUModel) -> float:
    num_non_faults = config.m - len(exp.faults)
    # inspect non-faulty elements first
    k = max(config.q - num_non_faults, 0)
    # short-cut if you've already exhausted your budget
    if (k <= 0):
        return 0.0
    # use the rest of the budget (k) to inspect faulty locations
    return _best_worst_faults(exp.faults, k, bu, rev=False)


def _avg_case_r(config: ExpConfig, exp: Exp, bu: BUModel) -> float:
    return (_best_case_r(config, exp, bu) + _worst_case_r(config, exp, bu))/2


def best(config: ExpConfig, exp: Exp, calc: Calc, bu: BUModel):
    if (calc in [Calc.WEFFORT, Calc.EXAM]):
        return _best_case_w(config, exp)
    else:
        return _best_case_r(config, exp, bu)


def worst(config: ExpConfig, exp: Exp, calc: Calc, bu: BUModel):
    if (calc in [Calc.WEFFORT, Calc.EXAM]):
        return _worst_case_w(config, exp)
    else:
        return _worst_case_r(config, exp, bu)


def avrg(config: ExpConfig, exp: Exp, calc: Calc, bu: BUModel):
    if (calc in [Calc.WEFFORT, Calc.EXAM]):
        return _avg_case_w(config, exp)
    else:
        return _avg_case_r(config, exp, bu)


def stmn(config: ExpConfig, exp: Exp, calc: Calc, bu: BUModel):
    n = config.m
    m = config.f
    k = config.q
    return (k * (n-m))/(m+1)


def sampled(config: ExpConfig, exp: Exp, calc: Calc, bu_model: BUModel,
            samples=100):
    elems = set(range(1, config.m+1))
    return exact_method(exp.faults, config.q, elems=elems, calc=calc,
                        bu=bu_model, samples=samples)


type_funcs: Dict[Type, Callable[[ExpConfig, Exp, Calc, BUModel], float]] = {
        Type.STMN: stmn, Type.PART: sampled, Type.BEST: best, Type.WRST: worst,
        Type.AVRG: avrg}


def incl_config(config: ExpConfig, restrictions: Dict[str, List[int]]) -> bool:
    for r, rng in restrictions.items():
        if (getattr(config, r) not in rng):
            return False
    return True


def get_raw_results(input_dir: str, restrictions: Dict[str, List[int]],
                    calcs: List[Calc] = None) -> RawResults:
    if (calcs is None):
        calcs = [Calc.WEFFORT, Calc.RECALL]
    raw_results: RawResults = rec_dd()
    for bu_model in BUModel.get_types():
        for calc in calcs:
            fname = f"exp_{calc_names[calc][1]}_{str(bu_model).lower()}.txt"
            path = osp.join(input_dir, fname)
            raw_results[bu_model][calc] = \
                get_category_results(path, bu_model, calc, restrictions)
    return raw_results


def get_category_results(input_: str, bu_model: BUModel, calc: Calc,
                         restrictions: Dict[str, List[int]]) \
                                 -> Dict[Type, RawData]:
    exps_orig, _ = read_exp_file(input_)
    ret = {}
    results: Dict[Type, List[float]] = {}
    runtimes: Dict[Type, List[float]] = {}
    for config in exps_orig:
        if (not incl_config(config, restrictions)):
            continue
        for exp in exps_orig[config]:
            for type_ in types[calc]:
                if (type_ is Type.FULL):
                    result, time = exp.expect_val, exp.expect_time
                elif (type_ is Type.BASE):
                    result, time = exp.formula_val, exp.formula_time
                else:
                    result, time = compute(config, exp, calc, bu_model,
                                           type_funcs[type_])
                results.setdefault(type_, []).append(result)
                runtimes.setdefault(type_, []).append(time.total_seconds())
    for type_ in types[calc]:
        ret[type_] = RawData(str(bu_model), calc, type_,
                             np.asarray(results[type_]),
                             np.asarray(runtimes[type_]))
    return ret


def type_dir(input_: str) -> str:
    if (osp.isdir(input_)):
        return input_
    else:
        raise ArgumentTypeError(f"can't open {input_}: No such file "
                                "or directory.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=type_dir)
    parser.add_argument('-m', action='store', default=None, type=intRange,
                        help='The number of elements in the tie')
    parser.add_argument('-f', action='store', default=None, type=intRange,
                        help='The number of faults in the tie')
    parser.add_argument('-l', '--l-max', action='store', default=None,
                        type=intRange, help='The maximum number of locations '
                        'per fault in the tie', dest='l')
    parser.add_argument('-q', action='store', default=None, type=intRange,
                        help='The inspection cut-off point.')
    parser.add_argument('-o', '--overlap', action='store', type=intRange,
                        help='The total number of overlapping '
                        'elements allowed.', dest='o')
    parser.add_argument('-c', '--calculation', choices=[Calc.WEFFORT, Calc.RECALL],
                        type=Calc.from_string, default=None,
                        help='Only include results from the given calculation')
    # calc_choices = [Calc.WEFFORT, Calc.PRECISION]
    # parser.add_argument('-c', '--calculation', choices=calc_choices,
    #                     type=Calc.from_string, default=Calc.WEFFORT,
    #                     help='Specifies the calculation to perform (default '
    #                     'Wasted Effort)', required=True)
    # parser.add_argument('-x', '--bug-understanding-model',
    #                     choices=BUModel.get_types(), type=BUModel.from_string,
    #                     help='The bug understanding model to use. Note: the '
    #                     'default imperfect strategy is l/2.',
    #                     default=BUModel.PERFECT, required=True)
    # parser.add_argument('-o', '--output-file', type=argparse.FileType('r'),
    #                     help='Specify the output file to print the results to')
    args = parser.parse_args()
    restrictions = {}
    for restr in ['m', 'f', 'l', 'q', 'o']:
        if (hasattr(args, restr) and getattr(args, restr) is not None):
            restrictions[restr] = getattr(args, restr)
    if (args.calculation is None):
        calcs = [Calc.WEFFORT, Calc.RECALL]
    else:
        calcs = [args.calculation]
    print(f"Restrictions: {restrictions}")
    raw_results = get_raw_results(args.input_dir, restrictions,
                                  calcs=calcs)

    # code to get the max difference of a particular technique
    # t = raw_results[BUModel.INEPT][Calc.RECALL]
    # diffs = diff(t[Type.FULL].result, t[Type.BASE].result)
    # zp = zip(diffs, t[Type.BASE].result, t[Type.FULL].result)
    # print(sorted(zp, key=lambda x: x[0], reverse=True)[:10])

    # produce table for the run-times
    for calc in [Calc.WEFFORT, Calc.RECALL]:
        printed_header = False
        for bu_model in BUModel.get_types():
            cur = raw_results[bu_model][calc]
            sort_types = sorted(cur.keys(), key=lambda t: type_order.index(t))
            if (not printed_header):
                print('', *[typ_names[t] for t in sort_types], sep=' & ',
                      end='\\\\\n')
                printed_header = True
            print(str(bu_model).capitalize(), *[f'{np.mean(cur[t].runtime):.4f}'
                  for t in sort_types], sep=' & ', end='\\\\\n')

    # plot the violin plots
    plot_violins(raw_results, BUModel.get_types(), calcs)
