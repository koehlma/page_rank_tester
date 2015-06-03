import re
import sys
try:
    import numpy
    import scipy.stats
except:
    sys.exit('Cannot import scipy/numpy.\n'
             'Please install it using:\n'
             'sudo zypper in python-scipy')
import subprocess
import threading
import traceback


class TestFailure(Exception):
    def __init__(self, msg):
        self.msg = msg


def run(sut, args, cwd, timeout=1, verbose=False, debug=False, capt_out=True):
    call_args = [sut] + args

    out = subprocess.PIPE if capt_out else subprocess.DEVNULL

    proc = subprocess.Popen(call_args, stdout=out,
                            stderr=subprocess.STDOUT, cwd=cwd,
                            universal_newlines=True)
    res = {}

    def reader():
        try:
            res['out'], _ = proc.communicate()
            res['status'] = proc.returncode
        except:
            raise TestFailure(traceback.format_exc())

    thread = threading.Thread(target=reader)
    thread.start()
    thread.join(timeout)
    if proc.poll() is None:
        proc.terminate()
        thread.join(2)
        if proc.poll() is None:
            proc.kill()
            thread.join()
        raise TestFailure('Timeout ({} seconds) expired!'.format(timeout))

    return proc, res['out']


def parse_ranks(out, print_all=False):
    res = {}
    for l in out.splitlines():
        if print_all:
            print(l)

        wds = l.split(None, 3)

        if len(wds) != 2:
            raise TestFailure('Unexpected line of illegal format in output: {}'
                              .format(l))

        try:
            score = float(wds[1])

            if wds[0] in res:
                raise TestFailure('Unexpectedly seeing score for node {} again'
                                  .format(wds[0]))

            res[wds[0]] = score
        except ValueError:
            raise TestFailure('Expecting float, got something else: {}'.format(
                wds[1]))

    return res


def expect_scores(proc, out, scores, delta=0.0, verbose=False, debug=False):
    expect_retcode(proc, 0, out, verbose, debug)

    res = parse_ranks(out)

    if len(res) != len(scores):
        raise TestFailure('Expecting to get scores for {} nodes, got {}'
                          .format(len(scores), len(res)))

    for n in scores:
        score = scores[n]
        if n not in res:
            raise TestFailure('No score for expected node: {}'.format(n))
        if abs(res[n]-score) > delta:
            raise TestFailure('Mismatch of score for node {}: '
                              'Expecting {}, got {} (allowed deviation: {})'.
                              format(n, score, res[n], delta))


def expect_scoresum1(proc, out, delta=1e-9, verbose=False, debug=False):
    expect_retcode(proc, 0, out, verbose, debug)

    score_sum = 0
    for l in out.splitlines():
        wds = l.split(None, 3)

        if len(wds) != 2:
            raise TestFailure('Unexpected line of illegal format in output: {}'
                              .format(l))

        try:
            score = float(wds[1])
            score_sum += score
        except ValueError:
            raise TestFailure('Expecting float, got something else: {}'.format(
                wds[1]))

    if abs(1.0-score_sum) > delta:
        raise TestFailure('Mismatch of score sum: Expecting 1.0 (+/- {}), '
                          'got {} '.format(delta, score_sum))


def expect_retcode(proc, retcode, out=None, verbose=False, debug=False):
    if retcode != -1 and proc.returncode != retcode:
        if verbose and out is not None:
            print('Program output:\n{}'.format(out))
        raise TestFailure('Wrong return code: expected {}, got {}'
                          .format(retcode, proc.returncode))

    if retcode == -1 and proc.returncode == 0:
        if verbose and out is not None:
            print('Program output:\n{}'.format(out))
        raise TestFailure('Expected a non-zero return code, got {}'
                          .format(proc.returncode))


def expect_stats(proc, out, name, nodes, edges, min_in, max_in, min_out,
                 max_out, verbose=False, debug=False):
    expect_retcode(proc, 0, out, verbose, debug)

    lines = out.splitlines()

    if len(lines) != 5:
        raise TestFailure('Unexpected number of lines in statistics output: '
                          'expecting 5, got {}'.format(len(lines)))

    m = re.match('(.*):', lines[0])
    if m is None:
        raise TestFailure('Unexpected line \'{}\' while expecting the '
                          'identifier line'.format(lines[0]))

    c_name = m.group(1)
    if c_name != name:
        raise TestFailure('Name in stats not matching: expecting \'{}\', '
                          'got \'{}\''.format(name, c_name))

    m = re.match('- num nodes: (\d+)', lines[1])
    if m is None:
        raise TestFailure('Unexpected line \'{}\' while expecting num nodes '
                          'line'.format(lines[1]))
    c_nnodes = int(m.group(1))
    if c_nnodes != nodes:
        raise TestFailure('Unexpected number of nodes in stats: '
                          'expected {}, got {}'.format(nodes, c_nnodes))

    m = re.match('- num edges: (\d+)', lines[2])
    if m is None:
        raise TestFailure('Unexpected line \'{}\' while expecting num edges '
                          'line'.format(lines[2]))
    c_nedges = int(m.group(1))
    if c_nedges != edges:
        raise TestFailure('Unexpected number of edges in stats: expected {}, '
                          'got {}'.format(edges, c_nedges))

    m = re.match('- indegree: (\d+)-(\d+)', lines[3])
    if m is None:
        raise TestFailure('Unexpected line \'{}\' while expecting indegree line'
                          .format(lines[3]))
    c_min_in = int(m.group(1))
    if c_min_in != min_in:
        raise TestFailure('Unexpected minimum in-degree in stats: expected {}, '
                          'got {}'.format(min_in, c_min_in))
    c_max_in = int(m.group(2))
    if c_max_in != max_in:
        raise TestFailure('Unexpected maximum in-degree in stats: expected {}, '
                          'got {}'.format(max_in, c_max_in))

    m = re.match('- outdegree: (\d+)-(\d+)', lines[4])
    if m is None:
        raise TestFailure('Unexpected line \'{}\' while expecting outdegree '
                          'line'.format(lines[4]))
    c_min_out = int(m.group(1))
    if c_min_out != min_out:
        raise TestFailure('Unexpected minimum out-degree in stats: '
                          'expected {}, got {}'.format(min_out, c_min_out))
    c_max_out = int(m.group(2))
    if c_max_out != max_out:
        raise TestFailure('Unexpected maximum out-degree in stats: '
                          'expected {}, got {}'.format(max_out, c_max_out))


def test_distribution(sut, args, cwd, ref_measures,
                      timeout=2, p_min=0.01, verbose=False, debug=False):
    ranks = {i: [] for i in ref_measures}

    min_num_runs = min([len(m) for m in ref_measures.values()])
    max_num_runs = max([len(m) for m in ref_measures.values()])
    if min_num_runs != max_num_runs:
        raise AssertionError('number of reference values must be the same for '
                             'all nodes ({} != {})'.format(
                                 min_num_runs, max_num_runs))
    num_runs = min_num_runs

    if debug:
        print('Executing {} runs...'.format(num_runs))

    for nr in range(num_runs):
        try:
            proc, out = run(sut, args, cwd, 10, verbose, debug)
        except TestFailure as e:
            raise TestFailure('Error in run {}: {}'.format(nr, e.msg))

        try:
            res = parse_ranks(out)
        except TestFailure as e:
            raise TestFailure('Run {} produced illegal output: {}'.format(
                nr, e.msg))

        if len(res) != len(ranks):
            raise TestFailure('In run {}: Expecting to get scores for {} '
                              'nodes, got {}'.format(nr, len(ranks), len(res)))

        for node, rank in res.items():
            if node not in ref_measures:
                raise TestFailure('In run {}: Unknown node: {}'
                                  .format(node))
            ranks[node].append(rank)

    error = None
    for node in sorted(ref_measures):
        ref = ref_measures[node]
        stud = ranks[node]
        ref_mean = numpy.mean(ref)
        ref_var = numpy.var(ref)
        stud_mean = numpy.mean(stud)
        stud_var = numpy.var(stud)

        p = scipy.stats.ks_2samp(ref, stud)[1]
        if debug:
            print('Node {}:\n  Ref:  {} +- {}\n  Stud: {} +- {}\n' \
                  '  p:    {} (must be >= {})' \
                  .format(node, ref_mean, ref_var, stud_mean, stud_var, p,
                          p_min))
        if p < p_min:
            msg = ('Statistical distribution does not match for node {}.\n'
                '  Expected mean {}, variance {}; got mean {}, variance {}.\n'
                '  K-S test sais p={}, which is less than the required {}.\n'
                   ).format(node, ref_mean, ref_var, stud_mean, stud_var, p,
                        p_min)
            if error is None:
                error = msg
            if debug:
                print(msg)

    if error is not None:
        raise TestFailure(error)
