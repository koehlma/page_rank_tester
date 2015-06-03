# -*- coding: utf-8 -*-
#
# Copyright (C) 2015, Maximilian Köhl <mail@koehlma.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import collections
import os.path
import random
import re
import tempfile

import numpy

import utils

graph_name_regex = re.compile(r'digraph\s*(\w+?)\s*\{')
graph_edge_regex = re.compile(r'(\w+?)\s*->\s*(\w+?)\s*;')

Statistic = collections.namedtuple('Statistic', ['num_nodes', 'num_edges',
                                                 'min_in', 'max_in',
                                                 'min_out', 'max_out'])


class InvalidGraphFile(Exception):
    pass


class Node:
    def __init__(self, name):
        self.name = name
        self.successors = []
        self.predecessors = []

    def __repr__(self):
        return '<Node {}>'.format(self.name)


class Graph:
    def __init__(self, name, nodes):
        self.name = name
        self.nodes = nodes

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as input_file:
            content = input_file.read().decode('utf-8')
        graph_name_match = graph_name_regex.search(content)
        if not graph_name_match:
            raise InvalidGraphFile('no graph name found')
        graph_name = graph_name_match.group(1)
        nodes = {}
        for edge in graph_edge_regex.finditer(content):
            from_name, to_name = edge.group(1), edge.group(2)
            if from_name not in nodes:
                nodes[from_name] = Node(from_name)
            if to_name not in nodes:
                nodes[to_name] = Node(to_name)
            nodes[from_name].successors.append(nodes[to_name])
            nodes[to_name].predecessors.append(nodes[from_name])
        return Graph(graph_name, sorted(nodes.values(), key=lambda n: n.name))

    @classmethod
    def generate(cls, nodes=10, edges=50, name_str=lambda i: 'N' + str(i)):
        nodes = [Node(name_str(index)) for index in range(nodes)]
        for edge in range(edges):
            from_node = random.choice(nodes)
            to_node = random.choice(nodes)
            from_node.successors.append(to_node)
            to_node.predecessors.append(from_node)
        return cls('Graph', [node for node in nodes
                             if node.successors or node.predecessors])

    @property
    def edges(self):
        return sum(len(node.successors) for node in self.nodes)

    @property
    def graphviz(self):
        yield 'digraph {} {{'.format(self.name)
        for from_node in self.nodes:
            for to_node in from_node.successors:
                yield '    {} -> {};'.format(from_node.name, to_node.name)
        yield '}'

    def statistic(self):
        min_in_degree, max_in_degree = self.edges, 0
        min_out_degree, max_out_degree = self.edges, 0
        for node in self.nodes:
            if len(node.predecessors) < min_in_degree:
                min_in_degree = len(node.predecessors)
            if len(node.predecessors) > max_in_degree:
                max_in_degree = len(node.predecessors)
            if len(node.successors) < min_out_degree:
                min_out_degree = len(node.successors)
            if len(node.successors) > max_out_degree:
                max_out_degree = len(node.successors)
        return Statistic(len(self.nodes), self.edges, min_in_degree,
                         max_in_degree, min_out_degree, max_out_degree)

    def random_page_rank(self, steps, probability=10):
        current_node = random.choice(self.nodes)
        counter = {node: 0 for node in self.nodes}
        for step in range(steps):
            rand_int = random.randint(0, 99)
            if rand_int < probability or not current_node.successors:
                current_node = random.choice(self.nodes)
            else:
                current_node = random.choice(current_node.successors)
            counter[current_node] += 1
        result = collections.OrderedDict()
        for index, node in enumerate(self.nodes):
            result[node] = counter[node] / steps
        return result

    def markov_page_rank(self, steps, probability=10):
        vector = numpy.array([1 / len(self.nodes)] * len(self.nodes),
                             dtype=numpy.float64)
        matrix = numpy.array([[0] * len(self.nodes)] * len(self.nodes),
                              dtype=numpy.float64)
        jump_fraction = probability / 100
        for from_index, from_node in enumerate(self.nodes):
            for to_index, to_node in enumerate(self.nodes):
                if from_node.successors:
                    out_sum = ((1 - jump_fraction) *
                               len([s for s in from_node.successors
                                    if s == to_node]) /
                               len(from_node.successors))
                    matrix[from_index][to_index] = (jump_fraction /
                                                    len(self.nodes) + out_sum)
                else:
                    matrix[from_index][to_index] = 1 / len(self.nodes)
        for step in range(steps):
            vector = numpy.dot(vector, matrix)
        result = collections.OrderedDict()
        for index, node in enumerate(self.nodes):
            result[node] = float(vector[index])
        return result


def action_run(arguments):
    graph = Graph.from_file(arguments.filename)
    if arguments.s:
        stats = graph.statistic()
        print('{}:'.format(graph.name))
        print('- num nodes: {}'.format(stats.num_nodes))
        print('- num edges: {}'.format(stats.num_edges))
        print('- indegree: {}-{}'.format(stats.min_in, stats.max_in))
        print('- outdegree: {}-{}'.format(stats.min_out, stats.max_out))
    if arguments.r is not None:
        result = graph.random_page_rank(arguments.r, arguments.p)
        for node, value in result.items():
            print('{} {:.10f}'.format(node.name, value))
    if arguments.m is not None:
        result = graph.markov_page_rank(arguments.m, arguments.p)
        for node, value in result.items():
            print('{} {:.10f}'.format(node.name, value))


def action_generate(arguments):
    pass


def action_test(arguments):
    graph = Graph.generate()
    statistic = graph.statistic()
    scores_reference = {n.name: round(v, 10) for n, v in
                        graph.markov_page_rank(arguments.steps,
                                               arguments.probability).items()}
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, 'graph.dot')
        with open(filename, 'wb') as graph_file:
            graph_file.write(('\n'.join(graph.graphviz)).encode('utf-8'))
        process_stats, output_stats = utils.run(arguments.binary,
                                                ['-s', filename],
                                                os.getcwd())
        args = ['-r', str(arguments.steps), '-p', str(arguments.probability),
                filename]
        process_random, output_random = utils.run(arguments.binary,
                                                  args, os.getcwd())
        args = ['-m', str(arguments.steps), '-p', str(arguments.probability),
                filename]
        process_markov, output_markov = utils.run(arguments.binary,
                                                  args, os.getcwd())

    try:
        utils.expect_stats(process_stats, output_stats, graph.name,
                           statistic.num_nodes, statistic.num_edges,
                           statistic.min_in, statistic.max_in,
                           statistic.min_out, statistic.max_out)
        utils.expect_scores(process_random, output_random, scores_reference,
                            arguments.delta)
        utils.expect_scores(process_markov, output_markov, scores_reference)
    except utils.TestFailure as error:
        print(error)

def action_dist(arguments):
    graph = Graph.generate()
    reference_values = {node.name: [] for node in graph.nodes}
    for i in range(arguments.reference_values):
        for node, value in graph.random_page_rank(1000).items():
            reference_values[node.name].append(value)
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, 'graph.dot')
        with open(filename, 'wb') as graph_file:
            graph_file.write(('\n'.join(graph.graphviz)).encode('utf-8'))
        args = ['-r', '1000', '-p', '10', filename]
        try:
            utils.test_distribution(arguments.binary, args, os.getcwd(),
                                    reference_values)
        except utils.TestFailure as error:
            print(error)


actions = {'run': action_run, 'generate': action_generate,
           'test': action_test, 'dist': action_dist}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action')

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('-s', '--statistic', action='store_true')
    run_parser.add_argument('-p', '--probability', type=int, default=10)
    run_parser.add_argument('-r', '--random-walk', type=int)
    run_parser.add_argument('-m', '--markov-chain', type=int)
    run_parser.add_argument('filename')

    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('output', type=argparse.FileType('wb'))

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('-d', '--delta', type=float, default=0.05)
    test_parser.add_argument('-s', '--steps', type=int, default=1000)
    test_parser.add_argument('-p', '--probability', type=int, default=10)
    test_parser.add_argument('binary')

    dist_parser = subparsers.add_parser('dist')
    dist_parser.add_argument('--reference-values', type=int, default=1000)
    dist_parser.add_argument('binary')

    arguments = parser.parse_args()

    if not arguments.action:
        parser.print_help()
        return 1

    actions[arguments.action](arguments)


if __name__ == '__main__':
    import sys
    sys.exit(main())
