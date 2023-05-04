import networkx as nx
import pandas as pd
import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import islice
import time


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )


def all_formulations(G, newDemand, n):
    edges_undirected_MT = []

    for u, v in G.edges():
        tail = min(u, v)
        head = max(u, v)
        edges_undirected_MT.append((tail, head, G[u][v]['weight']))

    paths = []
    for i in range(n):
        for j in range(n):
            if i != j:
                newPaths = []
                for path in k_shortest_paths(G, i, j, 5):
                    newPaths.append([(path[k], path[k + 1])
                                    for k in range(len(path) - 1)])
                # newPaths = list(sorted(nx.all_simple_edge_paths(G, i, j), key=len))[0:5]
                paths.extend(newPaths)

    # Create a new model
    m_MT = gp.Model("Max Throughput")
    m_minMLUEObj = gp.Model("Min MLU - flow")
    m_minMLUConstraint = gp.Model("Min MLU with constraint")

    MLU_minMLUEObj = m_minMLUEObj.addVar(name="MLU")
    MLU_minMLUConstraint = m_minMLUConstraint.addVar(name="MLU")

    edges_vars_MT = {}  # maps edge to (edge var, [path vars that use it])
    edges_vars_minMLUEObj = {}
    edges_vars_minMLUConstraint = {}
    for (src, dest, capacity) in edges_undirected_MT:
        tail = min(src, dest)
        head = max(src, dest)

        edges_vars_MT[(tail, head)] = (
            m_MT.addVar(name=f'({tail}, {head})'), [])
        m_MT.addConstr(edges_vars_MT[(tail, head)][0]
                       <= capacity, f'({tail}, {head})<={capacity}')

        edges_vars_minMLUEObj[(tail, head)] = (
            m_minMLUEObj.addVar(name=f'({tail}, {head})'), [])
        m_minMLUEObj.addConstr(edges_vars_minMLUEObj[(
            tail, head)][0] <= capacity, f'({src}, {tail})<={head}')  # >= 0 is already added
        m_minMLUEObj.addConstr(edges_vars_minMLUEObj[(
            tail, head)][0] / capacity <= MLU_minMLUEObj)

        edges_vars_minMLUConstraint[(tail, head)] = (
            m_minMLUConstraint.addVar(name=f'({tail}, {head})'), [])
        m_minMLUConstraint.addConstr(edges_vars_minMLUConstraint[(
            tail, head)][0] <= capacity, f'({src}, {tail})<={head}')  # >= 0 is already added
        m_minMLUConstraint.addConstr(edges_vars_minMLUConstraint[(
            tail, head)][0] / capacity <= MLU_minMLUConstraint)

    demand_vars_MT = {}
    demand_vars_minMLUEObj = {}
    demand_vars_minMLUConstraint = {}

    for i in range(n):
        for j in range(n):
            demand_vars_MT[(i, j)] = (newDemand[i][j], [])
            demand_vars_minMLUEObj[(i, j)] = (newDemand[i][j], [])
            demand_vars_minMLUConstraint[(i, j)] = (newDemand[i][j], [])

    objective_MT = 0
    flow_minMLUEObj = 0
    flow_minMLUConstraint = 0

    for path in paths:
        p_MT = m_MT.addVar(name=f'{path}')
        objective_MT += p_MT

        p_minMLUEObj = m_minMLUEObj.addVar(name=f'{path}')
        flow_minMLUEObj += p_minMLUEObj

        p_minMLUConstraint = m_minMLUConstraint.addVar(name=f'{path}')
        flow_minMLUConstraint += p_minMLUConstraint

        # add paths that try to satisfy a specific demand
        src, dest = path[0][0], path[-1][-1]
        demand_vars_MT[(src, dest)][1].append(p_MT)
        demand_vars_minMLUEObj[(src, dest)][1].append(p_minMLUEObj)
        demand_vars_minMLUConstraint[(src, dest)][1].append(p_minMLUConstraint)

        # this path uses edges --> need to 'credit' the edge
        for (node1, node2) in path:
            tail = min(node1, node2)
            head = max(node1, node2)
            edges_vars_MT[(tail, head)][1].append(p_MT)
            edges_vars_minMLUEObj[(tail, head)][1].append(p_minMLUEObj)
            edges_vars_minMLUConstraint[(tail, head)][1].append(
                p_minMLUConstraint)

    # relating edge to path variables
    for key in edges_vars_MT:
        m_MT.addConstr(edges_vars_MT[key][0] == sum(edges_vars_MT[key][1]))
        m_minMLUEObj.addConstr(edges_vars_minMLUEObj[key][0] == sum(
            edges_vars_minMLUEObj[key][1]))
        m_minMLUConstraint.addConstr(edges_vars_minMLUConstraint[key][0] == sum(
            edges_vars_minMLUConstraint[key][1]))

    # do not send more than what is demanded
    for key in demand_vars_MT:
        if demand_vars_MT[key][1]:
            m_MT.addConstr(
                sum(demand_vars_MT[key][1]) <= demand_vars_MT[key][0])
            m_minMLUEObj.addConstr(
                sum(demand_vars_minMLUEObj[key][1]) <= demand_vars_minMLUEObj[key][0])
            m_minMLUConstraint.addConstr(
                sum(demand_vars_minMLUConstraint[key][1]) >= demand_vars_minMLUConstraint[key][0])

    total_flow_minMLUEObj = m_minMLUEObj.addVar(name="total_flow")
    m_minMLUEObj.addConstr(total_flow_minMLUEObj == flow_minMLUEObj)

    total_flow_minMLUConstraint = m_minMLUConstraint.addVar(name="total_flow")
    m_minMLUConstraint.addConstr(
        total_flow_minMLUConstraint == flow_minMLUConstraint)

    # Set objective
    m_MT.setObjective(objective_MT, GRB.MAXIMIZE)
    # minimize MLU but reward satisfying as much demand as possible
    m_minMLUEObj.setObjective(
        MLU_minMLUEObj - total_flow_minMLUEObj/np.sum(newDemand), GRB.MINIMIZE)
    m_minMLUConstraint.setObjective(MLU_minMLUConstraint, GRB.MINIMIZE)

    # Optimize model
    m_MT.optimize()
    m_minMLUEObj.optimize()
    m_minMLUConstraint.optimize()

    # for v in m.getVars():
    #     print('%s %g' % (v.VarName, v.X))

    return m_MT.Runtime, m_minMLUEObj.Runtime, m_minMLUConstraint.Runtime
