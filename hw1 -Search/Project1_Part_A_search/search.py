# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    state = problem.getStartState()
    visited = []
    parent = {}
    parent[state] = None
    stack = util.Stack()
    stack.push(state)
    while not stack.isEmpty():
        state = stack.pop()
        if problem.isGoalState(state):
            break
        else:
            if not (state in visited):
                visited.append(state)
                for c in problem.getSuccessors(state):
                    if c[0] not in visited:
                        stack.push(c[0])
                        parent[c[0]] = (state, c[1])
    tmp = state
    action = []
    while True:
        from types import NoneType
        if isinstance(parent[tmp], NoneType):
            break
        action.append(parent[tmp][1])
        tmp = parent[tmp][0]
    action.reverse()
    return action




def breadthFirstSearch(problem):
    # state = problem.getStartState()
    # visited = []
    # parent = {}
    # parent[state] = None
    # queue = util.Queue()
    # queue.push(state)
    # while not queue.isEmpty():
    #     state = queue.pop()
    #     if problem.isGoalState(state):
    #         break
    #     else:
    #         if state not in visited:
    #             visited.append(state)
    #             for c in problem.getSuccessors(state):
    #                 if parent.has_key(c[0])== False and (c[0] not in visited):
    #                     parent[c[0]] =(state,c[1])
    #                     queue.push(c[0])
    #
    # tmp = state
    # action = []
    # while True:
    #     from types import NoneType
    #     if isinstance(parent[tmp],NoneType):
    #         break
    #     action.append(parent[tmp][1])
    #     tmp = parent[tmp][0]
    # action.reverse()
    # return action
    state = (problem.getStartState(), [])
    visited = []
    queue = util.Queue()
    queue.push(state)
    while not queue.isEmpty():
        state = queue.pop()
        if problem.isGoalState(state[0]):
            return state[1]
        else:
            if state[0] not in visited:
                visited.append(state[0])
                for c in problem.getSuccessors(state[0]):
                    if c[0] not in visited:
                        queue.push((c[0], state[1] + [c[1]]))
    return []


def uniformCostSearch(problem):
    state = (problem.getStartState(),[],0)
    visited = []
    queue = util.PriorityQueue()
    queue.push(state,0)
    while not queue.isEmpty():
        state = queue.pop()
        if problem.isGoalState(state[0]):
            return state[1]
        else:
            if state[0] not in visited:
                visited.append(state[0])
                for c in problem.getSuccessors(state[0]):
                    if c[0] not in visited:
                        queue.push((c[0],state[1]+[c[1]],state[2]+c[2]),state[2]+c[2])
    return  []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    state = (problem.getStartState(),[],0)
    visited = []
    queue = util.PriorityQueue()
    queue.push(state,0)
    while not queue.isEmpty():
        state = queue.pop()
        if problem.isGoalState(state[0]):
            return state[1]
        else:
            if state[0] not in visited:
                visited.append(state[0])
                for c in problem.getSuccessors(state[0]):
                    if c[0] not in visited:
                        heuristic_value = heuristic(c[0], problem)
                        queue.push((c[0],state[1]+[c[1]],state[2]+c[2]),state[2]+c[2]+heuristic_value)
    return  []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
