# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game

pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    r1 = A | B
    r2 = (~A) % ((~B) | C)
    r3 = logic.disjoin(~A, ~B, C)
    return logic.conjoin(r1, r2, r3)


def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')
    r1 = C % (B | D)
    r2 = A >> ((~B) & (~D))
    r3 = (~(B & (~C))) >> A
    r4 = (~D) >> C
    return logic.conjoin(r1, r2, r3, r4)


def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    wupus_a_1 = logic.PropSymbolExpr('WumpusAlive', 1)
    wupus_a_0 = logic.PropSymbolExpr('WumpusAlive', 0)
    wupus_b_0 = logic.PropSymbolExpr('WumpusBorn', 0)
    wupus_k_0 = logic.PropSymbolExpr('WumpusKilled', 0)
    r1 = wupus_a_1 % ((wupus_a_0 & (~wupus_k_0)) | ((~wupus_a_0) & wupus_b_0))
    r2 = ~(wupus_a_0 & wupus_b_0)
    r3 = wupus_b_0
    return logic.conjoin(r1, r2, r3)


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    return logic.pycoSAT(logic.to_cnf(sentence))


def atLeastOne(literals):
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    return logic.disjoin(literals)


def atMostOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    l = [(~literals[i] | ~literals[j]) for i in range(0, len(literals)) for j in range(0, len(literals)) if i != j]
    return logic.conjoin(l)


def exactlyOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in
    CNF (conjunctive normal form)that represents the logic that exactly one of
    the expressions in the list is true.
    """
    if len(literals) > 1:
        return atMostOne(literals) & atLeastOne(literals)
    else:
        return atLeastOne(literals)
        # if len(literals) == 0:
        #     return False
        # if len(literals) == 1:
        #     return literals
        # exprs = []
        # for expr in literals:
        #     tempExprs = list(literals)
        #     tempExprs.remove(expr)
        #     tempExprs.append(logic.Expr('~', expr))
        #     tempExprsExpr = logic.Expr('|', *tempExprs)
        #     exprs.append(tempExprsExpr)
        # exprsExpr = logic.Expr('&', *exprs)
        # return logic.Expr('~', exprsExpr)


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    v = model.values()
    k = model.keys()
    m = [logic.PropSymbolExpr.parseExpr(k[i]) for i in range(0, len(model)) if v[i] == True]
    m = [m[i] for i in range(0, len(m)) if m[i][0] in actions]
    for i in range(0, len(m)):
        m[i] = (m[i][0], int(m[i][1]))

    from operator import itemgetter
    m.sort(key=itemgetter(1))
    return [x[0] for x in m]


def getSolid(x, y, t, walls_grid):
    pre_down = False
    pre_up = False
    pre_left = False
    pre_right = False
    solid = []
    pre_down_loc = logic.PropSymbolExpr(pacman_str, x, y - 1, t - 1)
    pre_north_act = logic.PropSymbolExpr('North', t - 1)

    pre_up_loc = logic.PropSymbolExpr(pacman_str, x, y + 1, t - 1)
    pre_south_act = logic.PropSymbolExpr('South', t - 1)

    pre_left_loc = logic.PropSymbolExpr(pacman_str, x - 1, y, t - 1)
    pre_east_act = logic.PropSymbolExpr('East', t - 1)

    pre_right_loc = logic.PropSymbolExpr(pacman_str, x + 1, y, t - 1)
    pre_west_act = logic.PropSymbolExpr('West', t - 1)

    if 0 <= x < walls_grid.width and 0 <= y - 1 < walls_grid.height:
        pre_down = not walls_grid.data[x][y - 1]
    if not pre_down:
        solid.append(~pre_down_loc)
        solid.append(~pre_north_act)

    if 0 <= x < walls_grid.width and 0 <= y + 1 < walls_grid.height:
        pre_up = not walls_grid.data[x][y + 1]
    if not pre_up:
        solid.append(~pre_up_loc)
        solid.append(~pre_south_act)

    if 0 <= x - 1 < walls_grid.width and 0 <= y < walls_grid.height:
        pre_left = not walls_grid.data[x - 1][y]
    if not pre_left:
        solid.append(~pre_left_loc)
        solid.append(~pre_east_act)

    if 0 <= x + 1 < walls_grid.width and 0 <= y < walls_grid.height:
        pre_right = not walls_grid.data[x + 1][y]
    if not pre_right:
        solid.append(~pre_right_loc)
        solid.append(~pre_west_act)
    if len(solid) > 0:
        return logic.conjoin(solid)
    else:
        return False


def getAction(t):
    return exactlyOne(
        [logic.PropSymbolExpr('North', t), logic.PropSymbolExpr('South', t), logic.PropSymbolExpr('East', t),
         logic.PropSymbolExpr('West', t)])


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    pre_down = False
    pre_up = False
    pre_left = False
    pre_right = False
    literals = []
    if 0 <= x < walls_grid.width and 0 <= y - 1 < walls_grid.height:
        pre_down = not walls_grid.data[x][y - 1]
    if pre_down:
        pre_down_loc = logic.PropSymbolExpr(pacman_str, x, y - 1, t - 1)
        pre_north_act = logic.PropSymbolExpr('North', t - 1)
        literals.append(pre_down_loc & pre_north_act)
    if 0 <= x < walls_grid.width and 0 <= y + 1 < walls_grid.height:
        pre_up = not walls_grid.data[x][y + 1]
    if pre_up:
        pre_up_loc = logic.PropSymbolExpr(pacman_str, x, y + 1, t - 1)
        pre_south_act = logic.PropSymbolExpr('South', t - 1)
        literals.append(pre_up_loc & pre_south_act)
    if 0 <= x - 1 < walls_grid.width and 0 <= y < walls_grid.height:
        pre_left = not walls_grid.data[x - 1][y]
    if pre_left:
        pre_left_loc = logic.PropSymbolExpr(pacman_str, x - 1, y, t - 1)
        pre_east_act = logic.PropSymbolExpr('East', t - 1)
        literals.append(pre_left_loc & pre_east_act)
    if 0 <= x + 1 < walls_grid.width and 0 <= y < walls_grid.height:
        pre_right = not walls_grid.data[x + 1][y]
    if pre_right:
        pre_right_loc = logic.PropSymbolExpr(pacman_str, x + 1, y, t - 1)
        pre_west_act = logic.PropSymbolExpr('West', t - 1)
        literals.append(pre_right_loc & pre_west_act)
    if len(literals) > 0:
        return logic.PropSymbolExpr(pacman_str, x, y, t) % atLeastOne(literals)
    return logic.TRUE  # Replace this with your expression


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    actions = ['North', 'South', 'East', 'West']
    x, y = problem.startState
    t = 0
    final_expression = p_0 = logic.PropSymbolExpr(pacman_str, problem.startState[0], problem.startState[1], 0)
    initial_state = ([], [p_0], x, y, t)
    states = [initial_state]
    visited = []
    while len(states) > 0:
        state = states.pop(0)
        sentence, path, x, y, t = state[0], state[1], state[2], state[3], state[4]
        if not findModel(logic.conjoin(sentence + path)):
            continue
        if state != initial_state and (x, y) == problem.goal and findModel(logic.conjoin(sentence + path)):
            final_expression = logic.conjoin(sentence + path)
            break
        else:
            if (x, y) not in visited and not walls[x][y]:
                visited.append((x, y))
            if (x - 1, y) not in visited and not walls[x - 1][y]:
                tmp = logic.PropSymbolExpr(pacman_str, x - 1, y, t + 1) % (
                    logic.PropSymbolExpr('West', t) & logic.PropSymbolExpr(pacman_str, x, y, t))
                states.append(
                    ([tmp] + sentence, [logic.PropSymbolExpr(pacman_str, x - 1, y, t + 1)] + path, x - 1, y, t + 1))

            if (x + 1, y) not in visited and not walls[x + 1][y]:
                tmp = logic.PropSymbolExpr(pacman_str, x + 1, y, t + 1) % (
                    logic.PropSymbolExpr('East', t) & logic.PropSymbolExpr(pacman_str, x, y, t))
                states.append(
                    ([tmp] + sentence, [logic.PropSymbolExpr(pacman_str, x + 1, y, t + 1)] + path, x + 1, y, t + 1))
            if (x, y - 1) not in visited and not walls[x][y - 1]:
                tmp = logic.PropSymbolExpr(pacman_str, x, y - 1, t + 1) % (
                    logic.PropSymbolExpr('South', t) & logic.PropSymbolExpr(pacman_str, x, y, t))
                states.append(
                    ([tmp] + sentence, [logic.PropSymbolExpr(pacman_str, x, y - 1, t + 1)] + path, x, y - 1, t + 1))

            if (x, y + 1) not in visited and not walls[x][y + 1]:
                tmp = logic.PropSymbolExpr(pacman_str, x, y + 1, t + 1) % (
                    logic.PropSymbolExpr('North', t) & logic.PropSymbolExpr(pacman_str, x, y, t))
                states.append(
                    ([tmp] + sentence, [logic.PropSymbolExpr(pacman_str, x, y + 1, t + 1)] + path, x, y + 1, t + 1))

    return extractActionSequence(findModel(final_expression), actions)


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    actions = ['North', 'South', 'East', 'West']
    food = problem.startingGameState.data.food
    t = 0
    KB = []
    TB = [logic.PropSymbolExpr(pacman_str, problem.start[0][0], problem.start[0][1], 0)]

    for i in range(0, food.width):
        for j in range(0, food.height):
            if food[i][j]:
                TB.append(logic.PropSymbolExpr('F', i, j, t))
            else:
                TB.append(~logic.PropSymbolExpr('F', i, j, t))
    for i in range(0, walls.width):
        for j in range(0,walls.height):
            if not walls[i][j] and (i,j)!=(problem.start[0][0], problem.start[0][1]):
                TB.append(~logic.PropSymbolExpr('P', i, j, t))

    for time in range(0, 3000):
        TB_TMP = []
        for t in range(0, time):
            TB_TMP.append(getAction(t))
        food_eated = [~logic.PropSymbolExpr("F", x, y, time) for x,y in food.asList() if not walls[x][y]]
        for x, y in food.asList():
            for t in range(1, time + 1):
                pastFood = logic.PropSymbolExpr("F", x, y, t - 1)
                currentFood = logic.PropSymbolExpr("F", x, y, t)
                pacmanHere = logic.PropSymbolExpr("P", x, y, t)
                isNotEaten = (~pastFood | (pastFood & pacmanHere))
                TB_TMP.append(~currentFood % isNotEaten)
        for t in range(1, time + 1):
            for x in range(1, problem.getWidth() + 1):
                for y in range(1, problem.getHeight() + 1):
                    TB_TMP.append(pacmanSuccessorStateAxioms(x,y,t,walls))
        result = findModel(logic.conjoin(TB+TB_TMP+food_eated))
        if result:
            A = extractActionSequence(result, ["North", "South", "East", "West"])
            if A!=[]:
                return A


def heuristic(food_state, walls, foods):
    def manhattanDistance(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2)

    corners = foods  # These are the corner coordinates
    pos = (food_state[0], food_state[1])
    corners_visited = food_state[2:len(food_state)]
    unvisited_corners = [corners[i] for i in range(0, len(corners_visited)) if not corners_visited[i]]
    distances = [manhattanDistance(unvisited_corners[i], pos) for i in range(0, len(unvisited_corners))]
    h = 0
    if len(distances) > 0:
        h = max(distances)
    return h  # Default to trivial solution


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
