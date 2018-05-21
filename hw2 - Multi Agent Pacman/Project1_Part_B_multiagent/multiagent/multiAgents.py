# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        def isTerminal(state,depth):
           return state.isWin() or state.isLose() or depth==0
        def Value(state,agentIndex,depth):
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            if agentIndex==0:
                return maxValue(state,agentIndex,depth)
            else:
                return minValue(state,agentIndex,depth)

        def maxValue(state,agentIndex,depth):
            v = -float('inf')
            actions = state.getLegalActions(agentIndex)
            for a in actions:
                v = max(v,Value(state.generateSuccessor(agentIndex, a),agentIndex+1,depth))
            return v

        def minValue(state,agentIndex,depth):
            v = float('inf')
            actions = state.getLegalActions(agentIndex)
            for a in actions:
                if agentIndex <state.getNumAgents()-1:
                    v = min(v, Value(state.generateSuccessor(agentIndex, a), agentIndex + 1, depth))
                else:
                    v = min(v, Value(state.generateSuccessor(agentIndex, a), 0, depth-1))
            return v


        depth = self.depth

        value = -float('inf')
        action = Directions.STOP
        all_actions = gameState.getLegalActions(0)

        for a in all_actions:
            newState = gameState.generateSuccessor(0,a)
            v =Value(newState,1,depth)
            if v>value:
                value = v
                action = a
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        def isTerminal(state,depth):
           return state.isWin() or state.isLose() or depth==0
        def Value(state,agentIndex,depth,alpha,beta):
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            if agentIndex==0:
                return maxValue(state,agentIndex,depth,alpha,beta)
            else:
                return minValue(state,agentIndex,depth,alpha,beta)

        def maxValue(state,agentIndex,depth,alpha,beta):
            v = -float('inf')
            actions = state.getLegalActions(agentIndex)
            best_action = Directions.STOP
            for a in actions:
                val = Value(state.generateSuccessor(agentIndex, a),agentIndex+1,depth,alpha,beta)
                if val>v:
                    v = val
                    best_action = a
                if v> beta:
                    return v
                alpha = max(alpha,v)
            if depth == self.depth:
                return best_action
            else:
                return v

        def minValue(state,agentIndex,depth,alpha,beta):
            v = float('inf')
            actions = state.getLegalActions(agentIndex)
            for a in actions:
                if agentIndex <state.getNumAgents()-1:
                    v = min(v, Value(state.generateSuccessor(agentIndex, a), agentIndex + 1, depth,alpha,beta))
                else:
                    v = min(v, Value(state.generateSuccessor(agentIndex, a), 0, depth-1,alpha,beta))
                if v<alpha:
                    return v
                beta = min(beta,v)
            return v


        depth = self.depth

        alph = -float('inf')
        bet = float('inf')
        return Value(gameState,0,depth,alph,bet)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def isTerminal(state,depth):
           return state.isWin() or state.isLose() or depth==0
        def Value(state,agentIndex,depth):
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            if agentIndex==0:
                return maxValue(state,agentIndex,depth)
            else:
                return expValue(state,agentIndex,depth)

        def maxValue(state,agentIndex,depth):
            v = -float('inf')
            actions = state.getLegalActions(agentIndex)
            best_action = Directions.STOP
            for a in actions:
                val = Value(state.generateSuccessor(agentIndex, a),agentIndex+1,depth)
                if val>v:
                    v = val
                    best_action = a
            if depth==self.depth:
                return best_action
            else:
                return v

        def expValue(state,agentIndex,depth):
            v = 0
            actions = state.getLegalActions(agentIndex)
            prob = 1.0/len(actions)
            for a in actions:
                if agentIndex <state.getNumAgents()-1:
                    val = Value(state.generateSuccessor(agentIndex, a), agentIndex + 1, depth)
                else:
                    val = Value(state.generateSuccessor(agentIndex, a), 0, depth-1)
                v = v+prob*val
            return v

        return Value(gameState,0,self.depth)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 4).

      DESCRIPTION: <write something here so we know what you did>
    """
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    foodcount = currentGameState.getNumFood()
    ghoststates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()



    foodScore = 1/(foodcount+1)
    foodDist = float('inf')
    for food in foods:
        foodDist = min(foodDist,manhattanDistance(food,pos))
    if foodcount>0 :
        foodScore +=(1/(foodDist/foodcount+1))

    ghostDist = float('inf')
    for state in ghoststates:
        ghostpos = state.getPosition()
        if pos == ghostpos:
            return -float('inf')
        ghostDist = min(ghostDist,manhattanDistance(ghostpos,pos))
    ghostScore = 1/(ghostDist/(len(ghoststates)+1)+1)

    capDist = float('inf')
    for cap in capsules:
        capDist = min(capDist,manhattanDistance(pos,cap))
    capScore = 1/(capDist/(len(capsules)+1)+1)+1/(len(capsules)+1)

    return score+foodScore+ghostScore+capScore


# Abbreviation
better = betterEvaluationFunction

