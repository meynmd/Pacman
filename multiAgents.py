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
import random, util, sys, math

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

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


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

    def __init__(self, evalFn = 'adversarialEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      minimax agent
    """

    def getAction(self, gameState):
        return self.minimaxDecision(gameState)


    def minimaxDecision(self, gameState):
        legalActions = gameState.getLegalActions()

        minValues = []
        for a in legalActions:
            minValues.append(self.minValue(gameState.generateSuccessor(self.index, a), self.depth))
            print 'minimax value of ' + str(a) + ' is ' + str(minValues[-1])

        # minimax value is max of min-values
        maxValue = max(minValues)
        maxIdxs = [i for i in range(len(minValues)) if minValues[i] == maxValue]

        # resolve ties
        if len(maxIdxs) > 1:
            print 'resolving tie'
            bestActions = [legalActions[i] for i in maxIdxs]
            rewards = []
            for a in bestActions:
                s = gameState.generateSuccessor(self.index, a)
                rewards.append(rewardEvaluationFunction(s , 10 * self.depth))

            maxReward = max(rewards)
            rewardIdxs = [j for j in range(len(rewards)) if rewards[j] == maxReward]

            if len(rewardIdxs) > 1:
                return bestActions[random.choice(rewardIdxs)]
            else:
                return bestActions[rewardIdxs[0]]
        
        else:
            return legalActions[maxIdxs[0]]


    def maxValue(self, gameState, cutoff):
        # value of terminal state is given by evaluation fn
        if cutoff == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState, self.depth)
        
        v = -sys.maxint - 1
        # find the max value we can get from here
        for a in gameState.getLegalActions(self.index):
            v = max(v, self.minValue(gameState.generateSuccessor(self.index, a), cutoff - 1))

        #print 'maxValue returning ' + str(v)

        return v


    def minValue(self, gameState, cutoff):
        # value of terminal state is given by evaluation fn
        if cutoff == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState, self.depth)
        
        v = sys.maxint
        # find the min value we can get from here
        for a in gameState.getLegalActions(self.index):
            v = min(v, self.maxValue(gameState.generateSuccessor(self.index, a), cutoff - 1))

        #print 'minValue returning ' + str(v)

        return v
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


# evaluation function that takes only adversaries into account; not food, etc.
def adversarialEvaluationFunction(currentGameState, maxDistance):
    if currentGameState.isWin():
        return 1000000

    if currentGameState.isLose():
        return -1000000

    score = 0
    pmLocation = currentGameState.getPacmanPosition()
    for s in currentGameState.getGhostStates():
        manDist = manhattanDistance(pmLocation, s.configuration.getPosition()) 
        #if manDist < maxDistance:
        if s.scaredTimer > 0:
            score += maxDistance - manDist
        else:
            score += int(math.log(manDist, maxDistance))

    #print 'adversarial score: ' + str(score)

    return score


def rewardEvaluationFunction(currentGameState, maxDistance):
    score = 0
    pmLocation = currentGameState.getPacmanPosition()


    if currentGameState.data._foodEaten == currentGameState.getPacmanPosition():
        score = 10

    foodProx = 0
    count = 0
    grid = currentGameState.getFood().data

    for j in range(len(grid)):
        for i in range(len(grid[j])):
            if (i, j) == pmLocation:
                continue
            if grid[j][i] == True:
                d = manhattanDistance(pmLocation, (i, j))
                count += 1
                foodProx += 1. / float(d)

    score += int(foodProx)

    return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()
