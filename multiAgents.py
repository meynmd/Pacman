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
        print ' I am  in getAction'
        legalMoves = gameState.getLegalActions()
        #legalMoves_ghost =

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print scores
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
        print '###########################################'
        print 'I am in evaluation function'
        print currentGameState.getPacmanPosition()
        print currentGameState.getScore()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        g_states=currentGameState.getGhostStates()
        for j in range(len(g_states)):
            print j,g_states[j]
        for j in  range(len(newGhostStates)):
            print '***'
            print j,newGhostStates[j]

            print '***'









        "*** YOUR CODE HERE ***"
        #print 'Succesor Game State'
        #print newPos#newFood,newGhostStates,newScaredTimes
        #print newFood
        #print '###########################################'
        #for s in newGhostStates:
        print action
        print successorGameState.getPacmanPosition()
        print successorGameState.getScore()
        print newScaredTimes
        print '###########################################'

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
        #self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)





class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def mini_max_decision(self, gameState):
        legalMoves = gameState.getLegalActions()
        max_action = [float("-inf"), float("-inf")]
        for action in legalMoves:
            temp_state = gameState.generatePacmanSuccessor(action)
            temp_value  = self.Min_value(temp_state)
            if temp_value > max_action[0]:
                max_action=[temp_value,action]
        print max_action
        return max_action[1]





    def Max_value(self, gameState):
        if self.cur_depth==0:
            return self.evaluationFunction(gameState)

        max_action = float("-inf")
        legal_actions = gameState.getLegalActions()
        for action in legal_actions:
            temp_state = gameState.generatePacmanSuccessor(action)
            temp_value = self.Min_value(temp_state)
            if temp_value > max_action:
                max_action = temp_value

        return max_action







    def Min_value(self, gameState):
        if self.cur_depth==0:
            return self.evaluationFunction(gameState)

        min_act = float("inf")
        cur_ghost = gameState.getGhostStates()
        self.cur_depth -=1


        for j in range(len(cur_ghost)):
            legal_actions=gameState.getLegalActions(j+1)
            for act in legal_actions:
                new_ghost_state  = gameState.generateGhostSuccessor(act,j+1)
                temp_max = self.Max_value(new_ghost_state)
                if  temp_max < min_act:
                    min_act = temp_max

        return min_act






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
        """
        self.cur_depth = self.depth
        action = self.mini_max_decision(gameState)

        return action




    def evaluationFunction(self, currentGameState):

        ghost_states_positions = currentGameState.getGhostPositions()
        pacman_position = currentGameState.getPacmanPosition()
        manh = 0
        for pos in ghost_states_positions:
            manh=+(pos[0]-pacman_position[0])**2+ (pos[1]-pacman_position[1])**2



        # print 'kdk'

        return currentGameState.getScore()+(manh*10)




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





def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

