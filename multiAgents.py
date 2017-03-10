# multiAgents.py# --------------# Licensing Information:  You are free to use or extend these projects for# educational purposes provided that (1) you do not distribute or publish# solutions, (2) you retain this notice, and (3) you provide clear# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.# The core projects and autograders were primarily created by John DeNero# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).# Student side autograding was added by Brad Miller, Nick Hay, and# Pieter Abbeel (pabbeel@cs.berkeley.edu).from util import manhattanDistancefrom game import Directionsimport random, util, sys, math, threadingfrom game import Agentclass ReflexAgent(Agent):    """      A reflex agent chooses an action at each choice point by examining      its alternatives via a state evaluation function.      The code below is provided as a guide.  You are welcome to change      it in any way you see fit, so long as you don't touch our method      headers.    """    def getAction(self, gameState):        """        You do not need to change this method, but you're welcome to.        getAction chooses among the best options according to the evaluation function.        Just like in the previous project, getAction takes a GameState and returns        some Directions.X for some X in the set {North, South, West, East, Stop}        """        # Collect legal moves and successor states        legalMoves = gameState.getLegalActions()        # Choose one of the best actions        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]        bestScore = max(scores)        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]        chosenIndex = random.choice(bestIndices) # Pick randomly among the best        "Add more of your code here if you want to"        return legalMoves[chosenIndex]    def evaluationFunction(self, currentGameState, action):        """        Design a better evaluation function here.        The evaluation function takes in the current and proposed successor        GameStates (pacman.py) and returns a number, where higher numbers are better.        The code below extracts some useful information from the state, like the        remaining food (newFood) and Pacman position after moving (newPos).        newScaredTimes holds the number of moves that each ghost will remain        scared because of Pacman having eaten a power pellet.        Print out these variables to see what you're getting, then combine them        to create a masterful evaluation function.        """        # Useful information you can extract from a GameState (pacman.py)        successorGameState = currentGameState.generatePacmanSuccessor(action)        newPos = successorGameState.getPacmanPosition()        newFood = successorGameState.getFood()        newGhostStates = successorGameState.getGhostStates()        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]        "*** YOUR CODE HERE ***"        return successorGameState.getScore()class MultiAgentSearchAgent(Agent):    """      This class provides some common elements to all of your      multi-agent searchers.  Any methods defined here will be available      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.      You *do not* need to make any changes here, but you can if you want to      add functionality to all your adversarial search agents.  Please do not      remove anything, however.      Note: this is an abstract class: one that should not be instantiated.  It's      only partially specified, and designed to be extended.  Agent (game.py)      is another abstract class.    """    def __init__(self, evalFn = 'adversarialEvaluationFunction', depth = '2'):        self.index = 0 # Pacman is always agent index 0        self.evaluationFunction = util.lookup(evalFn, globals())        self.depth = int(depth)class MinimaxSearchAgent(MultiAgentSearchAgent):    """      minimax agent with or without alpha-beta pruning.      Superclass of:            MinimaxAgent            AlphaBetaAgent    """    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2', isAlphaBeta = False):        self.alphaBeta = isAlphaBeta        MultiAgentSearchAgent.__init__(self, evalFn, depth)    def getAction(self, gameState):        return self.minimaxDecision(gameState)    def minimaxDecision(self, gameState):        legalActions = gameState.getLegalActions()        #print '----------\n'        minValues = []        for a in legalActions:            if self.alphaBeta:                newstate = gameState.generateSuccessor(self.index, a)                minValues.append(self.minValueAlphaBeta(gameState.generateSuccessor(self.index, a), self.depth, -sys.maxint - 1, sys.maxint))            else:                minValues.append(self.minValue(gameState.generateSuccessor(self.index, a), self.depth))                        #print 'minimax(' + str(a) + '):\t' + str(minValues[-1])        # minimax value is max of min-values        maxValue = max(minValues)        maxIdxs = [i for i in range(len(minValues)) if minValues[i] == maxValue]        # resolve ties        if len(maxIdxs) > 1:            #print 'resolving tie'            bestActions = [legalActions[i] for i in maxIdxs]            rewards = []            for a in bestActions:                s = gameState.generateSuccessor(self.index, a)                rewards.append(scoreEvaluationFunction(s))            maxReward = max(rewards)            rewardIdxs = [j for j in range(len(rewards)) if rewards[j] == maxReward]            if len(rewardIdxs) > 1:                return bestActions[random.choice(rewardIdxs)]            else:                return bestActions[rewardIdxs[0]]                else:            return legalActions[maxIdxs[0]]    def maxValue(self, gameState, cutoff):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth)                # find the max value we can get from here        v = -sys.maxint - 1                for a in gameState.getLegalActions(self.index):            v = max(v, self.minValue(gameState.generateSuccessor(self.index, a), cutoff - 1))        return v    def minValue(self, gameState, cutoff):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth)        # find the min value we can get from here                v = sys.maxint        for i in range(1, gameState.getNumAgents()):            for a in gameState.getLegalActions(i):                v = min(v, self.maxValue(gameState.generateSuccessor(i, a), cutoff - 1))        return v            def maxValueAlphaBeta(self, gameState, cutoff, alpha, beta):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth)        # find the max value we can get from here        v = -sys.maxint - 1                for a in gameState.getLegalActions(self.index):            v = max(v, self.minValueAlphaBeta(gameState.generateSuccessor(self.index, a), cutoff - 1, alpha, beta))            if v >= beta:                return v            alpha = max(alpha, v)        return v    def minValueAlphaBeta(self, gameState, cutoff, alpha, beta):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth)        # find the min value we can get from here                v = sys.maxint        for i in range(1, gameState.getNumAgents()):            for a in gameState.getLegalActions(i):                v = min(v, self.maxValueAlphaBeta(gameState.generateSuccessor(i, a), cutoff - 1, alpha, beta))                if v <= alpha:                    return v                beta = min(beta, v)        return vclass MinimaxAgent(MinimaxSearchAgent):    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2'):        MinimaxSearchAgent.__init__(self, evalFn, depth, False)class AlphaBetaAgent(MinimaxSearchAgent):    """      minimax agent with alpha-beta pruning    """    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2'):        MinimaxSearchAgent.__init__(self, evalFn, depth, True)class ExpectimaxAgent(MultiAgentSearchAgent):    """      Your expectimax agent (question 4)    """    def getAction(self, gameState):        """          Returns the expectimax action using self.depth and self.evaluationFunction          All ghosts should be modeled as choosing uniformly at random from their          legal moves.        """        "*** YOUR CODE HERE ***"        util.raiseNotDefined()def minimaxEvaluationFunction(currentGameState, maxDistance):    #threshold = -50    grid = currentGameState.getFood().data    gridSize = len(grid) * len(grid[0])    threshold = -gridSize / (currentGameState.getNumAgents() - 1)**2    survivalScore = survivalEvaluationFunction(currentGameState, maxDistance)    delta = survivalScore - threshold    if delta < 1:        weight = 0.        #print 'survival Score: ' + str(survivalScore) + '\treward score: 0' + '\tthreshold: ' + str(threshold) + '\ttotal Score: ' + str(survivalScore)        return survivalScore    else:        weight = math.log(float(delta), abs(threshold))    if weight > 1.:        weight = 1.    #print '\tweight: ' + str(weight)    rewardScore = rewardEvaluationFunction(currentGameState, maxDistance)    score = int(survivalScore + weight * rewardScore)    #print 'survival Score: ' + str(survivalScore) + '\treward score: ' + str(rewardScore) + '\tthreshold: ' + str(threshold) + '\ttotal Score: ' + str(score)    return score# evaluation function that takes only adversaries into account; not food, etc.def survivalEvaluationFunction(currentGameState, maxDistance):    # check if the game would be over    if currentGameState.isWin():        return sys.maxint    if currentGameState.isLose():        return -sys.maxint - 1        score = 0.    pmLocation = currentGameState.getPacmanPosition()    numGhosts = currentGameState.getNumAgents() - 1    maxDist = float(maxDistance)    # determine nearest ghost    ghostDistances = [-1 for i in range(1 + numGhosts)]    nearestGhostDist = sys.maxint    nearestGhostIdx = -1    for i in range(1, 1 + numGhosts):        ghostLocation = currentGameState.data.agentStates[i].configuration.getPosition()        ghostDistances[i] = manhattanDistance(pmLocation, ghostLocation)        if ghostDistances[i] < nearestGhostDist:            nearestGhostIdx = i            nearestGhostDist = ghostDistances[i]    # loop over all ghosts    ghostScores = [0 for i in range(1 + numGhosts)]    for i in range(1, numGhosts + 1):        s = currentGameState.data.agentStates[i]        d = ghostDistances[i]        # reward eating ghosts        if currentGameState.data._eaten[i]:            ghostScores[i] = 1.            continue                # reward moving toward ghosts pacman can eat        if s.scaredTimer > 0:            if i == nearestGhostIdx and ghostDistances[i] <= maxDist:                ghostScores[i] = (float(s.scaredTimer) - d)/float(s.scaredTimer)                #print 'scared timer'        else:            if d < 1.1:                return -sys.maxint - 1            ghostScores[i] = -1./math.log(d, maxDist)            '''            if manDist == 1:                runScore = -1000            else:                runScore -= math.log(maxDist, manDist)            '''            #runScore -= float(maxDist/manDist)                #runScore = maxDist**(-(1./(d**2)))    for gs in ghostScores:        score += 100./numGhosts * gs    if score < 0.:        for gs in ghostScores:            if gs < -1.:                score *= abs(gs)    score /= float(numGhosts)    #print '\t' + str(int(score))    return int(score)def rewardEvaluationFunction(currentGameState, maxDistance):    score = 0    pmLocation = currentGameState.getPacmanPosition()    grid = currentGameState.getFood().data    numX = len(grid) - 2    numY = len(grid[0]) - 2    wallsGrid = currentGameState.data.layout.walls    numSquares = numX * numY    numFood = currentGameState.getNumFood()    if numFood == 0:        return sys.maxint    if currentGameState.data._foodEaten == pmLocation:        score = 200. / numSquares    proxScore = 0.    m = 1            for j in range(len(grid)):        for i in range(len(grid[j])):            if (j, i) == pmLocation or wallsGrid[j][i]:                continue            if grid[j][i] == False:                score += 100. / numSquares            else:                for h in range(j - m, j + m):                    for g in range(i - m, i + m):                        if h < 1 or g < 1 or h > len(grid) - 2 or g > len(grid[j]) - 2:                            continue                        if grid[h][g] == False:                            score -= 100. / numSquares            # reward pacman if he is close to this food            fp = 1. / math.log(float(manhattanDistance(pmLocation, (i, j)) + 0.01), maxDistance)            proxScore += fp / numFood    #print 'prox score: ' + str(proxScore)    return int(score)def scoreEvaluationFunction(currentGameState):    """      This default evaluation function just returns the score of the state.      The score is the same one displayed in the Pacman GUI.      This evaluation function is meant for use with adversarial search agents      (not reflex agents).    """    return currentGameState.getScore()