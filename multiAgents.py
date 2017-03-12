# multiAgents.py# --------------# Licensing Information:  You are free to use or extend these projects for# educational purposes provided that (1) you do not distribute or publish# solutions, (2) you retain this notice, and (3) you provide clear# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.# The core projects and autograders were primarily created by John DeNero# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).# Student side autograding was added by Brad Miller, Nick Hay, and# Pieter Abbeel (pabbeel@cs.berkeley.edu).from util import manhattanDistancefrom game import Directionsimport random, util, sys, mathfrom game import Agentclass ReflexAgent(Agent):    """      A reflex agent chooses an action at each choice point by examining      its alternatives via a state evaluation function.      The code below is provided as a guide.  You are welcome to change      it in any way you see fit, so long as you don't touch our method      headers.    """    def getAction(self, gameState):        """        You do not need to change this method, but you're welcome to.        getAction chooses among the best options according to the evaluation function.        Just like in the previous project, getAction takes a GameState and returns        some Directions.X for some X in the set {North, South, West, East, Stop}        """        # Collect legal moves and successor states        legalMoves = gameState.getLegalActions()        # Choose one of the best actions        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]        bestScore = max(scores)        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]        chosenIndex = random.choice(bestIndices) # Pick randomly among the best        "Add more of your code here if you want to"        return legalMoves[chosenIndex]    def evaluationFunction(self, currentGameState, action):        """        Design a better evaluation function here.        The evaluation function takes in the current and proposed successor        GameStates (pacman.py) and returns a number, where higher numbers are better.        The code below extracts some useful information from the state, like the        remaining food (newFood) and Pacman position after moving (newPos).        newScaredTimes holds the number of moves that each ghost will remain        scared because of Pacman having eaten a power pellet.        Print out these variables to see what you're getting, then combine them        to create a masterful evaluation function.        """        # Useful information you can extract from a GameState (pacman.py)        successorGameState = currentGameState.generatePacmanSuccessor(action)        newPos = successorGameState.getPacmanPosition()        newFood = successorGameState.getFood()        newGhostStates = successorGameState.getGhostStates()        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]        "*** YOUR CODE HERE ***"        return successorGameState.getScore()class MultiAgentSearchAgent(Agent):    """      This class provides some common elements to all of your      multi-agent searchers.  Any methods defined here will be available      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.      You *do not* need to make any changes here, but you can if you want to      add functionality to all your adversarial search agents.  Please do not      remove anything, however.      Note: this is an abstract class: one that should not be instantiated.  It's      only partially specified, and designed to be extended.  Agent (game.py)      is another abstract class.    """    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', obsRadius = sys.maxint):        self.index = 0 # Pacman is always agent index 0        self.evaluationFunction = util.lookup(evalFn, globals())        self.depth = int(depth)        self.observableRadius = float(obsRadius)class MinimaxSearchAgent(MultiAgentSearchAgent):    """      minimax agent with or without alpha-beta pruning.      Superclass of:            MinimaxAgent            AlphaBetaAgent    """    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2', isAlphaBeta = False, obsRadius = sys.maxint):        self.alphaBeta = isAlphaBeta        MultiAgentSearchAgent.__init__(self, evalFn, depth, obsRadius)    def getAction(self, gameState):        return self.minimaxDecision(gameState)    def minimaxDecision(self, gameState):        """        minimaxDecision        returns the best decision from gameState given max depth self.depth        """        #print '----------\n'        legalActions = gameState.getLegalActions()        minValues = []        for a in legalActions:            if self.alphaBeta:                newstate = gameState.generateSuccessor(self.index, a)                minValues.append(self.minValueAlphaBeta(gameState.generateSuccessor(self.index, a), self.depth, -sys.maxint - 1, sys.maxint))            else:                minValues.append(self.minValue(gameState.generateSuccessor(self.index, a), self.depth))                        #print 'minimax(' + str(a) + '):\t' + str(minValues[-1])        # minimax value is max of min-values        maxValue = max(minValues)        maxIdxs = [i for i in range(len(minValues)) if minValues[i] == maxValue]        bestActions = []        for idx in maxIdxs:            bestActions.append(legalActions[idx])        if len(bestActions) == 1:            return bestActions[0]        else:            # resolve ties            rewards = []            for a in bestActions:                s = gameState.generateSuccessor(self.index, a)                rewards.append(scoreEvaluationFunction(s))            maxReward = max(rewards)            rewardIdxs = [j for j in range(len(rewards)) if rewards[j] == maxReward]                        if len(rewardIdxs) > 1:                return bestActions[random.choice(rewardIdxs)]            else:                return bestActions[rewardIdxs[0]]    def maxValue(self, gameState, cutoff):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth, self.observableRadius)                # find the max value we can get from here        v = -sys.maxint - 1                for a in gameState.getLegalActions(self.index):            v = max(v, self.minValue(gameState.generateSuccessor(self.index, a), cutoff - 1))        return v    def minValue(self, gameState, cutoff):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth, self.observableRadius)        # find the min value we can get from here                v = sys.maxint        states = generateResultStates([gameState], 1, gameState.getNumAgents())                for s in states:            v = min(v, self.maxValue(s, cutoff - 1))        return v            def maxValueAlphaBeta(self, gameState, cutoff, alpha, beta):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth, self.observableRadius)        # find the max value we can get from here        v = -sys.maxint - 1                for a in gameState.getLegalActions(self.index):            v = max(v, self.minValueAlphaBeta(gameState.generateSuccessor(self.index, a), cutoff - 1, alpha, beta))            if v >= beta:                return v            alpha = max(alpha, v)        return v    def minValueAlphaBeta(self, gameState, cutoff, alpha, beta):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth, self.observableRadius)        # find the min value we can get from here                v = sys.maxint        states = generateResultStates([gameState], 1, gameState.getNumAgents())        for s in states:            v = min(v, self.maxValueAlphaBeta(s, cutoff - 1, alpha, beta))                        if v <= alpha:                return v            beta = min(beta, v)        return vclass MinimaxAgent(MinimaxSearchAgent):    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2', obsRadius = sys.maxint):        MinimaxSearchAgent.__init__(self, evalFn, depth, False, obsRadius)class AlphaBetaAgent(MinimaxSearchAgent):    """      minimax agent with alpha-beta pruning    """    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2', obsRadius = sys.maxint):        MinimaxSearchAgent.__init__(self, evalFn, depth, True, obsRadius)class ExpectimaxAgent(MultiAgentSearchAgent):    """      expectimax agent    """    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2', obsRadius = sys.maxint):        MultiAgentSearchAgent.__init__(self, evalFn, depth, obsRadius)    def getAction(self, gameState):        """          Returns the expectimax action using self.depth and self.evaluationFunction          All ghosts should be modeled as choosing uniformly at random from their          legal moves.        """                return self.maxAction(gameState, self.depth)    def maxAction(self, gameState, depth):        numAgents = gameState.getNumAgents()        actions = [a for a in gameState.getLegalActions(0)]        actionValues = [self.chanceValue(gameState.generateSuccessor(0, a), self.depth) for a in actions]        #print str(actionValues)        bestValue = max(actionValues)        bestIdxs = [i for i in range(0, len(actionValues)) if actionValues[i] == bestValue]        return actions[bestIdxs[0]]    def chanceValue(self, gameState, depth):        states = self.generateChanceStates([gameState], 1, gameState.getNumAgents())        if len(states) < 1:            return None        value = 0        for s in states:            if s.isWin():                value += 1000000            elif s.isLose():                value = -1000000            else:                value += self.evaluationFunction(s, depth)                return int(value / len(states))            def generateChanceStates(self, gameStates, firstIdx, lastIdx):        if firstIdx == lastIdx:            return gameStates        else:            newStates = []            for s in gameStates:                for a in s.getLegalActions(firstIdx):                    newStates.append(s.generateSuccessor(firstIdx, a))                        if len(newStates) < 1:                return gameStates            return self.generateChanceStates(newStates, firstIdx + 1, lastIdx)def minimaxEvaluationFunction(currentGameState, maxDistance, obsRange):    grid = currentGameState.getFood().data    gridSize = len(grid) * len(grid[0])    threshold = -gridSize / (currentGameState.getNumAgents() - 1)**2    survivalScore = survivalEvaluationFunction(currentGameState, 2*maxDistance, obsRange)    delta = survivalScore - threshold    if delta < 1:        print 'survival Score: ' + str(survivalScore) + '\treward score: 0' + '\tthreshold: ' + str(threshold) + '\ttotal Score: ' + str(survivalScore)        return survivalScore    else:        weight = math.log(float(delta), abs(threshold))    if weight > 1.:        weight = 1.    rewardScore = rewardEvaluationFunction(currentGameState, maxDistance, obsRange)    score = int(survivalScore + weight * rewardScore)    print 'survival Score: ' + str(survivalScore) + '\treward score: ' + str(rewardScore) + '\tthreshold: ' + str(threshold) + '\ttotal Score: ' + str(score)    return scoredef getObservableGhostPositions(currentGameState, cornerVisRange):    if cornerVisRange == sys.maxint:        positions = [(-1, -1)]        positions.extend(currentGameState.getGhostPositions())        return positions    pmX, pmY = currentGameState.getPacmanPosition()    wallGrid = currentGameState.data.layout.walls    nBound = pmY    sBound = pmY    wBound = pmX    eBound = pmX    ghostPositions = currentGameState.getGhostPositions()    ghostXs = [x for (x, y) in ghostPositions]    ghostYs = [y for (x, y) in ghostPositions]    while nBound < wallGrid.height and not wallGrid[pmX][nBound + 1]:        nBound += 1    while sBound > 0 and not wallGrid[pmX][sBound - 1]:        sBound -= 1    while eBound < wallGrid.width and not wallGrid[eBound + 1][pmY]:        eBound += 1    while wBound > 0 and not wallGrid[wBound - 1][pmY]:        wBound -= 1    #print '----------\nN: ' + str(nBound) +'\tE: ' + str(eBound) + '\tS: ' + str(sBound) + '\tW: ' + str(wBound)    visGhostPositions = []         visGhostPositions.append((-1, -1))     # want indexing from 1    for (x, y) in ghostPositions:        if (x == pmX and sBound <= y and nBound >= y) or (y == pmY and wBound <= x and eBound >= x):            visGhostPositions.append((x, y))                    elif (abs(pmX - x) <= cornerVisRange and abs(pmY - y) <= cornerVisRange):            visGhostPositions.append((x, y))        else:            visGhostPositions.append((-1, -1))    return visGhostPositions# evaluation function that takes only adversaries into account; not food, etc.def survivalEvaluationFunction(currentGameState, maxDistance, obsRange):    # check if the game would be over    if currentGameState.isWin():        return sys.maxint    if currentGameState.isLose():        return -sys.maxint - 1        score = 0.    pmLocation = currentGameState.getPacmanPosition()    #numGhosts = currentGameState.getNumAgents() - 1    maxDist = float(maxDistance)    ghostLocations = getObservableGhostPositions(currentGameState, obsRange)    numGhosts = len(ghostLocations) - 1    '''    print '---------\n'    for g in ghostLocations:        if g != (-1, -1):            print 'I can see ghost at' + str(g)    '''    # determine nearest ghost    ghostDistances = [-1 for i in range(1 + numGhosts)]    nearestGhostDist = sys.maxint    nearestGhostIdx = -1    for i in range(1, 1 + numGhosts):        #ghostLocation = currentGameState.data.agentStates[i].configuration.getPosition()        ghostLocation = ghostLocations[i]        if ghostLocation == (-1, -1):            ghostDistances[i] = sys.maxint        else:            ghostDistances[i] = manhattanDistance(pmLocation, ghostLocation)                    if ghostDistances[i] < nearestGhostDist:                nearestGhostIdx = i                nearestGhostDist = ghostDistances[i]    # loop over all ghosts    ghostScores = [0 for i in range(1 + numGhosts)]    for i in range(1, 1 + numGhosts):        # reward eating ghosts        if currentGameState.data._eaten[i]:            ghostScores[i] = 100000.            continue                if ghostLocations[i] == (-1, -1):            continue        # reward being near ghosts pacman can eat, penalize being near ghosts that can eat him        s = currentGameState.data.agentStates[i]        d = ghostDistances[i]        if s.scaredTimer > 0:            if i == nearestGhostIdx and ghostDistances[i] <= maxDist:                ghostScores[i] = (float(s.scaredTimer) - d)/float(s.scaredTimer)        else:            if d < 1.1:                return -sys.maxint - 1            ghostScores[i] = -1./math.log(d, maxDist)    for gs in ghostScores:        score += 100./numGhosts * gs    if score < 0.:        for gs in ghostScores:            if gs < -1.:                score *= abs(gs)    #score /= float(numGhosts)    return int(score)def rewardEvaluationFunction(currentGameState, maxDistance, obsRange):    if obsRange == sys.maxint:        return fullObsRewardEvaluationFunction(currentGameState, maxDistance)    else:        return partialObsRewardFunction(currentGameState, obsRange)def partialObsRewardFunction(currentGameState, cornerVisRange):    pmX, pmY = currentGameState.getPacmanPosition()    wallGrid = currentGameState.data.layout.walls    nBound = pmY    sBound = pmY    wBound = pmX    eBound = pmX    score = 0    foodGrid = currentGameState.getFood().data    visibleFood = []    visSqareCount = 1    while nBound < wallGrid.height - 1 and not wallGrid[pmX][nBound + 1]:        y = nBound + 1        if foodGrid[pmX][y]:            visibleFood.append((pmX, y))            score += manhattanDistance((pmX, pmY), (pmX, y))        nBound += 1        visSqareCount +=1    while sBound > 0 and not wallGrid[pmX][sBound - 1]:        y = sBound - 1        if foodGrid[pmX][y]:            visibleFood.append((pmX, y))            score += manhattanDistance((pmX, pmY), (pmX, y))        sBound -= 1        visSqareCount +=1    while eBound < wallGrid.width - 1 and not wallGrid[eBound + 1][pmY]:        x = eBound + 1        if foodGrid[x][pmY]:            visibleFood.append((x, pmY))            score += manhattanDistance((pmX, pmY), (x, pmY))        eBound += 1        visSqareCount +=1    while wBound > 0 and not wallGrid[wBound - 1][pmY]:        x = wBound - 1        if foodGrid[x][pmY]:            visibleFood.append((x, pmY))            score += manhattanDistance((pmX, pmY), (x, pmY))        wBound -= 1        visSqareCount +=1    #print '----------\nVisible food:\n' + str(visibleFood) + '\n'    return int(100. * score / visSqareCount)    def fullObsRewardEvaluationFunction(currentGameState, maxDistance):    score = 0    pmLocation = currentGameState.getPacmanPosition()    grid = currentGameState.getFood().data    numX = len(grid) - 2    numY = len(grid[0]) - 2    wallsGrid = currentGameState.data.layout.walls    numSquares = numX * numY    numFood = currentGameState.getNumFood()    if numFood == 0:        return sys.maxint    if currentGameState.data._foodEaten == pmLocation:        score = 200. / numSquares    proxScore = 0.    m = 1            for j in range(len(grid)):        for i in range(len(grid[j])):            if (j, i) == pmLocation or wallsGrid[j][i]:                continue            if grid[j][i] == False:                score += 100. / numSquares            else:                for h in range(j - m, j + m):                    for g in range(i - m, i + m):                        if h < 1 or g < 1 or h > len(grid) - 2 or g > len(grid[j]) - 2:                            continue                        if grid[h][g] == False:                            score -= 100. / numSquares            # reward pacman if he is close to this food            #fp = 1. / math.log(float(manhattanDistance(pmLocation, (i, j)) + 0.01), maxDistance)            fp = 100. / 2**manhattanDistance(pmLocation, (i, j))            proxScore += fp / numFood    return int(score + proxScore)def scoreEvaluationFunction(currentGameState, maxDistance = 0, obsRange = 0):    """      This default evaluation function just returns the score of the state.      The score is the same one displayed in the Pacman GUI.      This evaluation function is meant for use with adversarial search agents      (not reflex agents).    """    return currentGameState.getScore()def generateResultStates(gameStates, firstIdx, lastIdx):    if firstIdx == lastIdx:        return gameStates    else:        newStates = []        for s in gameStates:            for a in s.getLegalActions(firstIdx):                newStates.append(s.generateSuccessor(firstIdx, a))                    if len(newStates) < 1:            return gameStates        return generateResultStates(newStates, firstIdx + 1, lastIdx)from random import choicefrom collections import defaultdictclass MonteCarloAgent(Agent):    def __init__(self,):        self.total_reward = defaultdict(int)        self.visits = defaultdict(int)        self.parent_state = defaultdict(list)    def getAction(self,gameState,index=0,num_iterations=5,num_of_agents=4):        self.visits[(index,gameState)] = 0        self.total_reward[(index,gameState)]= 0        self.parent_state[(index,gameState)] = [('Root','Root')]        print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'        for iter in range(num_iterations):            print 'iter------------------------------------------>',iter            self.tree_traversal(gameState,index)        legalActions = gameState.getLegalActions(index)        max_avg_reward = float('-inf')        max_action = None        for action in legalActions:            successor = gameState.generateSuccessor(index,action)            if index>num_of_agents:                new_index =0            else:                new_index = index+1            visits = self.visits[(new_index,successor)]            if visits > 0:                cur_avg_reward = self.total_reward[(new_index,successor)]/visits            else:                raise(BaseException('Hey, Harish, I don\'t know what to do here! There have been no visits by agent #' + str(new_index) + ' to this state!'))            print cur_avg_reward            if cur_avg_reward > max_avg_reward:                max_avg_reward = cur_avg_reward                max_action = action        print 'HEre is the max Action =================================================>',max_action        return max_action    def ucb_calculate(self,index,gameState,parent_index,parent_state):        exploitation = self.total_reward[(index,gameState)]/self.visits[(index,gameState)]        exploration =  2*math.sqrt(math.log(self.visits[(parent_index,parent_state)]))        return exploitation+exploration    def update(self,index,gameState,score):        if self.parent_state[(index,gameState)]== [('Root','Root')]:            return        tuple_out = self.parent_state[(index, gameState)][0]        self.visits[tuple_out]+=1        self.total_reward[tuple_out]+=score        return self.update(tuple_out[0],tuple_out[1],score)    def tree_traversal(self,gameState,index,max_depth=100,num_of_agents=2):        cur_state = gameState        tree_expansion = True        for depth in range(1,max_depth+1):            ##Tree expansion code            if tree_expansion and self.visits[(index,cur_state)]==0:                print 'fggghvvg!!!!!!!!!!!!!!!!!!!!!!!!!'                rollout = True                tree_expansion=False                parent_index = index                parent_state = cur_state                legalActions = cur_state.getLegalActions(index)                successor_states = [(index, cur_state.generateSuccessor(index, action)) for action in legalActions]                index,cur_state = successor_states[0]                if index>=num_of_agents:                    index=0                else:                    index+=1                self.visits[(parent_index,parent_state)] = 1                self.parent_state[(index,cur_state)] = [(parent_index,parent_state)]            if self.visits[(index,cur_state)]>0:                print 'expansion phase'                indicator = True                rollout = True                tree_expansion = False                legalActions = cur_state.getLegalActions(index)                parent_index = index                parent_state = cur_state                successor_states = [(index, cur_state.generateSuccessor(index, action)) for action in legalActions]                if index>=num_of_agents:                    index=0                else:                    index+=1                max_val = float('-inf')                for child_state in successor_states:                    if self.visits[(index,child_state[1])] ==0:                        cur_state = child_state[1]                        print 'I am here '                        break                    ucb_val = self.ucb_calculate(index,child_state[1],parent_index,parent_state)                    if ucb_val>max_val:                        max_val = ucb_val                        cur_state = child_state[1]                print ' I am here toooo!!!!!!'                self.parent_state[(index,cur_state)] = [(parent_index,parent_state)]            ### This were we start rollout code.            if rollout:                print  'hjhjjkkm'                rollout_node = cur_state                rollout_index = index                rollout = False            legalActions = cur_state.getLegalActions(index)            if len(legalActions) > 0:                random_action = choice(legalActions)                cur_state = cur_state.generateSuccessor(index, random_action)                if index >= num_of_agents:                    index = 0                else:                    index += 1            elif depth==max_depth or cur_state.isWin() or cur_state.isLose():                score = cur_state.getScore()                print 'Before Update'                print self.total_reward                print self.visits                print '$$#$#$#$#$#$#$#$#$#$#$'                self.total_reward[(rollout_index,rollout_node)]+=score                self.visits[(rollout_index,rollout_node)] += 1                self.update(rollout_index,rollout_node,score)                print 'After Update'                print self.total_reward                print self.visits                print '$$#$#$#$#$#$#$#$#$#$#$'                break            else:                raise Exception(BaseException('No legal actions for agent #' + str(index) + ' and no one has won the game!'))