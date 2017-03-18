# multiAgents.py# --------------# Licensing Information:  You are free to use or extend these projects for# educational purposes provided that (1) you do not distribute or publish# solutions, (2) you retain this notice, and (3) you provide clear# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.# The core projects and autograders were primarily created by John DeNero# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).# Student side autograding was added by Brad Miller, Nick Hay, and# Pieter Abbeel (pabbeel@cs.berkeley.edu).from util import manhattanDistancefrom game import Directionsimport random, util, sys, mathfrom game import Agent,Actionsfrom ghostAgents import GhostAgentclass ReflexAgent(Agent):    """      A reflex agent chooses an action at each choice point by examining      its alternatives via a state evaluation function.      The code below is provided as a guide.  You are welcome to change      it in any way you see fit, so long as you don't touch our method      headers.    """    def getAction(self, gameState):        """        You do not need to change this method, but you're welcome to.        getAction chooses among the best options according to the evaluation function.        Just like in the previous project, getAction takes a GameState and returns        some Directions.X for some X in the set {North, South, West, East, Stop}        """        # Collect legal moves and successor states        legalMoves = gameState.getLegalActions()        # Choose one of the best actions        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]        bestScore = max(scores)        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]        chosenIndex = random.choice(bestIndices) # Pick randomly among the best        "Add more of your code here if you want to"        return legalMoves[chosenIndex]    def evaluationFunction(self, currentGameState, action):        """        Design a better evaluation function here.        The evaluation function takes in the current and proposed successor        GameStates (pacman.py) and returns a number, where higher numbers are better.        The code below extracts some useful information from the state, like the        remaining food (newFood) and Pacman position after moving (newPos).        newScaredTimes holds the number of moves that each ghost will remain        scared because of Pacman having eaten a power pellet.        Print out these variables to see what you're getting, then combine them        to create a masterful evaluation function.        """        # Useful information you can extract from a GameState (pacman.py)        successorGameState = currentGameState.generatePacmanSuccessor(action)        newPos = successorGameState.getPacmanPosition()        newFood = successorGameState.getFood()        newGhostStates = successorGameState.getGhostStates()        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]        "*** YOUR CODE HERE ***"        return successorGameState.getScore()class MultiAgentSearchAgent(Agent):    """      This class provides some common elements to all of your      multi-agent searchers.  Any methods defined here will be available      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.      You *do not* need to make any changes here, but you can if you want to      add functionality to all your adversarial search agents.  Please do not      remove anything, however.      Note: this is an abstract class: one that should not be instantiated.  It's      only partially specified, and designed to be extended.  Agent (game.py)      is another abstract class.    """    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', obsRadius = sys.maxint):        self.index = 0 # Pacman is always agent index 0        self.evaluationFunction = util.lookup(evalFn, globals())        self.depth = int(depth)        self.observableRadius = float(obsRadius)class MinimaxSearchAgent(MultiAgentSearchAgent):    """      minimax agent with or without alpha-beta pruning.      Superclass of:            MinimaxAgent            AlphaBetaAgent    """    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2', isAlphaBeta = False, obsRadius = sys.maxint):        self.alphaBeta = isAlphaBeta        MultiAgentSearchAgent.__init__(self, evalFn, depth, obsRadius)    def getAction(self, gameState):        return self.minimaxDecision(gameState)    def minimaxDecision(self, gameState):        """        minimaxDecision        returns the best decision from gameState given max depth self.depth        """        #print '----------\n'        legalActions = gameState.getLegalActions()        minValues = []        for a in legalActions:            if self.alphaBeta:                newstate = gameState.generateSuccessor(self.index, a)                minValues.append(self.minValueAlphaBeta(gameState.generateSuccessor(self.index, a), self.depth, -sys.maxint - 1, sys.maxint))            else:                minValues.append(self.minValue(gameState.generateSuccessor(self.index, a), self.depth))                        #print 'minimax(' + str(a) + '):\t' + str(minValues[-1])        # minimax value is max of min-values        maxValue = max(minValues)        maxIdxs = [i for i in range(len(minValues)) if minValues[i] == maxValue]        bestActions = []        for idx in maxIdxs:            bestActions.append(legalActions[idx])        if len(bestActions) == 1:            return bestActions[0]        else:            # resolve ties            rewards = []            for a in bestActions:                s = gameState.generateSuccessor(self.index, a)                rewards.append(scoreEvaluationFunction(s))            maxReward = max(rewards)            rewardIdxs = [j for j in range(len(rewards)) if rewards[j] == maxReward]                        if len(rewardIdxs) > 1:                return bestActions[random.choice(rewardIdxs)]            else:                return bestActions[rewardIdxs[0]]    def maxValue(self, gameState, cutoff):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth, self.observableRadius)                # find the max value we can get from here        v = -sys.maxint - 1                for a in gameState.getLegalActions(self.index):            v = max(v, self.minValue(gameState.generateSuccessor(self.index, a), cutoff - 1))        return v    def minValue(self, gameState, cutoff):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth, self.observableRadius)        # find the min value we can get from here                v = sys.maxint        states = generateResultStates([gameState], 1, gameState.getNumAgents())                for s in states:            v = min(v, self.maxValue(s, cutoff - 1))        return v            def maxValueAlphaBeta(self, gameState, cutoff, alpha, beta):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth, self.observableRadius)        # find the max value we can get from here        v = -sys.maxint - 1                for a in gameState.getLegalActions(self.index):            v = max(v, self.minValueAlphaBeta(gameState.generateSuccessor(self.index, a), cutoff - 1, alpha, beta))            if v >= beta:                return v            alpha = max(alpha, v)        return v    def minValueAlphaBeta(self, gameState, cutoff, alpha, beta):        # value of terminal state is given by evaluation fn        if cutoff == 0 or gameState.isWin() or gameState.isLose():            return self.evaluationFunction(gameState, self.depth, self.observableRadius)        # find the min value we can get from here                v = sys.maxint        states = generateResultStates([gameState], 1, gameState.getNumAgents())        for s in states:            v = min(v, self.maxValueAlphaBeta(s, cutoff - 1, alpha, beta))                        if v <= alpha:                return v            beta = min(beta, v)        return vclass MinimaxAgent(MinimaxSearchAgent):    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2', obsRadius = sys.maxint):        MinimaxSearchAgent.__init__(self, evalFn, depth, False, obsRadius)class AlphaBetaAgent(MinimaxSearchAgent):    """      minimax agent with alpha-beta pruning    """    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2', obsRadius = sys.maxint):        MinimaxSearchAgent.__init__(self, evalFn, depth, True, obsRadius)class ExpectimaxAgent(MultiAgentSearchAgent):    """      expectimax agent    """    def __init__(self, evalFn = 'minimaxEvaluationFunction', depth = '2', obsRadius = sys.maxint):        MultiAgentSearchAgent.__init__(self, evalFn, depth, obsRadius)    def getAction(self, gameState):        """          Returns the expectimax action using self.depth and self.evaluationFunction          All ghosts should be modeled as choosing uniformly at random from their          legal moves.        """                return self.maxAction(gameState, self.depth)    def maxAction(self, gameState, depth):        numAgents = gameState.getNumAgents()        actions = [a for a in gameState.getLegalActions(0)]        actionValues = [self.chanceValue(gameState.generateSuccessor(0, a), self.depth) for a in actions]        #print str(actionValues)        bestValue = max(actionValues)        bestIdxs = [i for i in range(0, len(actionValues)) if actionValues[i] == bestValue]        return actions[bestIdxs[0]]    def chanceValue(self, gameState, depth):        states = self.generateChanceStates([gameState], 1, gameState.getNumAgents())        if len(states) < 1:            return None        value = 0        for s in states:            if s.isWin():                value += 1000000            elif s.isLose():                value = -1000000            else:                value += self.evaluationFunction(s, depth)                return int(value / len(states))            def generateChanceStates(self, gameStates, firstIdx, lastIdx):        if firstIdx == lastIdx:            return gameStates        else:            newStates = []            for s in gameStates:                for a in s.getLegalActions(firstIdx):                    newStates.append(s.generateSuccessor(firstIdx, a))                        if len(newStates) < 1:                return gameStates            return self.generateChanceStates(newStates, firstIdx + 1, lastIdx)def minimaxEvaluationFunction(currentGameState, maxDistance, obsRange):    grid = currentGameState.getFood().data    gridSize = len(grid) * len(grid[0])    threshold = -gridSize / (currentGameState.getNumAgents() - 1)**2    survivalScore = survivalEvaluationFunction(currentGameState, 2*maxDistance, obsRange)    delta = survivalScore - threshold    if delta < 1:        #print 'survival Score: ' + str(survivalScore) + '\treward score: 0' + '\tthreshold: ' + str(threshold) + '\ttotal Score: ' + str(survivalScore)        return survivalScore    else:        weight = math.log(float(delta), abs(threshold))    if weight > 1.:        weight = 1.    rewardScore = rewardEvaluationFunction(currentGameState, maxDistance, obsRange)    score = int(survivalScore + weight * rewardScore)    #print 'survival Score: ' + str(survivalScore) + '\treward score: ' + str(rewardScore) + '\tthreshold: ' + str(threshold) + '\ttotal Score: ' + str(score)    return scoredef getObservableGhostPositions(currentGameState, cornerVisRange):    if cornerVisRange == sys.maxint:        positions = [(-1, -1)]        positions.extend(currentGameState.getGhostPositions())        return positions    pmX, pmY = currentGameState.getPacmanPosition()    wallGrid = currentGameState.data.layout.walls    nBound = pmY    sBound = pmY    wBound = pmX    eBound = pmX    ghostPositions = currentGameState.getGhostPositions()    ghostXs = [x for (x, y) in ghostPositions]    ghostYs = [y for (x, y) in ghostPositions]    while nBound < wallGrid.height and not wallGrid[pmX][nBound + 1]:        nBound += 1    while sBound > 0 and not wallGrid[pmX][sBound - 1]:        sBound -= 1    while eBound < wallGrid.width and not wallGrid[eBound + 1][pmY]:        eBound += 1    while wBound > 0 and not wallGrid[wBound - 1][pmY]:        wBound -= 1    #print '----------\nN: ' + str(nBound) +'\tE: ' + str(eBound) + '\tS: ' + str(sBound) + '\tW: ' + str(wBound)    visGhostPositions = []         visGhostPositions.append((-1, -1))     # want indexing from 1    for (x, y) in ghostPositions:        if (x == pmX and sBound <= y and nBound >= y) or (y == pmY and wBound <= x and eBound >= x):            visGhostPositions.append((x, y))                    elif (abs(pmX - x) <= cornerVisRange and abs(pmY - y) <= cornerVisRange):            visGhostPositions.append((x, y))        else:            visGhostPositions.append((-1, -1))    return visGhostPositions# evaluation function that takes only adversaries into account; not food, etc.def survivalEvaluationFunction(currentGameState, maxDistance, obsRange):    # check if the game would be over    if currentGameState.isWin():        return sys.maxint    if currentGameState.isLose():        return -sys.maxint - 1        score = 0.    pmLocation = currentGameState.getPacmanPosition()    #numGhosts = currentGameState.getNumAgents() - 1    maxDist = float(maxDistance)    ghostLocations = getObservableGhostPositions(currentGameState, obsRange)    numGhosts = len(ghostLocations) - 1    '''    print '---------\n'    for g in ghostLocations:        if g != (-1, -1):            print 'I can see ghost at' + str(g)    '''    # determine nearest ghost    ghostDistances = [-1 for i in range(1 + numGhosts)]    nearestGhostDist = sys.maxint    nearestGhostIdx = -1    for i in range(1, 1 + numGhosts):        #ghostLocation = currentGameState.data.agentStates[i].configuration.getPosition()        ghostLocation = ghostLocations[i]        if ghostLocation == (-1, -1):            ghostDistances[i] = sys.maxint        else:            ghostDistances[i] = manhattanDistance(pmLocation, ghostLocation)                    if ghostDistances[i] < nearestGhostDist:                nearestGhostIdx = i                nearestGhostDist = ghostDistances[i]    # loop over all ghosts    ghostScores = [0 for i in range(1 + numGhosts)]    for i in range(1, 1 + numGhosts):        # reward eating ghosts        if currentGameState.data._eaten[i]:            ghostScores[i] = 100000.            continue                if ghostLocations[i] == (-1, -1):            continue        # reward being near ghosts pacman can eat, penalize being near ghosts that can eat him        s = currentGameState.data.agentStates[i]        d = ghostDistances[i]        if s.scaredTimer > 0:            if i == nearestGhostIdx and ghostDistances[i] <= maxDist:                ghostScores[i] = (float(s.scaredTimer) - d)/float(s.scaredTimer)        else:            if d < 1.1:                return -sys.maxint - 1            ghostScores[i] = -1./math.log(d, maxDist)    for gs in ghostScores:        score += 100./numGhosts * gs    if score < 0.:        for gs in ghostScores:            if gs < -1.:                score *= abs(gs)    #score /= float(numGhosts)    return int(score)def rewardEvaluationFunction(currentGameState, maxDistance, obsRange):    if obsRange == sys.maxint:        return fullObsRewardEvaluationFunction(currentGameState, maxDistance)    else:        return partialObsRewardFunction(currentGameState, obsRange)def partialObsRewardFunction(currentGameState, cornerVisRange):    pmX, pmY = currentGameState.getPacmanPosition()    wallGrid = currentGameState.data.layout.walls    nBound = pmY    sBound = pmY    wBound = pmX    eBound = pmX    score = 0    foodGrid = currentGameState.getFood().data    visibleFood = []    visSqareCount = 1    while nBound < wallGrid.height - 1 and not wallGrid[pmX][nBound + 1]:        y = nBound + 1        if foodGrid[pmX][y]:            visibleFood.append((pmX, y))            score += manhattanDistance((pmX, pmY), (pmX, y))        nBound += 1        visSqareCount +=1    while sBound > 0 and not wallGrid[pmX][sBound - 1]:        y = sBound - 1        if foodGrid[pmX][y]:            visibleFood.append((pmX, y))            score += manhattanDistance((pmX, pmY), (pmX, y))        sBound -= 1        visSqareCount +=1    while eBound < wallGrid.width - 1 and not wallGrid[eBound + 1][pmY]:        x = eBound + 1        if foodGrid[x][pmY]:            visibleFood.append((x, pmY))            score += manhattanDistance((pmX, pmY), (x, pmY))        eBound += 1        visSqareCount +=1    while wBound > 0 and not wallGrid[wBound - 1][pmY]:        x = wBound - 1        if foodGrid[x][pmY]:            visibleFood.append((x, pmY))            score += manhattanDistance((pmX, pmY), (x, pmY))        wBound -= 1        visSqareCount +=1    #print '----------\nVisible food:\n' + str(visibleFood) + '\n'    return int(100. * score / visSqareCount)    def fullObsRewardEvaluationFunction(currentGameState, maxDistance):    score = 0    pmLocation = currentGameState.getPacmanPosition()    grid = currentGameState.getFood().data    numX = len(grid) - 2    numY = len(grid[0]) - 2    wallsGrid = currentGameState.data.layout.walls    numSquares = numX * numY    numFood = currentGameState.getNumFood()    if numFood == 0:        return sys.maxint    if currentGameState.data._foodEaten == pmLocation:        score = 200. / numSquares    proxScore = 0.    m = 1            for j in range(len(grid)):        for i in range(len(grid[j])):            if (j, i) == pmLocation or wallsGrid[j][i]:                continue            if grid[j][i] == False:                score += 100. / numSquares            else:                for h in range(j - m, j + m):                    for g in range(i - m, i + m):                        if h < 1 or g < 1 or h > len(grid) - 2 or g > len(grid[j]) - 2:                            continue                        if grid[h][g] == False:                            score -= 100. / numSquares            # reward pacman if he is close to this food            #fp = 1. / math.log(float(manhattanDistance(pmLocation, (i, j)) + 0.01), maxDistance)            fp = 100. / 2**manhattanDistance(pmLocation, (i, j))            proxScore += fp / numFood    return int(score + proxScore)def scoreEvaluationFunction(currentGameState, maxDistance = 0, obsRange = 0):    """      This default evaluation function just returns the score of the state.      The score is the same one displayed in the Pacman GUI.      This evaluation function is meant for use with adversarial search agents      (not reflex agents).    """    return currentGameState.getScore()def generateResultStates(gameStates, firstIdx, lastIdx):    if firstIdx == lastIdx:        return gameStates    else:        newStates = []        for s in gameStates:            for a in s.getLegalActions(firstIdx):                newStates.append(s.generateSuccessor(firstIdx, a))                    if len(newStates) < 1:            return gameStates        return generateResultStates(newStates, firstIdx + 1, lastIdx)from random import choicefrom collections import defaultdictclass MonteCarloAgent(Agent):    def __init__(self, depth=3,radius=5):        self.total_reward = defaultdict(list)        ##self.new_reward = defaultdict(list)        self.visits = defaultdict(int)        self.parent_state = defaultdict(list)        self.maxDepth = depth        self.radius = radius    def getAction(self,gameState,index=0,num_iterations=20,numAgents=3):        print '#####################################################$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%%'        self.total_reward = defaultdict(list)        self.visits = defaultdict(int)        self.parent_state = defaultdict(list)        self.children = defaultdict(list)        self.visits[(None, gameState)] = 0        self.total_reward[(None, gameState)]= [0,0,0]        self.parent_state[(None, gameState)] = [('Root', 'Root')]        print gameState.getPacmanPosition()        action=None        decide = self.decision_reward(gameState)        #decide = 2        print "The Decision Made by the agent",decide        #decide=2        initial_food = gameState.getNumFood()        for iter in range(num_iterations):            print 'Iteration Number``````````````````````````````````````````````````````````````````````````````````````````````````----->', iter            self.tree_traversal(gameState,action,decide,initial_food)        max_avg_reward = float('-inf')        max_action = None        print self.total_reward[(None, gameState)]        print len(self.children[(None,gameState)])        print 'making the final decision here@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'        for child_key in self.children[(None,gameState)]:            print child_key[0], self.total_reward[child_key][decide], self.visits[child_key]            if self.total_reward[child_key][decide]/self.visits[child_key] > max_avg_reward :                #print child_key[0],self.total_reward[child_key][decide],self.visits[child_key]                max_avg_reward = self.total_reward[child_key][decide]/self.visits[child_key]                max_action =  child_key[0]        if max_action == None:            print 'No max_action; taking first one'            max_action = legalActions[0]        print 'HEre is the max Action =================================================>', max_action,max_avg_reward        return max_action    def decision_reward(self,gamestate,numAgents=3):        distance_list = []        decision = 1        pac_pos=gamestate.getPacmanPosition()        for i in range(1,numAgents):            ghost_pos = gamestate.getGhostPosition(i)            if manhattanDistance(pac_pos,ghost_pos) < self.radius:                decision = 0        return decision    def ucb_calculate(self,action,gameState,parent_action,parent_state,decide):        try:            exploitation = self.total_reward[(action,gameState)][decide]/(self.visits[(action,gameState)]+1)            exploration =  2*math.sqrt(math.log(1+self.visits[(parent_action,parent_state)]))        except:            print 'dkfjdkjf'        return exploitation+exploration    def update(self,action,gameState,scores):        if self.parent_state[(action,gameState)]== [('Root','Root')]:            return        tuple_out = self.parent_state[(action, gameState)][0]        self.visits[tuple_out]+=1        self.total_reward[tuple_out][0]+=scores[0]        self.total_reward[tuple_out][1]+=scores[1]        self.total_reward[tuple_out][2]+=scores[2]        return self.update(tuple_out[0],tuple_out[1],scores)    def terminal_scores(self,initial_food,cur_state,loss=0):        food  = initial_food-cur_state.getNumFood()        score = cur_state.getScore()        survived = loss        return [survived,food,score]    def simulate_random_walk(self,rollout_state,action,initial_food,max_depth=4,lost=False,index=0,numAgents=3):        cur_state= rollout_state        print 'Here------------------------------------->'        for depth in range(max_depth):            if cur_state.isWin():                temp_cur_state=cur_state                break            if cur_state.isLose():                temp_cur_state=cur_state                lost=True                break            legalActions = cur_state.getLegalActions(index)            random_action = choice(legalActions)            temp_cur_state = cur_state.generateSuccessor(index, random_action)            temp_cur_state1 = temp_cur_state            for i in range(1, numAgents):                if temp_cur_state1.isLose():                    lost = True                    cur_state = temp_cur_state                    break                act = ghostmove_action(i, temp_cur_state)                if temp_cur_state1.generateSuccessor(i, act) != []:                    temp_cur_state1 = temp_cur_state1.generateSuccessor(i, act)                else:                    lost = True                    cur_state = temp_cur_state                    break            if temp_cur_state!=temp_cur_state1:                temp_cur_state=temp_cur_state1            if temp_cur_state.isLose():                lost=True                break            else:                cur_state = temp_cur_state        cur_state = temp_cur_state        if lost:            return self.terminal_scores(initial_food,cur_state,0)        else:            return self.terminal_scores(initial_food,cur_state,1)    def ghost_state_generation(self,cur_state,numAgents=3,roll_out=False):        initial_state = cur_state        for i in range(1, numAgents):            if cur_state.isLose():                print 'Ghost Moved I died'                cur_state =initial_state                roll_out = True                break            act_1 = ghostmove_action(i, cur_state)            cur_state = cur_state.generateSuccessor(i, act_1)        return roll_out,cur_state    def choose_best_ucb(self,action,cur_state,parent_action,parent_state,decide):        max_val = float('-inf')        for act, child_state in self.children[(action, cur_state)]:            ucb_val = self.ucb_calculate(act, child_state, parent_action, parent_state, decide)            if ucb_val > max_val:                max_val = ucb_val                cur_state = child_state                action = act        return action,cur_state    def find_rollout_node(self,cur_state,action,decide,index=0,numAgents=3,roll_out=False,lost=False,failed=False,already_visited_by_some_parent=False):        parent_action = action        parent_state = cur_state        legalActions = cur_state.getLegalActions(index)        if legalActions ==[]:            print 'hhjhjjjjjjjjj!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'            failed = True            return True,action,cur_state, failed        legalActions = list(set(legalActions) - set([c[0] for c in self.children[(action, cur_state)]]))        if legalActions != []:            successor_states = [(act, cur_state.generateSuccessor(index, act)) for act in legalActions]            failed_count=0            for act, child_state in successor_states:                temp_roll,temp_c_state=self.ghost_state_generation(child_state)                if temp_roll:                    failed_count+=1                    if failed_count!=len(successor_states):                        continue                    else:                        print 'Ohhhh god no children'                        failed = True                        roll_out = True                        break                if self.visits[(act, temp_c_state)] == 0 :                    #print action                    action, cur_state = act, temp_c_state                    roll_out=True                    break                if self.visits[(act,temp_c_state)]!=0:                    action,cur_state = act,temp_c_state                    roll_out = True                    already_visited_by_some_parent = True                    break        else:           action,cur_state = self.choose_best_ucb( action, cur_state,parent_action,parent_state,decide)        if failed:            if self.children[(parent_action, parent_state)]!=[]:                if len(self.children[(parent_action, parent_state)]) ==1:                    action, cur_state = self.children[(parent_action, parent_state)][0]                else:                    print 'I have 1+ actions to choosseeeeeeeeee'                    print self.children[(parent_action,parent_state)]                    temp_ucb=float('-inf')                    for act10,state10 in self.children[(parent_action,parent_state)]:                        val_ucb =self.ucb_calculate(act10,state10,parent_action,parent_state,decide)                        #print act10,val_ucb                        if val_ucb>temp_ucb:                            action=act10                            cur_state=state10                            temp_ucb = val_ucb                    print action                    #support_parent = self.parent_state[(parent_action, parent_state)][0]                    #action,cur_state= self.choose_best_ucb(parent_action,parent_state,support_parent[0],support_parent[1],decide)                failed=True               #print 'Even I dont have children I have some one to support'                #print self.children[(parent_action, parent_state)]                return roll_out,action,cur_state,failed            else:                print 'I dont have any one to support for me'                action,cur_state = parent_action,parent_state                return roll_out,action,cur_state,failed        if not already_visited_by_some_parent:            self.parent_state[(action, cur_state)] = [(parent_action, parent_state)]            if (action, cur_state) not in self.children[(parent_action, parent_state)]:                self.children[(parent_action, parent_state)] += [(action, cur_state)]            return roll_out,action,cur_state,failed    def tree_traversal(self,gameState,action,decide,initial_food,numAgents=3,random_decision=True,index=0,simulate=False):        cur_state = gameState        max_iterations = 5000        rollout_action=None        rollout_node=None        for k in range(max_iterations):            #try:            rollout,rollout_action,rollout_node,failed=self.find_rollout_node(cur_state,action,decide)            #except:            #print 'kdjfkdjkfdfjdkjfkdjfjdkfjkdjfkdjk'            if failed == True:                self.visits[(rollout_action, rollout_node)] += 1                scores = self.terminal_scores(initial_food,rollout_node,loss=0)                self.total_reward[(rollout_action, rollout_node)][0] += scores[0]                self.total_reward[(rollout_action, rollout_node)][1] += scores[1]                self.total_reward[(rollout_action, rollout_node)][2] += scores[2]                self.update(rollout_action, rollout_node, scores)                break            cur_state = rollout_node            action = rollout_action            if rollout:                break        if failed:            return        scores=self.simulate_random_walk(rollout_node,rollout_action,initial_food)        self.visits[(rollout_action, rollout_node)]+=1        if self.total_reward[(rollout_action, rollout_node)] == []:            self.total_reward[(rollout_action, rollout_node)] = scores        else:            self.total_reward[(rollout_action, rollout_node)][0] += scores[0]            self.total_reward[(rollout_action, rollout_node)][1] += scores[1]            self.total_reward[(rollout_action, rollout_node)][2] += scores[2]        self.update(rollout_action, rollout_node,scores)def ghostmove_action(index,state, prob_attack=0.8, prob_scaredFlee=0.8 ):    try:        ghostState = state.getGhostState(index)        legalActions = state.getLegalActions(index)        if legalActions==[]:            return None        pos = state.getGhostPosition(index)        isScared = ghostState.scaredTimer > 0        speed = 1        if isScared: speed = 0.5        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]        pacmanPosition = state.getPacmanPosition()        # Select best actions given the state        distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]        shortest_distance = min(distancesToPacman)        longest_distance = max(distancesToPacman)        if isScared:            bestindex = [i for i in range(len(distancesToPacman)) if distancesToPacman[i] == longest_distance]        else:            bestindex = [i for i in range(len(distancesToPacman)) if distancesToPacman[i] == shortest_distance]    except:        print 'I am here'    return legalActions[bestindex[0]]