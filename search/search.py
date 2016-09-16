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
import copy
import sets

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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    fringe = util.Stack()
    fringe.push(problem.getStartState())

    actions = util.Stack()
    actions.push([])
    closed = []
    while not fringe.isEmpty():
        root = fringe.pop()
        old_act = actions.pop()
        if problem.isGoalState(root):
            return old_act
        if root not in closed:
            closed.append(root)
            for successor in problem.getSuccessors(root):
                fringe.push(successor[0])
                actions.push(old_act + [successor[1]])
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    fringe = util.Queue()
    fringe.push(problem.getStartState())

    actions = util.Queue()
    actions.push([])
    closed = []
    while not fringe.isEmpty():
        root = fringe.pop()
        old_act = actions.pop()
        if problem.isGoalState(root):
            return old_act
        if root not in closed:
            closed.append(root)
            for successor in problem.getSuccessors(root):
                fringe.push(successor[0])
                actions.push(old_act + [successor[1]])



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    fringe = util.PriorityQueue()
    fringe.push(problem.getStartState(), 0)

    actions = util.PriorityQueue()
    actions.push([], 0)
    costs = util.PriorityQueue()
    costs.push(0,0)
    closed = sets.Set()
    while not fringe.isEmpty():
        root = fringe.pop()
        old_act = actions.pop()
        old_cost = costs.pop()
        if problem.isGoalState(root):
            return old_act
        if root not in closed:
            closed.add(root)
            for successor in problem.getSuccessors(root):
                fringe.push(successor[0], old_cost + successor[2])
                actions.push((old_act + [successor[1]]), old_cost + successor[2])
                costs.push(old_cost + successor[2],old_cost + successor[2])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    fringe = util.PriorityQueue()
    fringe.push(problem.getStartState(), 0)

    actions = util.PriorityQueue()
    actions.push([], 0)
    costs = util.PriorityQueue()
    costs.push(0,0)
    closed = [] #sets.Set()
    while not fringe.isEmpty():
        root = fringe.pop()
        old_act = actions.pop()
        old_cost = costs.pop()
        if problem.isGoalState(root):
            return old_act
        if root not in closed:
            closed.append(root) #closed.add(root)
            for successor in problem.getSuccessors(root):
                fringe.push(successor[0], old_cost + successor[2]+(heuristic(successor[0],problem)))
                actions.push((old_act + [successor[1]]), old_cost + successor[2]+(heuristic(successor[0],problem)))
                costs.push(old_cost + successor[2], old_cost+successor[2]+(heuristic(successor[0],problem)))



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
