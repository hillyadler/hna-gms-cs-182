# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 22:09:47 2016

@author: Geoffrey
"""

agents = gameState.getNumAgents()
            v = float('inf')
            index = index + 1
            if index == agents:
                for action in gameState.getLegalActions(index):
                    v = min(v, MAX_VALUE(gameState.generateSuccessor(index, action), d))
                return v
            elif index < agents:
                for action in gameState.getLegalActions(index):
                    v = min(v, MIN_VALUE(gameState.generateSuccessor(index, action), index, d))
                return v
        
        
            if d == self.depth:
                return self.evaluationFunction(state)
            v = float('inf')
            for action in state.getLegalActions(1):
                v = max(v, MAX_VALUE(state.generateSuccessor(0, action), d+1))
            return v 