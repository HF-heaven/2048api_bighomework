import numpy as np
import math

class Feature:
        
    def feature(self,board):
        board,merge_score = board
        monotonicity = 0
        flatness = 0
        for i in range(np.size(board,0)):
            list_monotonicity,list_flatness = self.list_monotonicity(board[i,:])
            monotonicity += list_monotonicity
            flatness += list_flatness
            list_monotonicity,list_flatness = self.list_monotonicity(board[:,i])
            monotonicity += list_monotonicity
            flatness += list_flatness
        blank_tile =  np.size(board[board == 0])
        return monotonicity/10,flatness,blank_tile,merge_score
        
        
    def list_monotonicity(self,array):
        size = np.size(array)
        #if size == 0:
        #    return 0,0
        delta = np.empty(size - 1)
        flatness = 0
        for i in range(size - 1):
            delta[i] = abs(array[i + 1] - array[i])
            if delta[i] == 0:
                flatness += 1
        monotonicity = np.sum(delta) - np.max(array) + np.min(array)
        return monotonicity,flatness
                
    def simplify(self,board):
        board = np.reshape(board,[1,16])
        table = {2**i:i for i in range(1,11)}
        table[0] = 0
        for i in range(16):
            board[0,i] = table[board[0,i]]
        return board