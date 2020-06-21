import numpy as np
import math

class Train:
        
    def write(self,array,score):
        a = np.load('train_data.npy')
        for i in range(np.size(array,0)):
            array_rot = self.trans(array[i])
            for j in range(np.size(array_rot,0)):
                axis0 = self.find(a,array_rot[j,:],0,np.size(a,0)-1)
                if not axis0[0]:
                    b = [score,1]
                    a = np.insert(a,axis0[1],values=[np.append(array_rot[j,:],[score,1])],axis=0)
                else:
                    weight = a[axis0[1],16]
                    c = a[axis0[1],17]
                    a[axis0[1],16] = (weight * c + score)/(c + 1)
                    a[axis0[1],17] += 1
        np.save('train_data.npy',a)
        
    def find(self,array,board,low,high):
        if board.ndim == 2:
            board = board[0]
        axis0_number = high - low
        if axis0_number == 0 or axis0_number == 1:
            if (array[low,0:16] == board).all():
                return [1,low]
            elif (array[high,0:16] == board).all():
                return [1,high]
            else:
                return [0,high]
        else:
            axis0_new = low + math.floor(axis0_number/2)
            i = 0
            while array[axis0_new,i] - board[i] == 0:
                i += 1
                if i == 16:
                    return [1,axis0_new]
            if array[axis0_new,i] - board[i] < 0:
                return self.find(array,board,low,axis0_new)
            else:
                return self.find(array,board,axis0_new,high)
            
                
    def board_weight(self,data):
        a = np.load('train_data.npy')
        axis0 = self.find(a,np.reshape(data,(1,16)),0,np.size(a,0)-1)
        if not axis0[0]:
            return None
        else:
            return a[axis0[1],16]
        
    def trans(self,array):
        a = np.reshape(array,(4,4))
        board = np.zeros((8,16))
        for i in range(0,4):
            board[i] = np.reshape(np.rot90(a,i),(1,16))
        a = a.T
        for i in range(0,4):
            board[i+4] = np.reshape(np.rot90(a,i),(1,16))
        return board