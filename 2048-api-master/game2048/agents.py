import numpy as np
import random
from game2048.features import Feature
  
 # 定义从一个数字列表中以一定的概率取出对应区间中数字的函数
def get_number_by_pro(number_list, pro_list):
    """
    :param number_list:数字列表
    :param pro_list:数字对应的概率列表
    :return:按概率从数字列表中抽取的数字
    """
    # 用均匀分布中的样本值来模拟概率
    x = random.uniform(0, 1)
    # 累积概率
    cum_pro = 0.0
    # 将可迭代对象打包成元组列表
    for number, number_pro in zip(number_list, pro_list):
        cum_pro += number_pro
        if x < cum_pro:
            # 返回值
            return number

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        #f = Feature()
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1

        #    if n_iter == 1:
        #        array = np.reshape(self.game.board,(1,16))
        #    else:
        #        array = np.insert(array,n_iter-1,values=np.reshape(self.game.board,(1,16)),axis=0)
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
        #    for i in range(4):
        #        print(f.feature(_next(self.game.board,i)))
        #f = Feature()
        #t.write(array,self.game.score)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def play(self, max_iter=np.inf, verbose=False):
        f = Feature()
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
        #    if n_iter == 1:
        #        array = np.reshape(self.game.board,(1,16))
        #    else:
        #        array = np.insert(array,n_iter-1,values=np.reshape(self.game.board,(1,16)),axis=0)
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
            '''
            next_feature = []
            for i in range(4):
                next_feature = np.append(next_feature,f.feature(_next(self.game.board,i)))
            download = np.load('train_data_fit.npz')
            np.savez('train_data_fit.npz',data = np.append(download['data'],[next_feature],axis = 0),label = np.append(download['label'],direction))
            '''
            if np.max(self.game.board) <= 64:
                download2 = np.load('epoch64_fit.npz')
                np.savez('epoch64_fit.npz',data = np.append(download2['data'],f.simplify(self.game.board),axis = 0),label = np.append(download2['label'],direction))
            elif np.max(self.game.board) == 128:
                download2 = np.load('epoch128_fit.npz')
                np.savez('epoch128_fit.npz',data = np.append(download2['data'],f.simplify(self.game.board),axis = 0),label = np.append(download2['label'],direction))
            elif np.max(self.game.board) == 256:
                download2 = np.load('epoch256_fit.npz')
                np.savez('epoch256_fit.npz',data = np.append(download2['data'],f.simplify(self.game.board),axis = 0),label = np.append(download2['label'],direction))
            elif np.max(self.game.board) == 512:
                download2 = np.load('epoch512_fit.npz')
                np.savez('epoch512_fit.npz',data = np.append(download2['data'],f.simplify(self.game.board),axis = 0),label = np.append(download2['label'],direction))
            elif np.max(self.game.board) == 1024:
                download2 = np.load('epoch1024_fit.npz')
                np.savez('epoch1024_fit.npz',data = np.append(download2['data'],f.simplify(self.game.board),axis = 0),label = np.append(download2['label'],direction))
        #f = Feature()
        #t.write(array,self.game.score)
        
    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class RandomAgent2(Agent):

    def step(self):
        direction = np.random.randint(0, 2)
        return direction
    
class YourOwnAgent(Agent):
        
    def __init__(self, game, display=None):
        super().__init__(game, display)

        from game2048.fit_data import TrainKeras as Train
        
        self.t1 = Train()
        self.t1.fit(score=64)
        self.t2 = Train()
        self.t2.fit(score=128)
        self.t3 = Train()
        self.t3.fit(score=256)
        self.t4 = Train()
        self.t4.fit(score=512)
        self.t5 = Train()
        self.t5.fit(score=1024)    
        '''
        self.t = Train()
        self.t.fit(None)
        '''
    
    def step(self):
        
        f = Feature()
        '''
        #next_feature = []
        #for i in range(4):
        #    next_feature = np.append(next_feature,f.feature(_next(self.game.board,i)))

        #epsilon = 0.1
        #if_random = random.uniform(0, 1)
        #board = self.game.board
        #Q = [None,None,None,None]
        #sum_weight = 0
        #for i in range(4):
        #    Q[i] = self.weight(_next(board,i))
        
        #if if_random > epsilon:
        #direction = Q.index(max(Q))
        #else:
        #    direction = get_number_by_pro(number_list=[0,1,2,3], pro_list=[1/4,1/4,1/4,1/4])
        
        return direction[0]
        '''
        table = {64:self.t1,128:self.t2,256:self.t3,512:self.t4,1024:self.t5}
        score = 64 if np.max(self.game.board) <= 64 else np.max(self.game.board)
        direction = table[score].board_to_move(f.simplify(self.game.board))
        #direction = self.t.board_to_move(f.simplify(self.game.board))
        #direction = t[table[np.max(self.game.board)]].board_to_move(self.game.board)
        repeat = 0

        while (_next(self.game.board,direction) == self.game.board).all():
            direction = (direction+1)%4
            repeat +=1
            if repeat == 4:
                break

        return direction
    
    def weight(self,data):
        f = Feature()
        features = np.empty([1,3])
        features[0,0:3] = f.feature(data)
        w = np.array([[-1],[2],[1]])
        Q = np.dot(features,w)
        return Q[0,0]
            
def _next(data,direction):
    '''
    direction:
        0: left
        1: down
        2: right
        3: up
    '''
    # treat all direction as left (by rotation)
    board_to_left = np.rot90(data, -direction)
    for row in range(data.shape[0]):
        core = _merge(board_to_left[row])
        board_to_left[row, :len(core)] = core
        board_to_left[row, len(core):] = 0

    # rotation to the original
    board = np.rot90(board_to_left, direction)
    return board
    
def _merge(row):
    '''merge the row, there may be some improvement'''
    non_zero = row[row != 0]  # remove zeros
    core = [None]
    merge_score = 0
    for elem in non_zero:
        if core[-1] is None:
            core[-1] = elem
        elif core[-1] == elem:
            core[-1] = 2 * elem
            merge_score += core[-1]
            core.append(None)
        else:
            core.append(elem)
    if core[-1] is None:
        core.pop()
    return core
