import torch

class DataAccess(object):
    def __init__(self, data_path):
        #self.data = io.open('data.txt', 'r', encoding="utf-8")
        self.data = open( data_path , 'r').read()
        self.chars = list(set(self.data))
        self.data_size, self.vocab_size = len(self.data), len(self.chars)
        print('data has %d characters, %d unique.' % (self.data_size, self.vocab_size))
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.offset = 0
    def getNextChar(self):
        if self.offset > (len(self.data) - 2):
            self.offset = 0
        chars = [
            torch.tensor(self.char_to_ix[self.data[self.offset]]),
            torch.tensor([self.char_to_ix[self.data[self.offset + 1]]]) # -1 + 1
            ]
        self.offset = self.offset + 1
        return chars

    def getIx(self, char):
        return self.char_to_ix[char]

    def getChar(self, ix ):
        return self.ix_to_char[ix]

    def getOneHot(self, index):
        oneHot = torch.zeros(1,self.vocab_size, dtype = torch.long)
        oneHot[0,index] = 1
        return oneHot