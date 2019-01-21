from dataAccess import DataAccess
from Model import LSTMPredictor
import torch
import torch.nn as nn
import torch.optim as optim

# set up objects
dataObject = DataAccess('data_small.txt') # read the data
model = LSTMPredictor(128,128, dataObject.vocab_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
model_path = 'my_sp_model'


for epoch in range(100000):
    for batch in range(dataObject.data_size):
        model.zero_grad()
        model.hidden = model.init_hidden() #
        chars = dataObject.getNextChar()
        tag_scores = model(chars[0])
        target = chars[1]
        loss = loss_function(tag_scores, target)
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0: # draw a sample from time to time which remains useless
        with torch.no_grad():
            print("Epoch: ", epoch)
            oldInputs = torch.tensor([dataObject.getIx('E')])
            print('E')
            for x in range(6):
                scores = model(oldInputs)
                nextChar = dataObject.getChar(scores[0].max(0)[1].item())
                print(nextChar)
                oldInputs = torch.tensor([dataObject.getIx(nextChar)])
        torch.save(model.state_dict(), model_path)




