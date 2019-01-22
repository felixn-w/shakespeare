from dataAccess import DataAccess
from Model import LSTMPredictor
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    dataObject = DataAccess('data_small.txt')  # read the data
    model = LSTMPredictor(128, 128, dataObject.vocab_size)
    #loss_function = nn.NLLLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    model_path = 'my_sp_model'
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        oldInputs = torch.tensor([dataObject.getIx('E')])
        print('E')
        for x in range(30):
            scores = model(oldInputs)
            nextChar = dataObject.getChar(scores[0].max(0)[1].item())
            print(nextChar)
            oldInputs = torch.tensor([dataObject.getIx(nextChar)])