from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    # edit
    batch_size = 32

    
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    
    # edit
    #loss_function = torch.nn.MSELoss()
    loss_function=torch.nn.BCEWithLogitsLoss()

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader):
            x, y = sample['input'], sample['label']
          
            
            # compute the model output
            yhat = model(x)
            loss=loss_function(yhat.reshape(-1,1),y.reshape(-1,1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # # calculate loss
            min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
            losses.append(min_loss)
            print(min_loss)
            

    torch.save(model.state_dict(), 'saved/saved_model.pkl')
    # pickle.dump(model, open('saved/saved_model.pkl', "wb"))
    # torch.save(model, 'saved/saved_model.pkl')
    # model = Action_Conditioned_FF()
    # model.load_state_dict(torch.load('saved/saved_model.pkl'))
    # print(model.eval())
if __name__ == '__main__':
    # edit
    no_epochs = 1000
    train_model(no_epochs)