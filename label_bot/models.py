# Github-Issues-Classifier
# Copyright(C) 2020 Georgios (Giorgos) Karantonis
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from simpletransformers.classification import MultiLabelClassificationModel


_HERE_DIR = os.path.dirname(os.path.abspath(__file__))



class ScoresHead(nn.Module):
    def __init__(self, custom_head=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam
        
        head = custom_head if custom_head else self.default_head()
        self.model = nn.Sequential(*head).to(self.device)
    
    
    def default_head(self):
        return [
            nn.Linear(9, 100), 
            nn.LeakyReLU(.2), 
            nn.BatchNorm1d(100), 
            nn.Linear(100, 3)
        ]
        
        
    def forward(self, titles, bodies, combined, train=True):
        if train: 
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        x = torch.cat((titles, bodies, combined), dim=1).to(self.device)
        x = self.model(x)
        x = torch.sigmoid(x)
        
        return x
    
    
    def fit(self, 
            titles, 
            bodies, 
            combined, 
            labels, 
            validation=True, 
            val_titles=None, 
            val_bodies=None, 
            val_combined=None, 
            val_labels=None, 
            epochs=35, 
            lr=5e-3, 
            verbose=True):
        
        losses = {
            'train' : [], 
            'val' : []
        }
            
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        
        labels = np.array(labels)
        
        for epoch in range(epochs):
            indices = np.arange(titles.shape[0])
            np.random.shuffle(indices)

            titles = titles[indices]
            bodies = bodies[indices]
            combined = combined[indices]
            labels = labels[indices]
            
            titles_tensor = torch.from_numpy(titles).to(self.device)
            bodies_tensor = torch.from_numpy(bodies).to(self.device)
            combined_tensor = torch.from_numpy(combined).to(self.device)
            
            labels = torch.FloatTensor(list(map(list, labels))).to(self.device)
            
            optimizer.zero_grad()
            outputs = self.forward(titles_tensor, bodies_tensor, combined_tensor)
                        
            loss = self.loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            losses['train'].append(loss.item())
            if validation:
                val_loss = self.evaluate(val_titles, val_bodies, val_combined, val_labels)
                losses['val'].append(val_loss)
            
            if verbose:
                print(f'Epoch: {epoch+1}')
                print(f'Training loss: {loss.item()}')
                if validation:
                    print(f'Validation loss: {val_loss}')
                    print('Validation Accuracy: ', accuracy(np.where(outputs.detach().cpu().numpy() > .5, 1, 0), 
                                                            labels.detach().cpu().numpy()))
                print()
                        
        return outputs.detach().cpu().numpy(), losses
    
    
    def evaluate(self, titles, bodies, combined, labels):
        losses = []
        
        titles_tensor = torch.from_numpy(titles).to(self.device)
        bodies_tensor = torch.from_numpy(bodies).to(self.device)
        combined_tensor = torch.from_numpy(combined).to(self.device)
        
        labels = torch.FloatTensor(list(map(list, labels))).to(self.device)
        
        outputs = self.forward(titles_tensor, bodies_tensor, combined_tensor, train=False)
        
        loss = self.loss(outputs, labels)
        
        return loss.detach().cpu().item()
    
    
    def predict(self, titles, bodies, combined):
        titles_tensor = torch.from_numpy(titles).to(self.device)
        bodies_tensor = torch.from_numpy(bodies).to(self.device)
        combined_tensor = torch.from_numpy(combined).to(self.device)
        
        predictions = self.forward(titles_tensor, bodies_tensor, combined_tensor, train=False)
        
        return predictions.detach().cpu().numpy()



class Bot:
    def __init__(self, use_head=True, model_name='roberta', model_path=os.path.join(_HERE_DIR, 'models', 'classification', 'roberta-base')):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.use_head = use_head
        self.model_path = model_path

        self.classifier = self.load_pretrained_model(model_name, model_path)


    def load_pretrained_model(self, name='roberta', from_path='roberta-base'):
        if from_path.endswith('/'):
            from_path = from_path[:-1]
            
        with open(f'{from_path}/model_args.json') as f:
            model_args = json.load(f)
            
        return MultiLabelClassificationModel(name, 
                                             from_path, 
                                             num_labels=3, 
                                             args=model_args, 
                                             use_cuda=self.device==torch.device('cuda'))


    def predict(self, title, body):
        if isinstance(title, str):
            title = pd.DataFrame([title])
        if isinstance(body, str):
            body = pd.DataFrame([body])

        if not isinstance(title, pd.DataFrame):
            title = pd.DataFrame(title)
        if not isinstance(body, pd.DataFrame):
            body = pd.DataFrame(body)

        title.columns = ['text']
        body.columns = ['text']
            
        if self.use_head:
            _, titles_scores = self.classifier.predict(title['text'])
            _, bodies_scores = self.classifier.predict(body['text'])
            _, scores = self.classifier.predict(title['text'] + ' ' + body['text'])

            head = ScoresHead()
            if self.device == 'cuda':    
                head.load_state_dict(torch.load(os.path.join(self.model_path, 'scores_head.pt')))
            else:
                head.load_state_dict(torch.load(os.path.join(self.model_path, 'scores_head.pt'), map_location=torch.device('cpu')))
            
            scores = head.predict(titles_scores, bodies_scores, scores)
        else:
            df = pd.DataFrame(title['text'] + ' ' + body['text'])
            df.columns = ['text']
            
            _, scores = self.classifier.predict(df['text'])

        
        return scores