# IMPORTS
import torch
from torch import nn

# MODEL
class Task1Model(nn.Module):
    def __init__(self, input_size):
        super(Task1Model, self).__init__()
        self.input_size = input_size
        self.title_layer = nn.Sequential(
            nn.Linear(input_size, 32), nn.ReLU(), nn.Dropout(p=0.1)
        )
        self.line_layer = nn.Sequential(
            nn.Linear(input_size + 32, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(64, 1),
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, title, line, token, predict=False):
        title_out = self.title_layer(title)

        if predict == True:
            line_embed = torch.cat((title_out, line), dim=0)
        else:
            line_embed = torch.cat((title_out, line), dim=1)

        line_logit = self.line_layer(line_embed)
        line_tanh = self.tanh(line_logit)
        out_token = self.sigmoid(self.classifier(token) * line_tanh)
        out_line = self.sigmoid(line_logit)

        return out_line, out_token
