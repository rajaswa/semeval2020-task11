# IMPORTS
import torch
from torch import nn

# MODEL
class Task1Model(nn.Module):
    def __init__(self, input_size):
        super(Task1Model, self).__init__()
        self.input_size = input_size

        self.title_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 16),
            nn.ReLU(),
        )
        self.line_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 16),
            nn.ReLU(),
        )
        self.token_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 16),
            nn.ReLU(),
        )

        self.title_regressor = nn.Sequential(nn.Dropout(p=0.05), nn.Linear(16, 1))
        self.line_classifier = nn.Sequential(nn.Dropout(p=0.05), nn.Linear(32, 1))
        self.token_classifier = nn.Sequential(nn.Dropout(p=0.05), nn.Linear(48, 1))

    def forward(self, title, line, token, predict=False):
        title_embed = self.title_layer(title)
        line_embed = self.line_layer(line)
        token_embed = self.token_layer(token)

        if predict == True:
            line_combined = torch.cat((title_embed, line_embed), dim=0)
        else:
            line_combined = torch.cat((title_embed, line_embed), dim=1)

        if predict == True:
            token_combined = torch.cat((title_embed, line_embed, token_embed), dim=0)
        else:
            token_combined = torch.cat((title_embed, line_embed, token_embed), dim=1)

        title_out = self.title_regressor(title_embed)
        line_out = self.line_classifier(line_combined)
        token_out = self.token_classifier(token_combined)

        return title_out, line_out, token_out
