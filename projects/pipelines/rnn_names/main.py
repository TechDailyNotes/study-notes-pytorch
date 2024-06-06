import torch
import torch.nn as nn
import torch.optim as optim
from utils import N_LETTERS, load_data, random_training_example

# Step 1: Hyperparameter Setup
# Step 1.1: Device Params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1.2: Data Params

# Step 1.3: Model Params
hidden_size = 128
lr = 0.05

# Step 1.4: Training Params
num_epochs = 10

# Step 1.5: Param Tuning

# Step 2: Data Setup
# Step 2.1: Data Source
category_lines, all_categories = load_data()

# Step 2.2: Data Preprocessing

# Step 2.3: Data Formatting

# Step 2.4: Data Features
n_categories = len(all_categories)

# Step 2.5: Data Accessories


# Step 3: Model Setup
# Step 3.1: Architecture Setup
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.hidden_size = hidden_size

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)
        out = self.softmax(self.i2o(combined))
        hidden = self.i2h(combined)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


model = RNN(N_LETTERS, hidden_size, n_categories).to(device)

# Step 3.2: Loss Setup
criterion = nn.NLLLoss()

# Step 3.3: Optimizer Setup
optimizer = optim.SGD(model.parameters(), lr=lr)

# Step 3.4: Scheduler Setup

# Step 4: Training Loop
for epoch in range(num_epochs):
    category, line, category_tensor, line_tensor = \
        random_training_example(category_lines, all_categories)
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    hidden_tensor = model.init_hidden()

    # print(f"line_tensor.shape = {line_tensor.shape}")
    # print(f"hidden_tensor.shape = {hidden_tensor.shape}")

    for token in range(line_tensor.size(0)):
        output_tensor, hidden_tensor = model(line_tensor[token], hidden_tensor)
    loss = criterion(output_tensor, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"epoch {epoch+1}/{num_epochs}, loss {loss.item()}")

# Step 5: Result Test
