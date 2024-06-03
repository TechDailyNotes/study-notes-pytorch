import matplotlib.pyplot as plt
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
num_epochs = 10_000
plot_steps = num_epochs // 10
print_steps = num_epochs // 100

# Step 1.5: Param Tuning

# Step 2: Data Setup
# Step 2.1: Data Source
category_lines, all_categories = load_data()

# Step 2.2: Data Preprocessing

# Step 2.3: Data Formatting

# Step 2.4: Data Features
print()

num_categories = len(all_categories)
print(f"len(all_categories) = {len(all_categories)}")

print()

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
        output = self.softmax(self.i2o(combined))
        hidden = self.i2h(combined)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


model = RNN(N_LETTERS, hidden_size, num_categories).to(device)

# Step 3.2: Loss Setup
criterion = nn.NLLLoss()

# Step 3.3: Optimizer Setup
optimizer = optim.SGD(model.parameters(), lr=lr)

# Step 3.4: Scheduler Setup

# Step 4: Training Loop
# input_tensor = letter_to_tensor('A')
# hidden_tensor = model.init_hidden()
# output, hidden = model(input_tensor, hidden_tensor)

# print(f"output.size() = {output.size()}")
# print(f"hidden.size() = {hidden.size()}")


def category_from_output(output):
    output_index = torch.argmax(output, dim=1).item()
    return all_categories[output_index]


# print(f"category_from_output(output) = {category_from_output(output)}")

plot_epoch_loss = 0.0
print_epoch_loss = 0.0
all_losses = []

for epoch in range(num_epochs):
    category, line, category_tensor, line_tensor = \
        random_training_example(category_lines, all_categories)
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    hidden_tensor = model.init_hidden().to(device)

    for token in range(line_tensor.size(0)):
        output, hidden = model(line_tensor[token], hidden_tensor)

    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plot_epoch_loss += loss.item()
    print_epoch_loss += loss.item()

    if (epoch + 1) % plot_steps == 0:
        all_losses.append(plot_epoch_loss / plot_steps)
        plot_epoch_loss = 0.0

    if (epoch + 1) % print_steps == 0:
        print_epoch_loss /= print_steps
        print(f"epoch {epoch + 1}/{num_epochs}, loss {print_epoch_loss}")
        print_epoch_loss = 0.0

plt.plot(all_losses)
plt.show()

# Step 5: Result Test
