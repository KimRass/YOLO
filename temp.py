import torch

# Define the size of your tensor
batch_size = 3
tensor_size = (batch_size, 5, 5)  # Example size, you can adjust according to your needs

# Define indices where you want to put 1 for each batch
indices = [
    [(1, 0), (3, 4)],  # Indices for the first batch
    [(0, 2), (4, 3)],  # Indices for the second batch
    [(2, 1), (3, 0)]   # Indices for the third batch
] 

# Initialize tensor with zeros
tensor = torch.zeros(tensor_size)

# Convert indices to tensor
indices_tensor = torch.tensor(indices)
indices_tensor.shape
indices_tensor[0]

# Create a mask tensor with shape (batch_size, 5, 5) and fill the specified indices with 1
mask = torch.zeros(tensor_size)
mask.scatter_(1, indices_tensor[:, :, 0].unsqueeze(2), 1)
mask.scatter_(2, indices_tensor[:, :, 1].unsqueeze(1), 1)

# Apply the mask to the tensor
tensor += mask

print(tensor)
