import torch

def matching(feature1, feature2):
    key1, array1 = next(iter(feature1.items()))
    key2, array2 = next(iter(feature2.items()))
    
    tensor1 = torch.tensor(array1[:, :, 1] / 255.0).float()
    tensor2 = torch.tensor(array2[:, :, 1] / 255.0).float()

    print(tensor1)
    print(tensor2)
    # Dot product (for 1D tensors, matmul defaults to dot product)
    correlation = torch.matmul(tensor1, tensor2)

    return correlation