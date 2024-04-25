import torch

data = torch.load('/home/kl23/code/ditl/ptfile/pku/output/04_21_01/Epoch_383/model_output_test.pt')
for i in range(data.shape[0]):
    data1 = data[i]
    print(f"{i}")
    for j in range(data1.shape[0]):
        print(data1[j])
        if j>=10:
            break
    print("")
    if i>=50:
        break
