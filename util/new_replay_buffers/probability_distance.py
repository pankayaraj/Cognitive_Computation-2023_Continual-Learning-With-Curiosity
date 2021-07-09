import torch
import numpy as np

def bhattacharyya_distance(mu1, mu2, std1, std2):


    if type(mu1) != torch.Tensor:
        mu1 = torch.Tensor(mu1)
        std1 = torch.Tensor(std1)
    if type(mu2) != torch.Tensor:
        mu2 = torch.Tensor(mu2)
        std2 = torch.Tensor(std2)


    mean_diff = mu1-mu2
    var1 = torch.exp(std1)
    var2 = torch.exp(std2)


    n = mu1.shape[0]
    avg_var = (var1 + var2)/2
    avg_var_inv = torch.ones(avg_var.shape)

    avg_det = 1
    det1 = 1
    det2 = 1

    for i in range(n):
        avg_var_inv[i] = 1/avg_var[i]

        avg_det *= avg_var[i]
        det1 *= var1[i]
        det2 *= var2[i]

    #eye = torch.eye(n)
    #avg_cov_inv = eye*avg_std_inv


    term1 = torch.matmul(mean_diff*avg_var_inv, mean_diff)/8
    term2 = np.log(avg_det/np.sqrt(det1*det2))/2

    return term1 + term2




#a = torch.Tensor([10,3,3])
#b = torch.Tensor([1,2,3])
#d = bhattacharyya_distance(b, a, b, b)
#print(d)