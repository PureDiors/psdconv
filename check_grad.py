import torch
import torch.autograd.gradcheck as gradcheck
from modules.ps_deform_conv import PSDeformConv, _PSDeformConv, PSDeformConvPack




conv = PSDeformConvPack(1,1,3,1,1,dilation=2).cuda()
inp = torch.randn([1,1,3,3]).cuda() 
inp.requires_grad = True
print(conv)

test = gradcheck(conv, (inp), eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True)
print(test)

