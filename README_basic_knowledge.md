# Basic knowledge

Before we understand all the readme file below there are some basic knowledge about the libraries as well as some deep learning knowledges that we need to know.


## `torch.nn.Linear(in_features, out_features, bias = True, device=none, dtype=none)`

So basically we apply the linear transformation for the incomming data $x$ to return an output data $y = xA^T + b$

with:
- $x$ of size `in_features`
- $y$ of size `out_features`
- $b$ of size `out_features`

This will result in generating a matrix $A$ of size (`in_features` x `out_features`)

<img src="./README images/torch_nn_Linear size.svg">

Therefore, `torch.nn.Linear(in_features, out_features, bias = True, device=none, dtype=none)` will create a weights matrix $A$ of size (`in_features` x `out_features`)


## `torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)`



<img src="./README images/torch_nn_Conv2d.svg">