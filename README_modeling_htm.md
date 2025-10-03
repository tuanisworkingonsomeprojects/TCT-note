
# Documentation for modeling_htm.py


## `class Embedding:`

The original image is split into smaller patches of size that is defined under config.patches in each ***getter*** methods in the `configs.py` file.

e.g.
```python
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config
```

We can see in the example that the patches size is (16 x 16), which also means that the original image is just a sequence of (16 x 16) images.

Each patch is then going through an Embedding process to convert from a 3 channels tensor to a 2D matrix.





That is a logical thinking process, when we implement it through code we will try to apply the 2D Convolution layer that has the 
- `kernel_size` of `patch_size`
- `stride` of `patch_size` to ensure none overlapping rule 
- `out_channels` size of `config.hidden_size` as below




<img src="./README images/Embedding Conv2d size.svg">


<img src="./README images/hidden_states_size.svg">
But what is config.hidden_size means?








## `class Attention:`
### `def __init__(self, config, vis):`

When you read this constructor method you will see the size configuration of the Attention layer. However, it is unclear without any visualization.

You also may see the lines of code that could a bit confusing.

```python
self.query = Linear(config.hidden_size, self.all_head_size)
self.key = Linear(config.hidden_size, self.all_head_size)
self.value = Linear(config.hidden_size, self.all_head_size)
```

You may ask:
- This is a multihead attention model isn't it should initialize the size of Query, Key, and Value for each head?

However, this is an implementation technique that initializes the bigger block of tensor that is **logically** a combination of different heads of the attention layer.

<img src="./README images/Attention size.svg">


## `def transpose_for_scores(self, x):`

<img src="./README images/Attention transpose for scores.svg">
 
## `def forward(self, hidden_states, crt_local_mod=None, query_mod=None, key_mod=None, top_n=None, device='cuda'):`

Firstly, we just pass the hidden_states through the 3 linear layers (Query, Key, Value) to obtain the  `mixed_query_layer`, `mixed_key_layer` and `mixed_value_layer` respectively.

```python
mixed_query_layer = self.query(hidden_states)
mixed_key_layer = self.key(hidden_states)
mixed_value_layer = self.value(hidden_states)
```

After that, we transform them 

- from `(batch_size, seq_len, all_head_size)` 

- to `(batch_size, num_attention_head, seq_len, attention_head_size)` 

(refer back to transpose_for_scores for more info)

With:

 - `all_head_size` = `num_attention_head` x `attention_head_size`
 - `seq_len` = `number_of_patches` + 1

```python
query_layer = self.transpose_for_scores(mixed_query_layer) # 1 * 12 * search_patches * 64
key_layer = self.transpose_for_scores(mixed_key_layer)
value_layer = self.transpose_for_scores(mixed_value_layer)
```

The transformation is nesscessary for accurate dimension when calculating the attention score.

Then we just apply the attention score, and attention probability formula.

```python
attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))                     # 1 * 12 * (search_patches+1) * (search_patches+1)
attention_scores = attention_scores / math.sqrt(self.attention_head_size)
attention_probs = self.softmax(attention_scores)    
```

Then, we will find the similarity beween the image query matrix with the search query matrix using `dot product` to approximate the cosine similarity. After that, we use softmax to normalized the result going from `0` to `1`. We can also have another point of view is that we will apply 1 more attention formula with 
- the QUERY matrix as `query_mod`.
- the KEY matrix as `query_layer`.
- the VALUE matrix as `crt_local_mod`

```python
if query_mod is not None:
    query_relevance = torch.matmul(query_mod[:, 1:, :], query_layer[:, :, 1:, :].transpose(-1, -2))       # 1 * 12 * target_patches * search_patches
    query_relevance = self.softmax(query_relevance / math.sqrt(query_mod.size(-1)))
    if crt_local_mod is not None:
        query_relevance = query_relevance * crt_local_mod  
```

<i>Notation:</i>
- `L_t`: target seq_len ([CLS excluded]) = `target_seq_len - 1`
    - Since we select `query_mod[:, 1:, :]`
    - This can also be understood to be the number of target patches (refer to Embedding class)
- `L_s`: search seq_len ([CLS excluded]) = `search_seq_len - 1`
    - Since we select `query_layer[:, :, 1:, :]`
    - This can also be understood to be the number of search patches (refer to Embedding class)
- `d_h`: `attention_head_size`
- `H`: `all_head_size` = `num_of_heads` x `d_h`
- `B`: batch_size

<br>
<i>Note:</i>

- `L_s` and `L_t` has `[CLS]` token excluded
- `attention_probs` size:
    - `(B, num_of_heads, L_s + 1, L_s + 1)`
- `query_mod[:, 1:, :]` size:
    - `(B, L_t, H)`

- `query_layer[:, :, 1:, :]` size:
    - `(B, num_of_heads, L_s, d_h)`

- `query_layer[:, :, 1:, :].transpose(-1, -2)` size:
    - `(B, num_of_heads, d_h, L_s)`


- `torch.matmul` only treat the last 2 dimensions for matrix multiplication, the leading dimensions are treated as batch dimensions. However, there is a mismatch in dimensions between `query_mod` and `query_layer`, here are the steps it takes to make the math physible.
    - It first tries to convert `query_mod` size from `(B, L_t, H)` to `(B, 1, L_t, H)`
    - Then, it find out the 2 last dimensions `(L_t, H)`, does not match with `(d_h, L_s)` for multiplication. 
    - It then try to match up the mismatch dimension by convert it to `(L_t, d_h)`, the remaining dimension `H / d_h` will be broadcast to the batch dimension. 
    - Since `H / d_h == num_of_heads`, that match with the broadcast batch dimension, change from `(B, 1)` to `(B, nums_of_heads)`
    - The whole process has changed `query_mod` dimension from `(B, L_t, H)` to `(B, num_of_heads, L_t, d_h)`

- `key_relevance` size:
    - Result from the last 2 dimensions: `(L_t, d_h)` x `(d_h, L_s)` => `(L_t, L_s)`
    - Result size: `(B, num_of_heads, L_t, L_s)`


After that we perform 2 argument sort on the `L_s` dimension (apply `torch.argsort` on each row)
- 1st argsort will be the index to pick in the orginal tensor to make the sorted tensor.
- 2nd argsort will be the index in the sorted list of the current element in the orginal list.

e.g.
```python
a = torch.tensor([1, 5, 2, 3, 7, 4, 9, 10])

b = torch.argsort(a)
>>> tensor([0, 2, 3, 5, 1, 4, 6, 7])

c = torch.argsort(b)
>>> tensor([0, 4, 1, 2, 5, 3, 6, 7])

a[b]
>>> tensor([ 1,  2,  3,  4,  5,  7,  9, 10])
```
We can see that element `5` in list `a` has index `4` in the sorted list `a[b]`

Then we will generate a tensor of boolean if the value is the `top_n` values.

<i>Note:</i>
- In the sorted list, the HIGHER the `index` of the value in the sorted list the HIGHER the `value`.
- Therefore, to find the top n value, we just need to find the top n highest index by finding the last n index (i.e. `True if index >= len(list) - top_n`)

```python
keep = torch.argsort(torch.argsort(query_relevance, dim=-1)) >= (query_relevance.shape[-1] - top_n)
```

After that we need to find in the `keep` matrix which basically the matrix to show if the <i>query value relevance</i> between target and search is in `top_n`.


After that, we need to find if the current query of the search sequence is in `top_n relevance` to any of the query of the target sequence.


```python
keep = torch.any(keep, dim=-2)
```

<img src="./README images/Attention dimension.svg">

Since `torch.any` is an agregation function and we have specify `dim=-2`, it will collapse the `L_t` dimension. Therefore, the final shape will be `(B, num_of_heads, L_s)`

Then, we concatenate the vector of `1` into `L_s` dimensions to ensure we can preserve the `[CLS]` token value when we apply the `keep` mask on the `attention_probs`.

```python
query_relevance = torch.cat((torch.ones(1, 12, 1, device=device), keep.clone()), dim=-1)
```
The shape of `query_relevance` now is `(B, num_of_heads, L_s + 1)`

We then appy this `keep` mask matrix onto the `attention_probs` matrix to only keep the top n relevance query.

```python
attention_probs = query_relevance[:, :, :, None] * attention_probs
```

We then repeat the same process for the `key_relevance`

```python
        # compute relevance between target key and search key
        if key_mod is not None:
            key_relevance = torch.matmul(key_mod[:, 1:, :], key_layer[:, :, 1:, :].transpose(-1, -2))             # 1 * 12 * target_patches * search_patches
            key_relevance = self.softmax(key_relevance / math.sqrt(key_mod.size(-1)))
            if crt_local_mod is not None:
                key_relevance = key_relevance * crt_local_mod
            keep = torch.argsort(torch.argsort(key_relevance, dim=-1)) >= (key_relevance.shape[-1] - top_n)       # 1 * 12 * target_patches * search_patches
            keep = torch.any(keep, dim=-2)                                                                        # 1 * 12 * search_patches
            keep = torch.ones_like(keep, dtype=torch.float32) * keep                                              # 1 * 12 * search_patches
            key_relevance = torch.cat((torch.ones(1, 12, 1, device=device), keep.clone()), dim=-1)                # 1 * 12 * (search_patches+1)
            attention_probs = key_relevance[:, :, None, :] * attention_probs                                      # 1 * 12 * (search_patches+1) * (search_patches+1)
            relevance = key_relevance
```

After that, it then apply another attention function with
- QUERY matrix as `query_relevance`.
- KEY matrix as `key_relevance`.
- VALUE matrix as `attention_probs`.


If we don't have the `query_mod` and `key_mod` but there is `crt_local_mod`, we can use it to mask over the attention probs. We apply softmax on the matrix then we just need to repeat 12 times in the `num_of_heads` dimension to create the same mask over all the attention heads, we then add `1` vector to the last dimension of `relevance`. Finally, we appy the mask on the `attention_probs`

```python
# modulate attention only by crt_local_mod
if crt_local_mod is not None and query_mod is None and key_mod is None:
    relevance = self.softmax(crt_local_mod).repeat(1, 12, 1)                                             # 1 * 12 * search_patches
    relevance = torch.cat((torch.ones(1, 12, 1, device=device), relevance.clone()), dim=-1)              # 1 * 12 * (search_patches+1)
    attention_probs = relevance[:, :, :, None] * attention_probs
```

We then multipy the `attention_probs` to the `value_layer`. The size of the result matrix will be
- Since the shape of `attention_probs` is `(B, nums_of_head, L_s + 1, L_s + 1)`
- The shape of `value_layer` is `(B, num_of_heads, L_s + 1, d_h)`
- The result matrix shape is `(B, num_of_heads, L_s + 1, d_h)`

```python
context_layer = torch.matmul(attention_probs, value_layer)
```

After that, we perform permutation
```python
context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
```
Effectively, the new shape will be `(B, L_s + 1, num_of_heads, d_h)`

We then keep the first 2 dimensions and and combine the last dimension to `all_head_size`

```python
new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
context_layer = context_layer.view(*new_context_layer_shape)
```

We then apply 1 more linear layer before we output the result
```python
attention_output = self.out(context_layer)
```

<img src='./README images/Attention Layer.svg'>



## class Mlp:

### `def __init__(self, config):`

Basically, this just initiate the 2 neuron layers.
- The first layer take an input size of `config.hidden_size` and has a number of neuron unit of `config.transformer['mlp_dim']` and has the activation function of `gelu`
- The second layer take the previous input from the 1st layer and has number of neuron unit of `config.hidden_size`, the activation function is `linear`.

## def forward(self, x):

It is pretty straight forward

```python
def forward(self, x):
    x = self.fc1(x)
    x = self.act_fn(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x
```

## class Block:

### `def __init__(self, config, vis):`

```python
super(Block, self).__init__()
self.hidden_size = config.hidden_size
self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
self.ffn = Mlp(config)
self.attn = Attention(config, vis)

```


### `def forward(self, x, crt_mod_method=None, crt_global_mod=None, crt_local_mod=None, query_mod=None, key_mod=None, top_n=None, device='cuda'):`


We first normalize the input before passing in the Attention Layer.
```python
h = x
x = self.attention_norm(x)
```

Then we pass the normalized input into the attention layer

```python
x, weights, query_layer, key_layer, value_layer, context_layer, attention_output, relevance = self.attn(x, crt_local_mod, query_mod, key_mod, top_n)
```

Notes that the return values of the attention layer will look like this
```python
return attention_output, weights, query_layer, key_layer, value_layer, context_layer, attention_output, relevance
```

Therefore, `x` will now be the `attention_output` of the attention layer.

The value `x[:, 1:]` (i.e. the `attention_ouput`) will then multiply element-wise with `crt_global_mod` and added to `h` (i.e. `x` before passing throught the attention layer), which create a residual connection between the attention layer. 

x is again assigned to h for another residual connection after the feedforward layer

x is then passed to the feedforward normalization layer and feed into the feedforward MLP.

x is then modulated again and added to h to create a residual connection.

<img src='./README images/Block Architecture.svg'>

## class Encoder

### `def __init__(self, config, vis):`

```python
super(Encoder, self).__init__()
self.vis = vis
self.layer = nn.ModuleList()
self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

for _ in range(config.transformer["num_layers"]):
    layer = Block(config, vis)
    self.layer.append(copy.deepcopy(layer))
```

### `def forward(self, hidden_states, crt_mod_method=None, crt_global_mod_layers=[], crt_global_mod=None, crt_local_mod=None, query_mod=None, key_mod=None, device='cuda'):`

