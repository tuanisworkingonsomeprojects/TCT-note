# Some definitions for configs.py

## ViT-[model_size]/[patch_size]

While reading the `configs.py` code you may encounter some methods such as `get_b16_config()`, and the function description like `"""Returns the ViT-B/16 configuration."""` is not explicit.

So what does **ViT-B/16** mean?

- ViT: Vision Transformer model
- B: is the size of the model (inspired by BERT naming convention).
    - BERT model size naming convention:
        - B: Base (12 Transformer layers)
        - L: Large (24 Transformer layers)
        - H: Huge (32 Transformer layers)

- /16: patch size - the input image is split into non-overlapping **16x16** pixel patches

So basiically, Vit-B/16 is a configuration for Vision Transoformer model of Base size with patch size of 16x16.

Similary, we can now understand the configuration for the `get_b32_config()`, `get_l16_config`, `get_l32_config()`, `get_h14_config()`

We only have 1 special case `get_r50_b16_config()`, however the function description is now understandable thanks to the definition.   `"""Returns the Resnet50 + ViT-B/16 configuration."""`

## config.classifier = 'token'
In some method that we discuss above such as `get_b16_config()`, we can see some methods that have a line `config.classifier = 'token'`.

So what does `config.classifier = 'token'` actually mean?

So in simple terms, we will prepend a learnable embedding vector / token `[CLS]` into the beginning of the input before the prepended input is passed into the Transformer.

Why prepending `[CLS]` token works?

Because when we do self-attention, we allow `[CLS]` token to see everything in the same input.

$$
\text{Attention}([\text{CLS}]) = \text{softmax}(\frac{Q_{[CLS]}K_{All}^T}{\sqrt{d_{K_{All}}}}) \cdot V_{All}
$$

When we use query matrix $Q_{[CLS]}$ $(1 \times d)$ we can match to the key matrix of all the tokens of the input $K_{All}$ (N x d) and then compute the attention weights matrix for each value of other token by dot product operation $Q_{[CLS]}K_{All}^T$ $((1 \times d) \cdot (d \times N) = (1 \times N))$ and scale it to [0, 1] range by using softmax activation function. Finally, get the weighted sum for each token vector by dot product the weights to the values matrix $V_{All}$: 

$$
\text{softmax}(\frac{Q_{[CLS]}K_{All}^T}{\sqrt{d_{K_{All}}}}) \cdot V_{All} = \text{Attention([CLS])}
$$

$(1 \times N) \cdot (N \times d_V) = (1 \times d_V)$


## config.hidden_size
