import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Split image into patches and then embed them.
    Paramaters
    ----------
    img_size : int
        Size of input image. (square)
        
    patch_size : int
        Size of one patch. (square)
        
    in_chans : int
        Number of input image channels.
    
    embed_dim : int
        The embedding dimension.
        
    Attributes
    ----------
    n_patches : int
        Number of patches inside our image.
    
    proj : nn.Conv2d
        Convolutional layer that converts our image into patches and their embeddings.
    """
    
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """ Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Input image with shape (n_samples, in_chans, img_size, img_size).
        
        Returns
        -------
        torch.Tensor
            Patch embeddings with shape (n_samples, n_patches, embed_dim).
        """
        x = self.proj(x) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5) 
        x = x.flatten(2) # (n_samples, embed_dim, n_patches)
        x = x.transopse(1, 2) # (n_samples, n_patches, embed_dim)
        
        return x
    
class Attention(nn.Module):
    """Attention mechanism.
    
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    
    n_heads : int
        Number of attention heads.
    
    qkv_bias : bool
        If True, add a learnable bias to query, key, value.
        
    attn_p : float
        Dropout probability for attention weights.
        
    proj_p : float
        Dropout probability for output.
    
    Attributes
    ----------
    scale : float
        Normalizing constant for attention weights.
        
    qkv : nn.Linear
        Linear transformation for query, key, value.
    
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention heads and projects it to the output dimension.
    
    attn_drop, proj_drop : nn.Dropout
        Dropout layers for attention weights and output.
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5 # Not to feed too large values to softmax
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x):
        """ Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (n_samples, n_patches + 1, dim).
        
        Returns
        -------
        torch.Tensor
            Output tensor with shape (n_samples, n_patches + 1, dim).
        """
        n_samples, n_tokens, dim = x.shape
        
        if dim != self.dim:
            raise ValueError(f"Input dimension {dim} does not match config {self.dim}")
        
        qkv = self.qkv(x) # (n_samples, n_patches +1 , dim * 3)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (n_samples, n_heads, n_patches + 1, head_dim)
        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q@k_t) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)
        
        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg) # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x) # (n_samples, n_patches + 1, dim)
        
        return x
    
class MLP(nn.Module):
    """Multi-layer perceptron.
    
    Parameters
    ----------
    in_features : int
        Number of input features.
        
    hidden_features : int
        Number of nodes in the hidden layer.
        
    out_features : int
        Number of output features.
        
    p : float
        Dropout probability.
    
    Attributes
    ----------
    fc : nn.Linear
        The first linear layer.
    
    act : nn.GELU
        GELU activation function.
        
    fc2 : nn.Linear
        The second linear layer.
        
    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        """ Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (n_samples, n_patches + 1, in_features).
        
        Returns
        -------
        torch.Tensor
            Output tensor with shape (n_samples, n_patches + 1, out_features).
        """
        x = self.fc1(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (n_samples, n_patches + 1, out_features)
        x = self.drop(x) # (n_samples, n_patches + 1, out_features)
        
        return x
    
class Block(nn.Module):
    """Transformer block.
    
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    
    n_heads : int
        Number of attention heads.
        
    mlp_ratio : float
        Ratio of mlp hidden dim to embedding dim.
        
    qkv_bias : bool
        If True, add a learnable bias to query, key, value.
        
    p, attn_p : float
        Dropout probability for attention weights and output.
        
    Attributes
    ----------
    norm1, norm2 : nn.LayerNorm
        Layer normalization layers.
        
    attn : Attention
        Attention layer.
        
    mlp : MLP
        MLP layer.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim, hidden_features=hidden_features, out_features=dim)
        
    def forward(self, x):
        """Run forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (n_samples, n_patches + 1, dim).
            
        Returns
        -------
        torch.Tensor
            Output tensor with shape (n_samples, n_patches + 1, dim).
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x

class VisionTransformer(nn.Module):
    """Simplified Implementation of Vision Transformer.
    
    Parameters
    ----------
    img_size : int
        Size of input image.
        
    patch_size : int
        Size of patch.
        
    in_chans : int
        Number of input channels.
    
    n_classes : int
        Number of classes.
        
    embed_dim : int
        Dimension of embedding.
        
    depth : int
        Number of transformer blocks.
        
    n_heads : int
        Number of attention heads.
        
    mlp_ratio : float
        Ratio of mlp hidden dim to embedding dim.
        
    qkv_bias : bool
        If True, add a learnable bias to query, key, value.
        
    p, attn_p : float
        Dropout probability for attention weights and output.
        
    Attributes
    ----------
    patch_embed : PatchEmbed
        Patch embedding layer.
        
    cls_token : torch.Tensor
        Class token.
        
    pos_emb : torch.Tensor
        Positional embedding.
        
    pos_drop : nn.Dropout
        Dropout layer.
        
    blocks : nn.ModuleList
        Transformer blocks.
    
    norm : nn.LayerNorm
        Layer normalization layer.
    """
    
    def __init__(self, img_size=384, patch_size=16, in_chans=3, n_classes=1000, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p)
        self.blocks = nn.ModuleList([Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.head = nn.Linear(embed_dim, n_classes) 
        
    def forward(self, x):
        """Run the forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (n_samples, in_chans, img_size, img_size).
        
        Returns
        -------
        logits : torch.Tensor
            Output tensor with shape (n_samples, n_classes).
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x) # (n_samples, n_patches, embed_dim)
        cls_tokens = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1) # (n_samples, n_patches + 1, embed_dim)
        x = x + self.pos_embed # (n_samples, n_patches + 1, embed_dim)
        x = self.pos_drop(x) # (n_samples, n_patches + 1, embed_dim) 
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x) # (n_samples, n_patches + 1, embed_dim)
        
        cls_token_final = x[:, 0] # (n_samples, embed_dim)
        x = self.head(cls_token_final) # (n_samples, n_classes)
        
        return x
    