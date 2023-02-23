import torch
import torch.nn as nn
from typing import Optional,Any,Tuple,Union

nn.TransformerDecoder

from functools import partial
from collections import OrderedDict

'''
    The Total Architecture of ViT
    
    input:[batch_size,channels,H,W]
    1.PatchEmbedding(nn.Conv2d(kernel_size=patch_size,stride=patch_size))    
    2.Add Class Token:[batch_size,1,embedding_dims] 
    3.Add Position Embedding:[batch_size,num_of_patches,embedding_dims]
    4.Go across [L] layers Transformer Encoder
    5.Go across MLP head
    6.Get the classification result

    (make an example by ImageNet)
    
    input:[batch_size,3,224,224]
    patch_size = 16 
    
    'then you will get 224/16 * 224/16 = 14*14 = 196 patches'
    num_of_patches = 196
    
    'for one patch 16*16 with 3 channels to represent'
    embedding_dims = channels * patch_size * patch_size = 3*16*16 = 768
    
    'so input after PatchEmbedding' 
    input_embedding:[batch_size,196,768]
    class token:[batch_size,1,768]
    input_ = concat([class_token,input_embedding],dim=1) 
           : [batch_size,197,768]
    position_embedding:[batch_size,197,768]
    input_ = input_ + position_embedding
    
    'Go across [L] layers Transformer Encoder'
    output = [batch_size,197,768]
    class_token = output[:,0]
                : [batch_size,1,768]
    
    'Go across MLP head'
    class = [batch_size,num_classes]
    
    'Softmax then get result'
    predict = torch.softmax(class,dim=1).argmax(dim=1)
    

'''

class PatchEmbedding(nn.Module):
    def __init__(self,img_size: Union[int,Tuple] = 224,
                 patch_size: Union[int,Tuple] = 16,
                 in_channels: int = 3,
                 embedding_dim: int = 768,
                 norm_layer: Optional[Any] = None):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size if isinstance(img_size,tuple) else (img_size,img_size)
        patch_size = patch_size if isinstance(patch_size,tuple)else (patch_size,patch_size)
        self.num_of_patches = (self.img_size[0]//patch_size[0]) * (self.img_size[1]//patch_size[1])

        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embedding_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embedding_dim) if norm_layer else nn.Identity()


    def forward(self,x):
        batch_size,channels,height,width = x.shape
        assert height == self.img_size[0] and width == self.img_size[1]

        #[B,C,H,W] -> [B,Embed,H/P,W/P] -> [B,Embed,N] -> [B,N,Embed]
        x = self.proj(x).flatten(start_dim = 2).transpose(1,2)
        x = self.norm(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets

    # create an random matrix:[batch_size,1,1,1,...]
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

    # 有drop_prob概率 < 1然后floor变成0，有drop_prob概率 > 1然后floor变成1
    random_tensor.floor_()  # binarize

    # turn mean to be x but not p*x
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MultiheadAttention(nn.Module):
    '''
        MultiHeadAttension Module

        input:qkv -> nn.Linear -> scaled dot product -> attn_drop -> nn.Linear -> drop

    '''
    def __init__(self,embedding_dim: int = 768,
                 num_heads: int = 8,
                 qkv_use_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 attn_drop_ratio = 0.,
                 proj_drop_ratio = 0.):
        super(MultiheadAttention, self).__init__()

        self.num_heads = num_heads
        head_dim = embedding_dim // num_heads
        self.scale = qk_scale if qk_scale else head_dim ** -0.5
        self.qkv = nn.Linear(embedding_dim,embedding_dim*3,bias=qkv_use_bias)
        self.attn_drop = nn.Dropout(p = attn_drop_ratio)
        self.proj = nn.Linear(embedding_dim,embedding_dim)
        self.proj_drop = nn.Dropout(p = proj_drop_ratio)


    def forward(self,x):
        batch_size,num_of_patches,embedding_dims = x.shape

        #[B,N,Embed*3]
        qkv = self.qkv(x).reshape(batch_size,num_of_patches,3,\
                                  self.num_heads,embedding_dims//self.num_heads)\
            .permute(2,0,3,1,4)
        #[3,B,H,N,Embed//H]

        q,k,v = qkv[0],qkv[1],qkv[2]

        #[B,H,N,N]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #[B,H,N,Embed//H] -> [B,N,Embed]
        # 这里直接指明维度能够检查错误
        x = (attn @ v).transpose(1,2).reshape(batch_size,num_of_patches,embedding_dims)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    '''
        FFN
        input -> nn.Linear(embed,embed*4) -> GELU -> dropout
        -> nn.Linear(embed*4,embed) -> dropout

    '''
    def __init__(self,in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Optional[nn.Module] = nn.GELU,
                 drop: float = 0.):
        super(MLP, self).__init__()
        out_features = out_features if out_features else in_features
        hidden_features = hidden_features if hidden_features else in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(p = drop)

    def forward(self,x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Block(nn.Module):
    '''
        TransformerEncoder Block
        input -> pre_norm -> MultiHeadAttension -> Droppath -> MLP -> Droppath

    '''
    def __init__(self,embedding_dim: int = 768,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 drop_ratio: float = 0.,
                 attn_drop_ratio: float = 0.,
                 drop_path_ratio: float = 0.,
                 act_layer: Optional[nn.Module] = nn.GELU,
                 norm_layer: Optional[nn.Module] = nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(embedding_dim)
        self.attn = MultiheadAttention(embedding_dim = embedding_dim,
                              num_heads = num_heads,
                              qkv_use_bias = qkv_bias,
                              qk_scale = qk_scale,
                              attn_drop_ratio = attn_drop_ratio,
                              proj_drop_ratio = drop_ratio)
        self.drop_path = DropPath(drop_prob = drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(embedding_dim)
        mlp_hidden_dims = int(embedding_dim * mlp_ratio)
        self.mlp = MLP(in_features = embedding_dim,
                       hidden_features = mlp_hidden_dims,
                       act_layer = act_layer,
                       drop = drop_ratio)

    def forward(self,x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class VisionTransformer(nn.Module):
    '''
        ViT

        input -> PatchEmbedding -> cat class token -> Add PositionEmbedding
        -> Transformer Block * n -> extract class token -> classification head
        -> result

    '''
    def __init__(self,img_size: Union[int,Tuple] = 224,
                 patch_size: Union[int,Tuple] = 16,num_classes: int = 1000,
                 num_heads: int = 12,embedding_dim: int = 768,
                 num_of_blocks: int = 12,in_channels: int = 3,
                 mlp_ratio: float = 4.,qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,drop_ratio: float = 0.,
                 attn_drop_ratio: float = 0.,drop_path_ratio: float=0.,
                 embedding_layer: Optional[Any] = PatchEmbedding,
                 norm_layer: Optional[Any] = None,
                 act_layer: Optional[Any] = None,
                 representation_size: Optional[int] = None,
                 distilled: bool = False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embedding_dim = embedding_dim
        self.num_tokens = 1
        norm_layer = norm_layer if norm_layer else partial(nn.LayerNorm,eps=1e-6)
        act_layer = act_layer if act_layer else nn.GELU

        self.patch_embed = embedding_layer(img_size = img_size,
                                           patch_size = patch_size,
                                           in_channels = in_channels,
                                           embedding_dim = embedding_dim)

        num_of_patches = self.patch_embed.num_of_patches

        self.cls_token = nn.Parameter(torch.zeros(1,1,embedding_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_of_patches+self.num_tokens,embedding_dim))
        self.pos_drop = nn.Dropout(p = drop_ratio)

        dpr = [x.item() for x in torch.linspace(0,drop_path_ratio,num_of_blocks)]
        self.blocks = nn.Sequential(*[
            Block(embedding_dim=embedding_dim,num_heads = num_heads,mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,qk_scale=qk_scale,drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i],act_layer=act_layer,norm_layer=norm_layer)
            for i in range(num_of_blocks)
        ])

        self.norm = norm_layer(embedding_dim)

        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc",nn.Linear(embedding_dim,representation_size)),
                ("act",nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self,x):
        x = self.patch_embed(x)
        # can copy dim=1 to n
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat([cls_token,x],dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:,0])

    def forward(self,x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _init_vit_weights(m):

    if isinstance(m,nn.Linear):
        nn.init.trunc_normal_(m.weight,std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m,nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embedding_dim=768,
                              num_of_blocks=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embedding_dim=768,
                              num_of_blocks=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embedding_dim=768,
                              num_of_blocks=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embedding_dim=768,
                              num_of_blocks=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embedding_dim=1024,
                              num_of_blocks=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embedding_dim=1024,
                              num_of_blocks=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embedding_dim=1024,
                              num_of_blocks=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embedding_dim=1280,
                              num_of_blocks=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model