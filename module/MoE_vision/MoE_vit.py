from timm.models.vision_transformer import Block
import torch
from torch import nn

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()

        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.Softplus())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2

class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            SimpleGate(dim=2),
            torch.nn.Linear(int(input_shape / 2), 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        # scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # scores = self.attention_layer(inputs)
        # outputs = scores*inputs
        return outputs, scores

class MoE_vit(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.num_expert = 3 #2~5为宜
        self.depth = 1 #1~2为宜
        self.unified_dim = dim
        self.image_attention = TokenAttention(self.unified_dim)

        # MMoE init
        image_expert_list = []
        for i in range(self.num_expert):
            image_expert = []
            for j in range(self.depth):
                image_expert.append(Block(dim=self.unified_dim, num_heads=8)) 
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)

        self.image_experts = nn.ModuleList(image_expert_list)
        
        self.image_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                            SimpleGate(),
                                            nn.BatchNorm1d(int(self.unified_dim / 2)),
                                            nn.Linear(int(self.unified_dim / 2), self.num_expert),
                                            # nn.Dropout(0.1),
                                            # nn.Softmax(dim=1)
                                            )
        # MLP输出weight
        self.mlp = MLP(input_dim=dim, embed_dims=[256], dropout=0.2, output_layer=True)
        self.act = nn.Softplus()
        
    def forward(self, x):
        # x : image_feature from Vit (BATCH, 197, 1024)
        image_feature = x
        # print('image_feature.shape',image_feature.shape)
        # [8, 197, 768]

        image_atn_feature, _ = self.image_attention(x)
        gate_image_value = self.image_gate(image_atn_feature)
        # print('gate_image_value.shape',gate_image_value.shape)
        # [8, 3] [bs,num_expert]

        refine_image_feature =  0
        for i in range(self.num_expert):
            image_expert = self.image_experts[i]
            tmp_image_feature = image_feature
            for j in range(self.depth):
                tmp_image_feature = image_expert[j](tmp_image_feature)
            refine_image_feature\
                += (tmp_image_feature * gate_image_value[:, i].unsqueeze(1).unsqueeze(1))
        refine_image_feature = refine_image_feature[:, 0]

        # print('refine_image_feature.shape',refine_image_feature.shape)
        # [8, 768] [bs,768]


        weight = self.mlp(refine_image_feature)
        weight = self.act(weight)  # Apply sigmoid activation
        # print('weight',weight)
        # print(weight.shape)
        # [8, 1]    
        return weight  # 返回输出



def main():
    num_experts = 3
    vit_config = {}  
    model = MoE_vit(num_experts, vit_config)

    x = torch.randn(32, 64, 1024)
    output = model.forward(x)
    print(output.shape)

if __name__ == "__main__":
    main()
