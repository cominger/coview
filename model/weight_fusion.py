import torch
import torch.nn as nn
import math

class weight_fusion(nn.Module):
    def __init__(self, class_one, class_two, pre_trained_net):
        super(weight_fusion,self).__init__()
        self.pre_trained_net = pre_trained_net
        self.wf_layer_scene = nn.Parameter(torch.ones(class_one))
        self.wf_layer_action = nn.Parameter(torch.ones(class_two))
        #self.wf_layer_scene = nn.Linear(class_one, class_one, bias=False)
        #self.wf_layer_scene.weight = nn.Parameter(torch.diag(torch.ones(class_one)))
        #self.wf_layer_action = nn.Linear(class_two, class_two, bias=False)
        #self.wf_layer_action.weight = nn.Parameter(torch.diag(torch.ones(class_two)))
    def forward(self, x):        
        scene_out, action_out = self.pre_trained_net(x)
        scene_out = torch.mul(scene_out, self.wf_layer_scene)
        action_out = torch.mul(action_out, self.wf_layer_action)
        #action_out = self.wf_layer_action(action_out)
        return scene_out, action_out

