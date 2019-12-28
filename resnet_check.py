import torch as t
import torchvision as tv
from arxiv_baseline import ResNet
# import bad_grad
from visualize_net import make_dot

tv.models.resnet50()

# myModel = ResNet()
myModel = tv.models.resnet50()
myInput = t.randn((1,3,224,224),dtype=t.float32,requires_grad=True)
myOutput = myModel(myInput)
g = make_dot(myOutput)
g.view()
z = myOutput.mean()
# get_dot = bad_grad.register_hooks(z)
z.backward()
# dot = get_dot()
# dot.save('tmp.dot')
#
# import pydotplus as ps
# g = ps.graph_from_dot_file('./tmp.dot')
# g.write_jpg('res.jpg')
