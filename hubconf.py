dependencies = ['torch']

import torch.hub

from model import BiSeNet as _BiSeNet

_pretrained_url = 'https://github.com/valgur/face-parsing.PyTorch/releases/download/v1.0/bisenet-468e13ca.pt'


def BiSeNet(pretrained=False, progress=True, map_location=None, n_classes=19):
    net = _BiSeNet(n_classes)
    if pretrained:
        net.load_state_dict(torch.hub.load_state_dict_from_url(
            _pretrained_url, map_location=map_location, progress=progress, check_hash=True))
    net.eval()
    return net
