import torch
import torch.nn.functional as F
import sys
sys.path.append("..")
from build import morse_complex as mc
from utils.functions import *
from concurrent.futures import ThreadPoolExecutor

class DMTSkeletonize(torch.nn.Module):

    def __init__(self, reparameterize=None, threshold = 0, epsilon = 0, delta = -1, foreground = True):
        super(DMTSkeletonize, self).__init__()
        self.reparameterize = reparameterize
        self.threshold = threshold
        self.epsilon = epsilon
        self.delta = delta
        self.foreground = foreground

    def _stochastic_discretization(self, img):
        """
        Function to binarize the image so that it can be processed by our skeletonization method.
        In order to remain compatible with backpropagation we utilize the reparameterization trick and a straight-through estimator.
        """

        alpha = (img + 1e-8) / (1.0 - img + 1e-8)

        uniform_noise = torch.rand_like(img)
        uniform_noise = torch.empty_like(img).uniform_(1e-8, 1 - 1e-8)
        logistic_noise = (torch.log(uniform_noise) - torch.log(1 - uniform_noise))

        img = torch.sigmoid((torch.log(alpha) + logistic_noise * self.beta) / self.tau)
        img = (img > 0.5).float()

        return img
    
    def _fixed_discretization(self, img):
        img = (img > 0.5).float()
        return img

    def _skeleton_(self, img):
        d = distance_transform(img).astype(np.float32)
        MC = mc.MorseComplex(d)
        MC.process_lower_stars(10, 10, 10)

        # TODO: For now filtering is disabled
        # MC.prepare_morse_skeleton_below(threshold=self.threshold, epsilon=self.epsilon, delta=self.delta)
        MC.extract_morse_skeleton_below(threshold=self.threshold, dimension=3)

        pixels_below = np.array(MC.get_morse_skeleton_below())
        
        skeleton = np.zeros_like(img)
        skeleton[pixels_below[:,0], pixels_below[:,1], pixels_below[:,2]] = 1

        return skeleton

    def forward(self, img):

        if self.reparameterize=="stochastic":
            img_bin = self._stochastic_discretization(img.detach().clone())
        elif self.reparameterize=="fixed":
            img_bin = self._fixed_discretization(img.detach().clone())
        else:
            img_bin = img.detach().clone()

        if self.foreground:
            img_bin = 1-img_bin

        items = [img_[0].cpu().numpy() for img_ in img_bin]

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Map the function to the items and execute in parallel
            results = list(executor.map(self._skeleton_, items))
            results = [torch.from_numpy(result).to(img.device) for result in results]

        skeleton = torch.stack(results, dim=0).unsqueeze(1)
        return skeleton*img
        
