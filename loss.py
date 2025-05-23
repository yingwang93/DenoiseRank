import torch

from losses.approxNDCG import approxNDCGLoss
from losses.oridinal import ordinal
from losses.pointwise import pointwise_rmse
from losses.listNet import listNet
from losses.rankNet import rankNet
from losses.lambdaLoss import lambdaLoss

class BaseRankLoss(torch.nn.Module):
    def forward(self, score, target):
        raise NotImplementedError

    def forward_per_list(self, score, target, length):
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)

        loss_per_list = [
            self(score_of_list, target_of_list)
            for score_of_list, target_of_list in zip(score_per_list, target_per_list)
        ]
        losses = torch.stack(loss_per_list)

        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            return losses.sum()

        loss = losses.mean()
        return loss


class MSELoss(BaseRankLoss):
    def forward(self, score, target):
        return torch.nn.functional.mse_loss(score, target)


class OrdinalLoss(BaseRankLoss):

    def forward(self, score, target):
        return ordinal(score[torch.newaxis, :], target[torch.newaxis, :])


class SoftmaxLoss(BaseRankLoss):
    def forward(self, score, target):
        softmax_score = torch.nn.functional.log_softmax(score, dim=-1)
        loss = -(softmax_score * target).mean()
        return loss

class ApproxNDCGLoss(BaseRankLoss):
    def forward(self, score, target):
        return approxNDCGLoss(score[torch.newaxis, :], target[torch.newaxis, :])
    
class PointwiseLoss(BaseRankLoss):
    def forward(self, score, target):
        return pointwise_rmse(score[torch.newaxis, :], target[torch.newaxis, :], 1)
    
    
class ListNetLoss(BaseRankLoss):
    def forward(self, score, target):
        return listNet(score[torch.newaxis, :], target[torch.newaxis, :])
    
class RankNetLoss(BaseRankLoss):
    def forward(self, score, target):
        return rankNet(score[torch.newaxis, :], target[torch.newaxis, :])

class NDCGLoss2PPLoss(BaseRankLoss):
    def forward(self, score, target):
        return lambdaLoss(score[torch.newaxis, :], target[torch.newaxis, :], weighing_scheme='ndcgLoss2PP_scheme')
class LambdaLoss(BaseRankLoss):
    def forward(self, score, target):
        return lambdaLoss(score[torch.newaxis, :], target[torch.newaxis, :])

class SoftmaxApproxNDCGLoss(BaseRankLoss):
    def forward(self, score, target, lambad=0.1):
        softmax_score = torch.nn.functional.log_softmax(score, dim=-1)
        softLoss = -(softmax_score * target).mean()
        appLoss = approxNDCGLoss(score[torch.newaxis, :], target[torch.newaxis, :])
        return lambad * softLoss + (1.0-lambad) * appLoss