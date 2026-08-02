from .loss import *

# class MyCriterion(nn.Module):
#     def __init__(self,entropy_weight=0.05):
#         super(MyCriterion, self).__init__()
#         self.entropy_weight = entropy_weight
#         print(self.entropy_weight)
#         self.DC = DiceLoss()
#         self.CE = CELoss()
#         self.CEW = CEWeightedLoss()

#     def forward(self, pred, label, entropy_map=None):
#         label_onehot = F.one_hot(label.long(), 3)
#         pred_t = F.one_hot(pred.argmax(dim=1).long(), 3)

#         dice_loss = self.DC(pred_t, label_onehot)
#         ce_loss = self.CE(pred, label_onehot)
#         supervised_loss = 0.5 * ce_loss + 0.5 * dice_loss

#         if entropy_map is not None:
#             entropy_loss = self.CEW(pred, label_onehot, entropy_map)
#             supervised_loss += self.entropy_weight * entropy_loss
#             # print(f"DC: {dice_loss.detach().cpu().numpy():.4f} || CE: {ce_loss.detach().cpu().numpy():.4f} || Entropy: {entropy_loss.detach().cpu().numpy():.4f}")
#         else:
#             # print(f"DC: {dice_loss.detach().cpu().numpy():.4f} || CE: {ce_loss.detach().cpu().numpy():.4f}")
#             pass

#         return supervised_loss


class MyCriterion(nn.Module):
    def __init__(self, entropy_weight=0.05, num_classes=3):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.DC = DiceLoss(num_classes=num_classes)
        self.CE = CELoss()
        self.CEW = CEWeightedLoss()

    def forward(self, pred, label, entropy_map=None):
        # Dice must operate on soft probabilities derived from the logits.
        # Applying argmax here would break its gradient path to the model.
        dice_loss = self.DC(pred, label)
        ce_loss = self.CE(pred, label)
        supervised_loss = 0.5 * ce_loss + 0.5 * dice_loss

        if entropy_map is not None:
            entropy_loss = self.CEW(pred, label, entropy_map)
            supervised_loss = supervised_loss + self.entropy_weight * entropy_loss

        return supervised_loss
