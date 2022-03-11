from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from fairseq.criterions import register_criterion
import logging
import torch
import math
from fairseq import metrics, utils

@register_criterion('my_label_smoothed_cross_entropy')
class MyLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        c_loss_weight,
        beta,
    ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.c_loss_weight=c_loss_weight
        self.num_updates=0
        self.beta=beta
        # self.ls=ls

    def set_num_updates(self, num_updates):
        self.num_updates=num_updates

    def add_args(parser):
        parser.add_argument('--c-loss-weight', default=0., type=float,
                            help='weight of confidence loss')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--beta', default=0., type=float, metavar='D',
                            help='epsilon for controling decay')
        # parser.add_argument('--ls',default=0.7,type=float,help="control label smoothing")      

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss,ori = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        # c_weight=self.c_loss_weight*math.exp(-self.num_updates/self.beta)/math.pow(net_output[2].mean(),2)
        c_weight=self.c_loss_weight*math.exp(-self.num_updates/self.beta)
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ori_nll_loss":ori.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "c":net_output[2].sum().data,
            "c_loss":(-torch.log(net_output[2])).sum().data,
            "c_weight":c_weight,
        }
        loss+=c_weight*(-torch.log(net_output[2])).sum()
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        del c_weight, ori, nll_loss, net_output
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=False) 
        target = model.get_targets(sample, net_output)
        # c = net_output[2] #[bsz*tgt,1]
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample) 
        loss, nll_loss, ori_loss = conf_label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            c=net_output[2].view(-1, 1),
            step=self.num_updates,
        )
        return loss, nll_loss, ori_loss

    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)

        sample_size = sum(log.get("ntokens", 0) for log in logging_outputs)
        c_sum = sum(log.get('c', 0) for log in logging_outputs)
        metrics.log_scalar('ave_c', c_sum / sample_size , sample_size, round=4)

        ori_sum = sum(log.get('ori_nll_loss', 0) for log in logging_outputs)
        metrics.log_scalar('ori_nll_loss', ori_sum / sample_size , sample_size, round=4)
        
        c_loss_sum = sum(log.get('c_loss', 0) for log in logging_outputs)
        metrics.log_scalar('c_loss', c_loss_sum / sample_size , sample_size, round=4)
        c_weight = sum(log.get('c_weight', 0) for log in logging_outputs)
        metrics.log_scalar('c_weight', c_weight /4, sample_size, round=4)

        del sample_size, c_sum, ori_sum, c_loss_sum, c_weight

def conf_label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True,c=None, step=0):
    # lprobs: [bsz * tgt, vocab]
    # target: [bsz * tgt]
    # c: [bsz*tgt,1]
    
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1) #[bsz*tgt,1]
    drop_prob = 0.5
    c_tmp = c
    c_mask = (torch.rand(c.shape)<drop_prob).cuda()
    c = torch.where(c_mask, torch.ones_like(c), c)

    ori_loss=(-torch.log(lprobs).gather(dim=-1, index=target)).sum()

    lprobs=torch.log(lprobs*c+(1-c)*torch.nn.functional.one_hot(target, lprobs.size(-1)).squeeze(1))
    
    nll_loss = -lprobs.gather(dim=-1, index=target) #[bsz*tgt,1]
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True) #[bsz*tgt,1]

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)


    epsilon = epsilon*torch.pow(torch.exp(1-c_tmp/torch.mean(c_tmp)),1)
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    # loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * 2 *(torch.rand(smooth_loss.shape).cuda() * smooth_loss)
    return loss.sum(), nll_loss.sum(), ori_loss
    
