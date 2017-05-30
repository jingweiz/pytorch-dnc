from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from random import randint
import torch

from core.env import Env

class RepeatCopyEnv(Env):
    def __init__(self, args, env_ind=0):
        super(RepeatCopyEnv, self).__init__(args, env_ind)

        # state space setup
        self.batch_size = args.batch_size
        self.len_word = args.len_word
        self.min_num_words = args.min_num_words
        self.max_num_words = args.max_num_words
        self.min_repeats = args.min_repeats
        self.max_repeats = args.max_repeats
        self.max_repeats_norm = args.max_repeats_norm   # to normalize the repeat flag to make it easier for the network
        self.logger.warning("Word     {length}:   {%s}", self.len_word)
        self.logger.warning("Words #  {min, max}: {%s, %s}", self.min_num_words, self.max_num_words)
        self.logger.warning("Repeats  {min, max}: {%s, %s}", self.min_repeats, self.max_repeats)

    def _preprocessState(self, state):
        # NOTE: state input in size: batch_size x num_words  x len_word
        # NOTE: we return as:        num_words  x batch_size x len_word
        # NOTE: to ease feeding in one row from all batches per forward pass
        for i in range(len(state)):
            state[i] = np.transpose(state[i], (1, 0, 2))
        return state

    @property
    def state_shape(self):
        # NOTE: we use this as the input_dim to be consistent with the sl & rl tasks
        return self.len_word + 2

    @property
    def action_dim(self):
        # NOTE: we use this as the output_dim to be consistent with the sl & rl tasks
        # NOTE: this is different from copy, cos it also has to output an end bit
        return self.len_word + 1

    def render(self):
        pass

    def _readable(self, datum):
        return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

    def visual(self, input_ts, target_ts, mask_ts, output_ts=None):
        """
        input_ts:  [(num_wordsx(repeats+1)+3) x batch_size x (len_word+2)]
        target_ts: [(num_wordsx(repeats+1)+3) x batch_size x (len_word+1)]
        mask_ts:   [(num_wordsx(repeats+1)+3) x batch_size x (len_word+1)]
        output_ts: [(num_wordsx(repeats+1)+3) x batch_size x (len_word+1)]
        """
        input_ts  = self._unnormalize_repeats(input_ts)
        output_ts = torch.round(output_ts * mask_ts) if output_ts is not None else None
        input_strings  = [self._readable(input_ts[:, 0, i])  for i in range(input_ts.size(2))]
        target_strings = [self._readable(target_ts[:, 0, i]) for i in range(target_ts.size(2))]
        mask_strings   = [self._readable(mask_ts[:, 0, 0])]
        output_strings = [self._readable(output_ts[:, 0, i]) for i in range(output_ts.size(2))] if output_ts is not None else None
        input_strings  = 'Input:\n'  + '\n'.join(input_strings)
        target_strings = 'Target:\n' + '\n'.join(target_strings)
        mask_strings   = 'Mask:\n'   + '\n'.join(mask_strings)
        output_strings = 'Output:\n' + '\n'.join(output_strings) if output_ts is not None else None
        # strings = [input_strings, target_strings, mask_strings, output_strings]
        # self.logger.warning(input_strings)
        # self.logger.warning(target_strings)
        # self.logger.warning(mask_strings)
        # self.logger.warning(output_strings)
        print(input_strings)
        print(target_strings)
        print(mask_strings)
        print(output_strings) if output_ts is not None else None

    def sample_random_action(self):
        pass

    def _normalize_repeats(self, repeats):
        return repeats / self.max_repeats_norm

    def _unnormalize_repeats(self, input_ts):
        if input_ts.size(1) == 1:
            return input_ts
        else:
            return input_ts.cpu() * self.unnormalize_input_ts.transpose(0, 1)

    def _generate_sequence(self):
        """
        generates [batch_size x num_words x len_word] data and
        prepare input & target & mask

        Returns:
        exp_state1[0] (input) : starts w/ a start bit, then the seq to be copied
                              : then an repeat flag, then 0's
            [0 ... 0, 1, 0;   # start bit
             data   , 0, 0;   # data with padded 0's
             0 ... 0, 0, 3;   # repeat flag (would also be normaized)
             0 ......... 0]   # num_wordsxrepeats+1 rows of 0's
        exp_state1[1] (target): 0's until after inputs has the repeat flag, then
                              : the seq to be copied, then an end bit
            [0 ... 0, 0;      # num_words+2 rows of 0's
             data   , 0;      # data
             data   , 0;      # data
             data   , 0;      # data
             0 ... 0, 1;]     # end bit
        exp_state1[2] (mask)  : 1's for all row corresponding to the target
                              : 0's otherwise}
            [0;               # num_words+2 rows of 0's
             1];              # num_wordsxrepeats+1 rows of 1's
        NOTE: we pad extra rows of 0's to the end of those batches with smaller
        NOTE: length to make sure each sample in one batch has the same length
        """
        self.exp_state1 = []
        # we prepare input, target, mask for each batch
        batch_num_words     = np.random.randint(self.min_num_words, self.max_num_words+1, size=(self.batch_size))
        batch_repeats       = np.random.randint(self.min_repeats, self.max_repeats+1, size=(self.batch_size))
        max_batch_num_words = np.max(batch_num_words)
        max_batch_repeats   = np.max(batch_repeats)

        self.exp_state1.append(np.zeros((self.batch_size, max_batch_num_words * (max_batch_repeats + 1) + 3, self.len_word + 2))) # input
        self.exp_state1.append(np.zeros((self.batch_size, max_batch_num_words * (max_batch_repeats + 1) + 3, self.len_word + 1))) # target
        self.exp_state1.append(np.zeros((self.batch_size, max_batch_num_words * (max_batch_repeats + 1) + 3, 1)))                 # mask
        self.unnormalize_ts = torch.ones(self.batch_size, max_batch_num_words * (max_batch_repeats + 1) + 3, self.len_word + 2)
        for batch_ind in range(self.batch_size):
            num_words = batch_num_words[batch_ind]
            repeats   = batch_repeats[batch_ind]
            data      = np.random.randint(2, size=(num_words, self.len_word))
            data_rep  = np.tile(data, (repeats, 1))
            # prepare input  for this sample
            self.exp_state1[0][batch_ind][0][-2] = 1                        # set start bit
            self.exp_state1[0][batch_ind][1:num_words+1, 0:self.len_word] = data
            self.exp_state1[0][batch_ind][num_words+1][-1] = self._normalize_repeats(repeats)   # normalize the repeat flag
            self.unnormalize_ts[batch_ind][num_words+1][-1] = self.max_repeats_norm             # to ease visualization w/ unnormalized repeat flag
            # prepare target for this sample
            self.exp_state1[1][batch_ind][num_words+2:num_words*(repeats+1)+2, 0:self.len_word] = data_rep
            self.exp_state1[1][batch_ind][num_words*(repeats+1)+2][-1] = 1  # set end bit
            # prepare mask   for this sample
            self.exp_state1[2][batch_ind][num_words+2:num_words*(repeats+1)+3, :] = 1

    def reset(self):
        self._reset_experience()
        self._generate_sequence()
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        self._generate_sequence()
        return self._get_experience()
