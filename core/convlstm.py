# import torch
# import torch.nn as nn
import tensorflow as tf
import core.common as common

class LayernormConvLSTMCell():

    def __init__(self, input_dim, hidden_dim, kernel_size, activation_function=None):
        super(LayernormConvLSTMCell, self).__init__()

        self.activation_function = activation_function

        self.input_dim = input_dim # 32*16
        self.hidden_dim = hidden_dim # 32*16

        self.kernel_size = kernel_size # 3*3
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # 1, 1

        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=False)  # [2*32*16, 4*32*16, 3*3, [1, 1]]

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined = tf.concat([input_tensor, h_cur], axis=-1)

        # combined_conv = self.conv(combined)
        in_channels = self.input_dim + self.hidden_dim
        out_channels=4 * self.hidden_dim
        combined_conv = common.convolutional(combined, (3, 3, in_channels,  out_channels), True, 'lstm_encoding')

        # cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, self.hidden_dim, dim=1)

        # b, c, h, w = h_cur.size()
        # b, c, h, w = tf.shape(h_cur)

        # i = torch.sigmoid(cc_i)
        # f = torch.sigmoid(cc_f)
        # o = torch.sigmoid(cc_o)
        i = tf.sigmoid(cc_i)
        f = tf.sigmoid(cc_f)
        o = tf.sigmoid(cc_o)

        # cc_g = torch.layer_norm(cc_g, [h, w])
        cc_g = tf.layers.batch_normalization(cc_g, beta_initializer=tf.zeros_initializer(),
                                        gamma_initializer=tf.ones_initializer(),
                                        moving_mean_initializer=tf.zeros_initializer(),
                                        moving_variance_initializer=tf.ones_initializer(), training=True)

        g = self.activation_function(cc_g)

        c_next = f * c_cur + i * g
        # c_next = torch.layer_norm(c_next, [h, w])
        c_next = tf.layers.batch_normalization(c_next, beta_initializer=tf.zeros_initializer(),
                                        gamma_initializer=tf.ones_initializer(),
                                        moving_mean_initializer=tf.zeros_initializer(),
                                        moving_variance_initializer=tf.ones_initializer(), training=True)

        h_next = o * self.activation_function(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        #         torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
        return (tf.zeros([batch_size, self.hidden_dim, height, width], dtype=tf.float32),
                tf.zeros([batch_size, self.hidden_dim, height, width], dtype=tf.float32))

hyper_channels = 32
class LSTMFusion():
    def __init__(self):
        super(LSTMFusion, self).__init__()

        input_size = hyper_channels * 16

        hidden_size = hyper_channels * 16

        self.lstm_cell = LayernormConvLSTMCell(input_dim=input_size,
                                                  hidden_dim=hidden_size,
                                                  kernel_size=(3, 3),
                                                  activation_function=tf.nn.crelu)

    def forward(self, current_encoding, current_state):
        batch, channel, height, width = current_encoding.size()

        if current_state is None:
            hidden_state, cell_state = self.lstm_cell.init_hidden(batch_size=batch,
                                                                  image_size=(height, width))
        else:
            hidden_state, cell_state = current_state

        next_hidden_state, next_cell_state = self.lstm_cell(input_tensor=current_encoding,
                                                            cur_state=[hidden_state, cell_state])

        return next_hidden_state, next_cell_state