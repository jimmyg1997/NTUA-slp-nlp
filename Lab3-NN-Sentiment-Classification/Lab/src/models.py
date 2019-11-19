import torch
import numpy as np
from torch import nn
from attention import SelfAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class BaselineDNN(nn.Module):
    
    @staticmethod
    def mean_pooling(x, lengths):
        sums = torch.sum(x, dim=1)
        _lens = lengths.view(-1, 1).expand(sums.size(0), sums.size(1))
        means = sums / _lens.float()

        return means

    @staticmethod
    def max_pooling(x):
        maximum = torch.max(x,dim = 1)
        maximum = maximum[0]
        return maximum

    def init_hidden(self,num_layers,batch_size,hidden_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(num_layers, batch_size, hidden_size)
        hidden_b = torch.randn(num_layers, batch_size, hidden_size)
        hidden_a = torch.autograd.Variable(hidden_a)
        hidden_b = torch.autograd.Variable(hidden_b)
        return (hidden_a, hidden_b)

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """


    def __init__(self, output_size, embeddings,trainable_emb=False):
        """
        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """
        #Necessary initiliazations for the DNN
        super(BaselineDNN, self).__init__()
        num_embeddings, emb_dim = embeddings.shape

        # 1 - define the embedding layer
        self.embedding = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = emb_dim) # EX4
        #Input: batch_size x seq_length
        #Output: batch_size x seq_length x embedding_dimension

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        #self.embedding.weight.data.copy_(torch.from_numpy(embeddings)) # EX4
        
        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad = trainable_emb)

        self.rnn = nn.LSTM(input_size = emb_dim, hidden_size = 20, num_layers = 1, batch_first=True)
        self.bi_rnn = nn.LSTM(input_size = emb_dim, hidden_size = 20, num_layers = 1, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(attention_size = 20, batch_first = True, non_linearity = "tanh")
        
        # 4 - define a non-linear transformation of the representations
        # Non-linearities
        self.non_linearity1 = nn.ReLU()
        self.non_linearity2 = nn.Tanh()

        # 5 - define the final Linear layer which maps
        # the representations to the classes 

        self.classifier = nn.Linear(in_features = 40, out_features=output_size) # EX5
        ##Input: batch_size x input_size (hidden_size of LSTM in this case or ??)
        #Output: batch_size x output_size




    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        #1 - embed the words, using the embedding layer
        #Input: batch_size x seq_len x 1
        #Output: batch_size x seq_len x embedding_dim
        embeddings = self.embedding(x)

        #####################
        #    1o ερωτημα     #
        #####################
        ############1.1############### ΚΑΙ ΠΡΕΠΕΙ + self.classifier = nn.Linear(in_features = 40, out_features=output_size)
        #representations_mean = self.mean_pooling(embeddings, lengths)
        #representations_max = self.max_pooling(embeddings)
        #representations = torch.cat((representations_mean,representations_max),1)
        #representations = self.non_linearity1(representations)

        ############1.2###############
        #representations_mean = self.mean_pooling(embeddings, lengths)
        #representations = representations_mean

        ###########################
        #    2o ερωτημα - LSTM    #
        ###########################
        ############2.1##############
        #Input: batch_size x seq_len x embedding_dim
        #representations_LSTM: batch_size x seq_len x embedding_dim
        #hn: batch_size x embedding_dim

        #representations_LSTM, (hn, cn) = self.rnn(embeddings)

        #Input: batch_size x seq_len x embedding_dim
        #Output: batch_size x embedding_dim

        #representations_LSTM = self.last_timestep(representations_LSTM,lengths)
        #representations = representations_LSTM

        ############2.2############### + self.classifier = nn.Linear(in_features = 3*emb_dim, out_features=output_size)
        #representations_LSTM, (hn, cn) = self.rnn(embeddings)
        #representations_mean = self.mean_pooling(representations_LSTM, lengths)
        #representations_max = self.max_pooling(representations_LSTM)
        #representations_LSTM = self.last_timestep(representations_LSTM,lengths)
        #representations = torch.cat((representations_mean,representations_max,representations_LSTM),1)

        ##############################
        #    3o ερωτημα - Attention  #
        ##############################
        ############3.1###############
        #representations,attentions = self.attention(embeddings, lengths)
        ############3.2###############
        #representations_LSTM, (hn, cn) = self.rnn(embeddings)
        #representations, attentions = self.attention(representations_LSTM, lengths)


        ##############################
        # 4o ερωτημα - Bidiractional #
        ##############################
        ############4.1############### + 
        #representations_LSTM, (hn, cn) = self.bi_rnn(embeddings)
        #representations_mean = self.mean_pooling(representations_LSTM, lengths)
        #representations_max = self.max_pooling(representations_LSTM)

        #hidden_size = 20
        #representations_LSTM_fw = representations_LSTM[:, :, :hidden_size]
        #representations_LSTM_bw = representations_LSTM[:, :, hidden_size:]
        #representations_LSTM_fw = self.last_timestep(representations_LSTM_fw,lengths)
        #representations_LSTM_bw = self.last_timestep(representations_LSTM_bw,lengths)
        #representations = torch.cat((representations_mean,representations_max,representations_LSTM_fw,representations_LSTM_bw),1)



        ############4.2############### + self.classifier = nn.Linear(in_features = 2*emb_dim, out_features=output_size)
        hidden_size = 20
        representations_LSTM, (hn, cn) = self.bi_rnn(embeddings)
        representations_LSTM_fw = representations_LSTM[:, :, :hidden_size]
        representations_LSTM_bw = representations_LSTM[:, :, hidden_size:]
        representations_LSTM_fw,attentions_fw = self.attention(representations_LSTM_fw, lengths)
        representations_LSTM_bw,attentions_bw = self.attention(representations_LSTM_bw, lengths)
        representations = torch.cat((representations_LSTM_fw,representations_LSTM_bw),1)


        # 4 - project the representations to classes using a linear layer
        logits = self.classifier(representations)  # EX6
        #Input: batch_size x embedding_dim
        #Output: batch_size x 3

        
        return logits
        #return logits, attentions
