import torch
import numpy as np
from torch import nn

class BaselineDNN(nn.Module):
    
    @staticmethod
    def mean_pooling(x, lengths):
        sums = torch.sum(x, dim=1)
        _lens = lengths.view(-1, 1).expand(sums.size(0), sums.size(1))
        means = sums / _lens.float()
        return means

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
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_dim) # EX4
        #Input: batch_size * seq_length
        #Output: batch_size * seq_length * embedding_dimension

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        #self.embedding.weight.data.copy_(torch.from_numpy(embeddings)) # EX4
        
        # 3 - define if the embedding layer will be frozen or finetuned
        #1st way: self.embedding.weight.requires_grad = trainable_emb # EX4
        #2nd way: weight = torch.FloatTensor(embeddings), self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=trainable_emb)

        # 4 - define a non-linear transformation of the representations
        # Non-linearities
        self.non_linearity1 = nn.ReLU()
        self.non_linearity2 = nn.Tanh()

        # 5 - define the final Linear layer which maps
        # the representations to the classes1   014§322.   

        self.classifier = nn.Linear(in_features=emb_dim, out_features=output_size) # EX5
        ##Input: batch_size x input_size (hidden_size of LSTM in this case or ??)
        #Output: batch_size x output_size




    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        #encode
        # 1 - embed the words, using the embedding layer

        embeddings = self.embedding(x)
        
        # 2 - construct a sentence representation out of the word embeddings
    
        representations = self.mean_pooling(embeddings, lengths)

        
        # 3 - transform the representations to new ones.
        
        representations = self.non_linearity1(representations)


        # 4 - project the representations to classes using a linear layer
        logits = self.classifier(representations)  # EX6
        return logits
