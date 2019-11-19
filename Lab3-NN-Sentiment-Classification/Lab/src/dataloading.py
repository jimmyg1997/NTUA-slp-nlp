from torch.utils.data import Dataset
from tqdm import tqdm
import re
import string
import numpy
from ekphrasis.classes.tokenizer import SocialTokenizer
#from nltk.tokenize import TweetTokenizer
#Function to parse the text file given, line by line
def parser(strings):
    tokens = []
    social_tokenizer = SocialTokenizer(lowercase=False).tokenize 
    for s in strings:
        #temp = tokenize(s)
        temp = social_tokenizer(s)
        if temp != []:
            tokens.append(temp)
    return tokens

def tokenize(s):
    s_temp = s.strip().lower()
    s_temp = re.sub('[^A-Za-z\n\s]+', '', s_temp)
    s_temp = s_temp.replace('\n', ' ')
    s_temp = " ".join(s_temp.split())
    s_temp = s_temp.split(' ')
    s_temp[:] = [item for item in s_temp if (item != '' and 'http' not in item)]
    return s_temp


class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx,avg_length):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """
        # EX2
        #tweet = TweetTokenizer()
        #self.data = tweet.tokenize(X)

        self.data = parser(X)
        self.labels = y
        self.word2idx = word2idx
        self.avg_length = avg_length

    
    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):

        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3
        example = []
        for word in self.data[index]:
            if word in self.word2idx:
                example.append(self.word2idx[word])
            else:
                example.append(self.word2idx["<unk>"])
                #example.append("<unk>")

        avg_diff = len(self.data[index]) - self.avg_length
        #Zero padding if needed#
        if(avg_diff < 0):
            example += [0 for i in range(abs(avg_diff))]
        else:
            example = example[:self.avg_length]

        return numpy.array(example), self.labels[index], len(self.data[index])
        

