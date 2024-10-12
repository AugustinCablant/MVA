# I> Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)

*References* : https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

Sequence-to-sequence models are deep learning models that have achieved a lot of success in tasks like machine translation, text summarization, and image captioning.
In neural machine translation, a sequence is a series of words, processed one after another. The output is, likewise, a series of words. (Translation French -> English for example)

**The model is composed of an encoder and a decoder:**

- The encoder processes each item in the input sequence, it compiles the information it captures into a vector (called the context). After processing the entire input sequence, the encoder sends the context over to the decoder, which begins producing the output sequence item by item.
- You can set the size of the context vector when you set up your model. It is basically the number of hidden units in the encoder RNN. 
- By design, a RNN takes two inputs at each time step: an input (in the case of the encoder, one word from the input sentence), and a hidden state

The context vector turned out to be a bottleneck for these types of models. It made it challenging for the models to deal with long sentences. A solution was proposed in Bahdanau et al., 2014 (https://arxiv.org/pdf/1409.0473) and Luong et al., 2015 (https://arxiv.org/pdf/1508.04025). These papers introduced and refined a technique called “Attention”, which highly improved the quality of machine translation systems. Attention allows the model to focus on the relevant parts of the input sequence as needed.

**The Attention mechanism**

- First, the encoder passes a lot more data to the decoder. Instead of passing the last hidden state of the encoding stage, the encoder passes all the hidden states to the decoder.

- Second, an attention decoder does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the decoder does the following:
    a. Look at the set of encoder hidden states it received – each encoder hidden state is most associated with a certain word in the input sentence
    b. Give each hidden state a score (let’s ignore how the scoring is done for now)
    c. Multiply each hidden state by its softmaxed score, thus amplifying hidden states with high scores, and drowning out hidden states with low scores