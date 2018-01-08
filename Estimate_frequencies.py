def Estimate_Frequency(Set)
    from toolz import frequencies
    wordcount_x = frequencies(' '.join(Set).split(' '))

    # Kraino is a framework that helps in fast prototyping Visual Turing Test models
    # This function takes wordcounts and returns word2index - mapping from words into indices, 
    # and index2word - mapping from indices to words and building the vocabulary.
    
    from kraino.utils.input_output_space import build_vocabulary
    word2index_x, index2word_x = build_vocabulary(
    this_wordcount=wordcount_x,
    truncate_to_most_frequent=0)
    word2index_x
return word2index_x
    
    