this is to compute prior.

just train the model for the first 20 epochs for all the training samples with the corresponding noise rate.
save the probabilities of all epochs. 
record the one with the highest test acc.
then use anaconda to compute the average of all the output of probability according to samples' true labels.
