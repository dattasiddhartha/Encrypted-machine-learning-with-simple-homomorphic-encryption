### Implementation of orthogonal/inverted matrix-based homomorphic encrpytion for somewhat-encrypyted machine learning

-for experimental purposes only

Taking the UCI credit default dataset, we built a benchmark classification model (~75%).

Then encrypted the dataset using a set of matrix transformations based on the homomorphic encryption schemata [here](https://www.cs.cmu.edu/~rjhall/JOS_revised_May_31a.pdf).

Running a standard backpropagation neural network model yielded similar accuracy (~74%) to the vanilla model, indicating no loss of insight/pattern during encryption.
