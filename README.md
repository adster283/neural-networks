# Single layer perceptron

I based the code on a tutorial from GeeksForGeeks with some changes to better fit my needs.
https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/

## Running the single layer perceptron script

To run the script you can use the following command:

`python3 perceptron.py ionosphere.data`

The above command assumes you are the the current directory and are wanting to train on the ionosphere data. You can easily change the data file for another by passing the path to the desired data.

*The script is configured to train for atleast 100 epochs and until if has no information gain for more than 21 epochs*