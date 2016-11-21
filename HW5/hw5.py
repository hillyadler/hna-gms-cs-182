from math import log

class TextClassifier:
    """
    In this question, you will implement a classifier that predicts
    the number of stars a reviewer gave a movie from the text of the review.

    You will process the reviews in the dataset we provide, then
    implement a Naive Bayes classifier to do the prediction.

    But first, some math!
    """

    def q0(self):
        """
        Return your full name as it appears in the class roster
        """
        return ["Geoffrey Stevens", "Hilly Adler"]

    def q1(self):
        """
        Suppose you roll a 4-sided die of unknown bias, and you observe
        the following sequence of values:
        3   1   4   4   2   1   2   4   2   1   2   1   1   1   1   4   3   4   4   1
        Given only this information, what are the most likely
        probabilities of rolling each side? (Hardcoding is fine)
        """
        return [0.4, 0.2, 0.1, 0.3]

    def q2(self):
        """
        You just fit a multinomial distribution!

        Now suppose you have a prior belief that the die is fair.
        After some omitted math involving a conjugate Dirichlet distribution,
        you realize that you can encode this prior by simply adding
        some "fake" observations of each side. The number of observations
        is the "strength" of your prior belief.
        Using the same observations as in q1 and a prior with a per-side
        "strength" of 2, what are the probabilities of rolling each side??
        """
        return [10./28., 6./28., 4./28., 8./28.]

    def q3(self, counts=[1,1,3,8]):
        """
        You might be wondering what dice have to do with NLP.
        We will model each possible rating (one of the five numbers of stars)
        as a die, with each word in the dictionary as a face.

        This is a multinomial Naive Bayes classifier, because the words are
        drawn from a per-rating multinomial distribution and we treat
        each word in a review as independent (conditioned on the rating). That is,
        once the rating has emitted one word to the review, the next word
        has the same distribution over possible values as the first.

        In this question, you will write a function that computes p(word|rating), the
        probability that the rating under question will produce
        each of the four words in our dictionary. We will run this function
        5 times, once for each rating. We pass in the number of times each
        word shows up in any review corresponding to the current rating.
        """

        return [float(x) / sum(counts) for x in counts]

    def q4(self, infile):
        """
        You'll notice that actual words didn't appear in the last question.
        Array indices are nicer to work with than words, so we typically
        write a dictionary encoding the words as numbers. This turns
        review strings into lists of integers. You will count the occurrences
        of each integer in reviews of each class.

        The infile has one review per line, starting with the rating and then a space.
        Note that the "words" include things like punctuation and numbers. Don't worry
        about this distinction for now; any string that occurs between spaces is a word.

        You must do three things in this question: build the dictionary,
        count the occurrences of each word in each rating and count the number
        of reviews with each rating.
        The words should be numbered sequentially in the order they first appear.
        counts[ranking][word] is the number of times word appears in any of the
        reviews corresponding to ranking
        nrated[ranking] is the total number of reviews with each ranking
        """

        fopen = open(infile, 'r')
        d ={}
        counts = [[], [], [], [], []]
        nrated = [0, 0, 0, 0, 0]

        key = 0
        for line in fopen:
            words = line.split()
            ranking = int(words.pop(0))
            for word in words:
                if word not in d:
                    d[word] = key
                    key = key + 1
                    for r in range(5):
                        counts[r].append(0)

                counts[ranking][d[word]] += 1
            nrated[ranking] += 1

        self.dict = d
        self.counts = counts
        self.nrated = nrated

    def q5(self, alpha=1):
        """
        Now you'll fit the model. For historical reasons, we'll call it F.
        F[rating][word] is -log(p(word|rating)).
        The ratings run from 0-4 to match array indexing.
        Alpha is the per-word "strength" of the prior (as in q3).
        (What might "fairness" mean here?)
        """

        f = []
        total_words = 0
        for rating in range(5):
            total_words = float(sum(self.counts[rating]))
            unique = float(len(self.counts[rating]))
            f.append( [0 for x in range(len(self.dict))] )
            for key, value in self.dict.iteritems():
                f[rating][value] = -log( (self.counts[rating][value] + alpha) / (total_words + alpha*unique) )

        self.F = f

    def q6(self, infile):
        """
        Test time! The infile has the same format as it did before. For each review,
        predict the rating. Ignore words that don't appear in your dictionary.
        Are there any factors that won't affect your prediction?
        You'll report both the list of predicted ratings in order and the accuracy.
        """

        fopen = open(infile, 'r')

        predictions = []
        correct = 0

        for line in fopen:
            words = line.split()
            true_rank = int(words.pop(0))
            max_prob = float('inf')
            for rank in range(5):
                prob = -log(float(self.nrated[rank]) / sum(self.nrated))
                for word in words:
                    if word in self.dict:
                        prob += self.F[rank][self.dict[word]]
                if prob < max_prob:
                    predicted_rank = rank
                    max_prob = prob
            if predicted_rank == true_rank:
                correct += 1

            predictions.append(predicted_rank)

        return (predictions, float(correct) / len(predictions) )

    def q7(self, infile):
        """
        Alpha (q5) is a hyperparameter of this model - a tunable option that affects
        the values that appear in F. Let's tune it!
        We've split the dataset into 3 parts: the training set you use to fit the model
        the validation and test sets you use to evaluate the model. The training set
        is used to optimize the regular parameters, and the validation set is used to
        optimize the hyperparameters. (Why don't you want to set the hyperparameters
        using the test set accuracy?)
        Find and return a good value of alpha (hint: you will want to call q5 and q6).
        What happens when alpha = 0?
        """

        best_alpha = 2;
        self.q5(best_alpha)
        predictions, best_accuracy = self.q6(infile)

        for alpha in range(3,11):
            # fit the model
            self.q5(alpha)

            # test the alpha
            predictions, accuracy = self.q6(infile)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha

        return best_alpha

    def q8(self):
        """
        We can also "hallucinate" reviews for each rating. They won't make sense
        without a language model (for which you'll have to take CS287), but we can
        list the 3 most representative words for each class. Representative here
        means that the marginal information it provides (the minimal difference between
        F[rating][word] and F[rating'][word] across all rating' != rating) is maximal.
        You'll return the strings rather than the indices, and in decreasing order of
        representativeness.
        """
        '''
            reps = ['X','X','X']
            difs = [float('-inf'),float('-inf'),float('-inf')]
            # update lists
            for k in range(3):
                if min_dif > difs[k]:
                    del reps[difs.index(min(difs))]
                    difs.remove(min(difs))
                    difs.append(min_dif)
                    reps.append(m)
                    break

        all_reps.append(reps)
        '''

        marginals = []
        # iterate over ratings
        for i in range(5):
            # list of minimums
            minimums = []
            for m in range(len(self.F[i])):
                min_dif = float('inf')
                other_ratings = range(5); other_ratings.remove(i)
                for j in other_ratings:
                    dif = self.F[j][m] - self.F[i][m]
                    if dif <= min_dif:
                        min_dif = dif
                minimums.append(min_dif)
            marginals.append(minimums)

        reps = []
        for i in range(5):
            first = marginals[i].index(max(marginals[i]))
            marginals[i][first] = float('-inf')
            second = marginals[i].index(max(marginals[i]))
            marginals[i][second] = float('-inf')
            third = marginals[i].index(max(marginals[i]))
            marginals[i][third] = float('-inf')
            reps.append([first, second, third])


        final_reps = []
        for x in reps:
            words = []
            for word in x:
                for key, value in self.dict.iteritems():
                    if word == value:
                        words.append(key)
            final_reps.append(words)

        print(final_reps)
        return final_reps
        #return [["182", "compsci", "."] for _ in range(5)]

    """
    You did it! If you're curious, the dataset came from (Socher 2013), which describes
    a much more sophisticated model for this task.
    Socher, R., Perelygin, A., Wu, J. Y., Chuang, J., Manning, C. D., Ng, A. Y., and Potts, C. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the conference on empirical methods in natural language processing (EMNLP), volume 1631, page 1642. Citeseer.
    """

if __name__ == '__main__':
    c = TextClassifier()
    print "Processing training set..."
    c.q4('mini.train')
    print len(c.dict), "words in dictionary"
    print "Fitting model..."
    c.q5()
    print "Accuracy on validation set:", c.q6('mini.valid')[1]
    print "Good alpha:", c.q7('mini.valid')
    c.q5() #reset alpha
    print "Happy words:", " and ".join(c.q8()[4][:2])
