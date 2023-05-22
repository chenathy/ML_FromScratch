<h2>Naive Bayes brief summary </h2>

<h4> P( y | X ) = ( P( X | y ) * P(y) ) / P(X) </h4>


<ul>
    <li><h3>Gaussian</h3></li>
    <li><h3>Multinomial</h3></li>
    <li><h3>...</h3></li>
</ul>

<dl>
    <dt>Gather training data: </dt>
    <dd>Collect a set of labeled training data that can be used to train the Naive Bayes model. 
        This data should include a set of features or attributes for each example, along with the corresponding class labels.
    </dd>
    <dt>Prepare the data: </dt>
    <dd>Once you have the training data, you need to preprocess and clean it. 
        This involves converting the raw data into a form that can be used by the Naive Bayes algorithm. 
        This may include steps such as tokenizing text data, removing stop words, and converting categorical data into numerical representations.</dd>
    <dt>Compute prior probabilities:</dt>
    <dd>Calculate the prior probabilities of each class label in the training data. 
        This involves determining the frequency of each class label in the data set.
    </dd>
    <dt>Compute conditional probabilities:</dt>
    <dd>For each feature and class label combination, calculate the conditional probability of that feature given the class label. 
        This involves calculating the frequency of each feature for each class label and dividing by the total number of examples in that class.
    </dd>
    <dt>Apply Bayes' theorem:</dt>
    <dd>Use Bayes' theorem to calculate the probability of each class label given a set of input features. 
        This involves multiplying the prior probability of each class label by the conditional probability of each feature given that class label, 
        and normalizing the result to obtain a probability distribution over all possible class labels.
    </dd>
    <dt>Make predictions: </dt>
    <dd>Once the Naive Bayes model has been trained, it can be used to make predictions on new, unlabeled data. To do this, simply apply the model to the input features of the new data and choose the class label with the highest probability.
    </dd>
</dl>
 


<br>

<b>In many cases, the denominator P(X) is not required for making class predictions.  
Therefore, in practice, <u>the division by P(X) is often omitted</u> in Naive Bayes implementations</b>  
we directly compare the joint probabilities (likelihood multiplied by class priors) to determine the class with the highest probability.  
This is known as the "proportional" form of Naive Bayes, where the division is skipped.







<h3>Assumptions</h3>
<ul>
    <li>Naive Bayes is a probabilistic algorithm that uses Bayes' theorem to make predictions.</li>
    <br>
    <li>It assumes that the features are conditionally independent given the class, hence the name "naive".</li>
    <br>
    <li>The model calculates the posterior probability of each class given the input features, using Bayes' theorem.</li>
    <br>
    <li>The class with the highest posterior probability is then predicted as the output.</li>
    <br>
    <li>Naive Bayes is commonly used for classification problems, 
        where the goal is to predict the class of a new data point given its features.</li>
</ul>



