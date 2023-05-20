<h2>Naive Bayes brief summary </h2>

<h4> P( y | X ) = ( P( X | y ) * P(y) ) / P(X) </h4>


<ul>
    <li><h3>Gaussian</h3></li>
    <li><h3>Multinomial</h3></li>
    <li><h3>...</h3></li>
</ul>

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



