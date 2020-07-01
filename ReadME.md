## Install

pip install sortedcontainers

## Data

    * Download MNIST dataset from: https://www.kaggle.com/c/digit-recognizer
    * each image comprises 28x28 matrix of pixels
    * Flattened into a D=784 vector
    * No RGB channel (black and white); each pixel only stores the pixel intensity
    * Pixel intensities are a number in: [1,255]; We will scale to [0,1]
    * first col is labels for digits [0, 9]
    * Only use train.csv (N=42k points)

## Posterior probability thresholds and AUC/ROC 
* We use AUC to see how good a model using an ROC curve that shows the relation between P(C|X) and FPR and NPR values. 
    * The ROC curve shows how picking a different threshold for P(C|X) can impact classification results and FPR an NPR
* Another use of AUC is  when we have Imbalanced classes
    * If we have 10 samples from class 0 and 990 samples from class 1, by picking class 1 we get 99% accuracy
    * Hence AUC will produce a more useful number
    
## AUC/ROC

* Area Under Curve (AUC): is the area under ROC curve (Receiver Operating Characteristic)
* It is plot of the TPR on y-axis and FPR on x-axis
    * Typically we choose 0.5 as threshold when we want to assign P(C|X) to a class: 
    * in a binary classification if threshold > 0.5 choose class 1 else choose class 0
    * ROC curve is a plot of TPR and FPR for different settings of the threshold from 0..1
    * in binary classification we only have one threshold
* Why we are testing classification TPR and FPR with different thresholds?
    * In medical diagnosis we don't care about false positives as much as false negatives- we can tolerate false positives a little higher
    * In Roadside drug tests: we care more about false negative and it is famous for its high false positive rate
    * We use AUC to see how good a model is and to set threshold for P(C|X) for classification        

* AUC measures
    * AUC = 0.5 : Random guessing - ROC curve is a diagonal line from bottom left to top right, 
        since it covers half of the entire graph the AUC = 0.5: (0,0)->(100,100) 
    * AUC = 1 : Perfect classifier - ROC curve is bottom left to top left to top right, 
        since it covers the entire graph the AUC = 1: (0,0) -> (0,100) -> (100,100)
    * 0.5 <= AUC <=1 :  In a real problem
    
* FPR and NPR
    * TPR: True Positive rate (sensitivity)  = TP/(TP+FN)
    * FPR: False Positive rate(Specificity)  = FP/(FP+TN)    
    
* Sklearn has a function to calculate the AUC
    * sklearn.metrics.roc_auc_score(y_true,y_score)
    * y_true: accepts True labels
    * y_score: accepts output probabilities P(C|X)

## Finding the posterior probability P(C|X)
* Sci-kit learn is an ML library and its API has `fit`,`predict` and `score` methods for every classification/regression class
* Regression is predicting a real valued number rather than a category so `score` gives a different output for classification vs regression
* we use the following for both classification and regression
    * `fit(X,Y)`
    * `predict(X)`
    * `score(X,Y)`: it returns accuracy for classification and R^2 for regression
* one big advantage of logistic regression/Bayes/NNs is that we get posterior probability: p(C|X)
    * with sklearn , even decision trees and KNN get a probabilistic output
    *` model.predict_proba(X) `
    * Posterior probability helps you to see how confident you the algorithm is in its classification
    * we use it to compute: cross-entropy loss; it is just a negative log-likelihood of model prediction
    

## Train-Test Curve
Model complexity-Model prediction error: 
High complexity: very complex model (low training error - high test error)
Low complexity: very simple model (high training and test error)
mid complexity (smaller training error - small test error)

## Bias-Variance trade-off
*   Bias is a number that tells us how wrong the model is. 
    High bias means we are getting a lot of our predictions wrong
    And it usually happens when the model is too simple
*   Variance is a number that tells how complex the model is.
    High variance means there is a lot of twist and turns in your models prediction function 
*   Bias and Variance change in opposite ways in terms of model complexity.
    Under-fitting: When the model is too simple we usually have:        High bias and Low variance 
    Over-fitting: When the model is too complex we usually have:        Low bias and High variance 
    Good balance: When the model is a good representation of the data:  Low bias, Low variance
    
## Improve implementations
* Algorithmic: e.g. by finding more efficient algorithms (like searching through a sortedlist instead of a list)
* Programmatic: e.g. dot product instead of for loop over element-wise products. Use numpy which is highly efficient for some tasks

* standard deviation is the square root of the variance
* Bayes classifier is rooted in probability (e.g Gaussian)
* Bayes classifiers(and Bayes rule) treats PDFs (prob. density func. for continues) and PMF (prob. mass. func. for discrete values) similarly 
* The bell shaped curve is not a probability, its a prob. density

y: label - x: data
posterior = likelihood.prior/(evidence)
p(y|x) = p(x|y)p(y)/ p(x)
p(x,y) = p(x|y)p(y)
p(x)   = sum(p(x,y)) for all y

 
## Unbalanced classes:
When we have unbalanced classes, using prior in addition to likelihood is required
- posterior = likelihood.prior/(evidence)
- p(y|x) = p(x|y)p(y)/ p(x)

## curse of dimensionality
For high dimensional data (curse of dimensionality) probabilities approach 0
  hence its better to work with log probabilities instead
  since log() is monotonically increasing the argmax rule produces the same result. 
  Also logpdf is faster to compute than pdf
  K_* = argmax_k {logp(x|y=k} + log p(y=k)})
  
## Bayes classifier
1- For continuous inputs we use Gaussian distribution: use Gaussian models (calculate the PDFs)
    * PDF of a Gaussian = f(x) = 1/sqrt(2.pi.sig^2) exp(-(x-mu)^2/2(sig)^2)
     - scipy.stats.norm.pdf    
    * PDF of multi-component Gaussian: f(x) = 1/(sqrt(2.pi)^D .covariance )exp(-0.5(x-mu)^T. inverse(covariance). (x - mu) 
     - scipy.stats.multivariate_normal.pdf
     
2- For Boolean inputs, we can use Bernoulli distribution 
    * f(k) = (theta ^ k=1) (1 - theta)^(k=0)
3- For discrete counts we use binomial (multinomial) distribution (calculate the PMFs)
    * Unlike PDFs, PMFs return an actual probability
    * Binomial distribution: it is for counts of successes out of total number of trials 
    f(k) = (k out of n) (theta ^ k) (1 - theta)^(n-k)
    * multinomial distribution
 
## Naive Bayes Classifier
* Naive Bayes make an assumption that input are not correlated and they are all independent.
In Scikit-learn, Naive Bayes comes in 3 forms:
    * GaussianNB
    * MultinomialNB
    * BernoulliNB
    
* Naive means all input features are independent; 
    * P(X|C) = mult(p(X_i|C))
    
   * When two random variables are independent their covariance is 0
        * cov(x_i,x_j) = E[(x_i - mu_i)(x_j - mu_j)]

* Non-Naive means we just can't equate the above (no independence)
    * P(X|C) = HMMs

## Singular covariance problem 
* sometimes when you want to invert the covariance you come across singular covariance problem 
* it is a matrix equivalent of dividing by 0
* to address this we add Identity matrix times a small number like lamda=10^-3
   
## Generative vs Discriminative classifiers

* Discriminative classifiers: We start with X, we get Y
    * All classifiers output a probability. The probability we make our predictions from is a posterior probability P(C|X)
    * We call classifiers that model that probability directly like logistic regression a discriminative classifier
    * Given the Data X, these classifier learn how to discriminate between each class
* Generative classifiers: We start with Y (the class) and Model X
    * They can still allow you to discriminate between classes but main calculation is p(X|C)
    * The assumption is that each class has its own structure and therefore its own distribution of X
    * Each variable is modeled directly and you can change your model p(X|C) if results are poor
    * Advantage is that you know exactly how each variable affects result
    * Disadvantage: discriminative models usually work better
    
## Decision Trees and Maximizing Information Gain
* One key feature: We only look at one attribute at a time (Each condition checks only 1 column of X)
* `Information Entropy`: theory behind choosing best splits in Decision Trees
    * We want to choose a split that maximizes reduction in uncertainty (Entropy)
    * Information Entropy relates to variance. Wide variance means we don't know much about the data we will get
    and slim variance means we are more confident
    * Entropy is a measure of how much information we get from finding out the value of a random variable
        * Entropy = E[-log2(p)]
        * Entropy = E[-log2(p)] = H(p) = - sum(p(x)log(p(x)))
        * Information Gain (IG) = entropyBeforeSplit - entropyAfterSplit
        * where the entropy after the split is the sum of entropies of each branch weighted by 
        the number of instances down that branch.
        * Information gain = IG(Y|Split on X) =
          H(Y) - p(x in Y_left)H(Y_left|X values) - p(x in Y_right)H(Y_right|X values) =
          H(Y) - sum(p(x)log2(p(x)))
        * H(Y): it is the starting entropy, it measures the entropy of the target (or output variable) before splitting
            * E.g. for a data with two classes H(Y)=-p(target class0)log2(p(target class0)) -p(target class1)log2(p(target class1))
            * where p(target class0) = the proportion of class0 examples
            * where p(target class1) = the proportion of the class1 examples
            * Entropy tells you how homogeneous is your data 
                * entropy = 0 : when your samples are completely homogeneous 
                * entropy = 1 : when your samples are equally divided across classes
        * we stop splitting once we see zero entropy(no information change after splitting)
        * 0 <= IG(Y|split on X) <=1
        * dH/dp = 0 for finding p
        
* Implementation summary  
    
    * For discrete case: We want to find the best IG and the attribute that gives us the best IG,   
      * we loop though all the data and split the data to find a condition:
        (find a Y that goes to the left node and find a Y that goes to the right node) 
      * Then we calculate the IG, if it is better than the current best; we set it to our current best. 
      * Once we found the best attribute to split on we split the data 
      into the left and right sides then we create a Tree node for the left and right child
      and we fit those nodes to their corresponding data
    For continuous case:
       * first we sort X's for current column in order, and sort Y accordingly
       * find all boundary points where Y changes from one value to another
       * calculate information gain when splitting at each boundary
       * keep the split which gives the maximum information gain
       * to do the split we only need to consider the midpoint between any 2 sorted 
        X's because splitting anywhere else will not change the entropy
       * Only need to consider boundaries between differing labels,
       * further from boundary -> higher entropy -> lower information gain
        
    * Base cases: 
      * if the IG is 0 we gain nothing from splitting, and make it a leaf node
      * if it is a leaf node then we should just take the most likely class
      * we should avoid over fitting:
        * we can achieve 100% on training set by having a tree of arbitrary depth
        * but it will not lead to good generalization, so we should set max_depth
        * when we hit max_depth, we stop recursing, ans make leaf node
        * means every TreeNode must know its own depth, and max_Depth
      * If there is only 1 sample in the data subset, predict that sample's label
      * If we have > 1 samples, but all have the same label predict this label
        

* Maximizing Information Gain : how will we use Information Entropy to help us choose the best attributes in our data 
    * `ID3 (Iterative Dichotomiser 3)`: 
        We like to find the attribute that best splits the data based on `maximum information gain`
        Once we make that split, we never use that attribute again. All the children should choose the 
        maximum information gain based on the rest of the attributes given to split up the data. The
        
    * In our implementation in this directory however it is not required that an attribute can only be split once
      The splits that happen in ID# algorithm don't have to be necessarily binary but our approach only has binary splits
    
    * Information gain calculation: 
        * If we have a 50% chance of being in class 1 and 50% chance of being in class 2: H(Y)=1
        * Entropy: H(p) = - sum(p(x)log(p(x))) for all x values
        * Information gain = 1 - Entropy 
        
 
## Perceptron Loss Function:
* Only incorrect examples contribute to the losses
* The loss function only increases for mis-classified samples
  x_i is close to hyperplane even though it might be misclassified the 
  value of (w_T.x_i) is smaller, it means its further from the w which 
  is perpendicular to the hyperplane (smaller angel between w and hyperplane)
  the further w_i is from the hyperplane of w the bigger the loss is
  e.g. w_T.x_i is max
* L(y,y_hat) = - sum(y_i (w_T.x_i)1(y_i!=y_hat_i))   
   * 1(true) = 1
   * 1(false) = 0  
* We use Stochastic Gradient descent to minimize the loss with respect to w_i
* we take small steps i direction of dL/dw 
   * dL/dw = - sum(y_i.x_i.1)(y!=y_i)

## K-fold cross validation
* it is a popular way to choose hyper parameters
* split the dta into K parts 
* loop K times
* In each iteration take 1 part out (validation set) use the rest for training
* this approach gives us k scores(accuracies) so we can use the mean
* we can use statistical testing to check if one hyper parameter setting
  is "statistically significantly" better than another
  * from sklearn import cross_validation
  * scores = cross_validation.cross_val_score(model, X, Y, cv=K)
    
## Feature extraction and feature selection

* Hyper parameter selection using : Greedy Method 
    * build a classifier for each individual feature, pick the best via cross-validation
    * build another set of classifies all of which contain the first (best) feature
    and another one. Pick the next best one using cross.val. Now we have 2 features
    * repeat
    * which features we select is also a hyperparameter and its very important

* Hyper parameter selection using : Automatic Method
  * PCA is a dimensionality reduction technique and 
  * it doesn't require domain knowledge for feature selection
  * all the outputs are uncorrelated (no redundancy)
  * Outputs are sorted by information contained (measured by variance)
  * we can choose enough such that we retain 95% or any percentage of the original variance 
  * disadvantage it is a linear transformation
   
## Comparison to deep learning:
* NNs have many hyper-parameters

## ML Web Services
Web APIS e.g. Twitter/Instagram/Facebook/Youtube
* example code: app_trainer.py/app_caller.py/app.py

sudo pip install tornado
sudo pip install requests

* app.py uses Tornado framework
    * Create an endpoint/predict, which take in parameter "input"
    * will store entire vector of data at this argument
    * application will make prediction using model, return JSON with prediction {"prediction":k}
    * Make it a POST even though POST's technicaly should be used for requests which change the data on server
    * don't want entire vector show up in URL (would happend with GET)
* app_caller.py
* simulate how an extr=ernal application would call API
* choose digit at random, call web service to make prediction, print prediction + true label, show image
