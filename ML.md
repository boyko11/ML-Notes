# Machine Learning Notes

## Decision Trees
* Algo
	* 1. Pick the best attribute - Best: splitting things roughly in half(minimize entropy maximize mutual information)
	* 2. Ask question
	* 3. Follow the path
	* 4. Go to 1
	* Until you ge the answer
* Boolean AND:

   	 A
  1 / \ 0
   B   F
1 / \ 0
 T   F

* Boolean OR - mirror of AND
* Boolean XOR - full tree
* N-Or needs N nodes, so does N-And. both are knowns as ANY - the number of needed nodes is O(n) with respect to the number of attributes
* N-Xor - even or odd parity; odd parity if the the number of T is odd, then it outputs T, othrwise F
* Need 2^(n-1) nodes for N-Xor tree - aka HARD, since it is exponential
* To solve problems with DT, we look for 'ANY' type of problems over 'HARD' type of problems
* Number of all possible decision trees over binary attributes with binary outcomes:
* n attributes 2^n possible node set-ups, this is without the all possible outcomes
* how many possible outcomes are there, this si the same as asking, how many number can we represent with a 2^n binary string
* or how many possible combinations are there for 2^n binary variables - 2^(2^n) =>
* The Hypothesys space of decision trees is very expresive
* This also means that you MUST have some clever way to search through them
* ID3
	* Loop:
		* Pick A <- best attribute
		* Assign A as decision attribute for node
		* For Each Value of A create a descendent on node
		* Sort training example to their corresponding A value leaves
		* IF Perfectly classified - STOP
		* ELSE go back to Loop without considering this attribute A anymore
* Best Attribute - the one with the highest information Gain
* Gain(S, A) = Entropy(S) - Sigma_v( (|S_v|/|S|) * Entropy(S_v) )
* Entropy - measure of randomness
* -Sigma_v( p(v) * log( p(v) ) )
* The best attribute is the one with the maximum gain
* Restriction Bias - what kind of restriction do you have on your hypothesis, e.g. when doing DT, we are ONLY considering DT, not lines, poly curves, etc
* Preference Bias - what sort of hypothesys from the possible  hypothesis set we PREFER, that is at the heart of INDUCTIVE BIAS
* ID3 Bias
* Prefers good splits near the top - given two trees that perfectly fit the training data, ID3 would pick the one with better splits at the top
* Prefers trees that produce the correct answer
* ID3 Prefers shorter trees. This comes naturally from good splits at the top - you gonna take trees, that separet the data well early(at the top), over tress that produce useless splits, hence growing the tree size unnecessarily
* Continuous Attributes
* Does it make sense to repeat an attribute along a particular path on the tree
	* If it is a continuous one, yes, for example you split on age -50+ at the top, but then you split on another age value farther down
	* If it is a boolean Attribute, NO, it doesn't, since you've already split on it some time ago on the upstream, you would have only instances with the sane value for that attribute, so there really isn't anything to split one, i.e. if you split on it it would just create only one child node, which won't help. The latter would be automatically picked up by ID3, because this would be the worst attribute to split on
* ID3 - when do we really stop?
	* When everything is classified correctly
	* No more attributes to try
	* Does it makes sense to trust our data and fit it perfectly?
	* NO - this is overfitting - you fit even the noise
	* Complex, big trees tend to overfit
	* CrossValidation - you set some data aside and you see which tree does best on it
	* You could also check against the validation set anytime after you grow the tree by one level.
	* If you start getting worse scores on the validation set, it is time stop growing that tree. Which is just computer speak, in real life, you want to let real trees grow, cuz we need more of those
	* You could also do pruning - you let ID3 build the full tree, then start removing a subtree at a time and see what happens with the validation score. You keep pruning until the validation score stops improving.
	* There are variations of that. You could keep pruning until the validation set deteriorates significantly, for some definition of significant. The idea is to create a smaller, less complex tree, that generalizes better.
	* Regression Trees 
		* How do we split - variance? SSE, correlation
		* Output: average, local linear fit, etc

## Regression and Classification

* Lin Reg
* Best Fit line - the line that minimizes the squared error of y distances - least squared error
* Use Calculus to find it - gradient descent of the Error function:
* E(m, b) = Sigma_i( (y_i - (mx + b))^2 )
* We are using the SSquaredE, because it is smooth and well behaved in a Calculus sense
* It has also been proven that using SSE results in the Maximum Likelihood Hypothesys
* constant line:
	* E(c) = Sigma_i( (y_i - c)^2 )
	* Derivative_wrt_c( E(c)) = Sigma_i( 2*(y_i - c)* -1)
	* Derivative_wrt_c( E(c)) = -2*Sigma_i(y_i) + 2*Sigma_i(c)
	* Derivative_wrt_c( E(c)) = 0 at the min =>
	* Sigma_i(c) = Sigma_i(y_i) <==>
	* n * c = Sigma_i(y_i) =>
	* c = Sigma_i(y_i) / n    
	* this is the mean y value - the best fit horizontal(constant line) is the average of all y's
* Order of polynomial:
	* 0 - constant
	* 1 - line (mx^1 + b)
	* 2 - parabola c2 * x^2 + c1 * x^1 + c0
* You can fit as higher of polynomial as the number of training records - but this would overfit - fit the training data too well - won't be general enough
* As the number of degrees increase, although you'd get a super low and possibly zero training error, the testing error at some point would start going up
* Trying to find the best c's
* c0 + c1*x + c2*x^2 + c3*x^3 = y
* [ 1  x_1 x_1^2 x_1^3]   *   [ c0      = y1
  [ 1  x_2 x_2^2 x_2^3]         c1        y2
  [ 1  x_3 x_3^2 x_3^3]         c2        y3
  [...................]         c3 ]     ...

* X * w = Y
* X_t*X*w = X_t*Y
* Inverse(X_t*X) * X_t*X * w = Inverse(X_t*X) * X_t * Y
* w = Inverse(X_t*X) * X_t * Y
* X_t * X - has nice properties - it is invertible and it does the right things in terms of minimizing the least squares - it does it as a projection

* Errors - data is noisy; sensors are noisy; data might be tamperred with, intentionally or not; transcription errors; unmodelled influences, variables that matter aren't captured
* How to minimize modelling the errors, rather than the underlying function:
* Cross Validation:
* The Test is Representative -  a stand-in for the distribution of the real data
* The data is assumed to be IID - Idependent Identical Distribution - the Training set , the Testing set and the real world data are assumed to be from the same distribution
* We are trying to use a model that is complex enough to fit the data well, yet not to complex so it does well on the test data and generalizes well to the unseen data
* If we don't have access to the Test Set - we can hold out some of the train set and use it as a stand-in for the test data - cross-validation set
* CrossValidation - k-fold - break the train set to k folds; 
* for each fold, fit the model using this fold as the test set and the rest of the folds combined as a training set
* average the test error from each testing fold - that's your cross-validation error
* select the model for the fold that caused the lowest error
* CV learning curve for the polynomial curve fitting - falls and gets close to the training error, but then rises as the degree of the polynomial rises

## Neural Nets

* Perceptrons are linear function that always that computes hyperplanes
* Perceptrons always compute these half-planes
* There is a dividing line equal to the threshold
* anything above(or to the left) positive, anything below(or to the right) - negative
* w1*x1 + w2*x2 >= theta - positive
* w1*x1 + w2*x2 < theta - negative
* AND, OR and NOT are representable as perceptron units
* show weights:
* AND X1: 1/2  X2: 1/2  Theta: 3/4
* OR  X1: 1    X2: 1    Theta: 3/4
* NOT X1: -1   Theta: 0
* XOR: OR - AND ( 0 1 1 1 ) - ( 0 0 0 1) = ( 0 1 1 0 )   X1: 1     X2: 1     AND: -2   Theta: 3/4
* Standard Gradient Descent vs Stochastic Gradient Descent
	* In Standard the error is summed up over all training records, then the weights are updated
	* In Stochastic the error is calculated for the individual training instance and the weights are update immediately - for each training instance
	* Because it uses true gradient, standard gradient descent is often used with larger learning rate(step size) per weight udate than Stochastic Gradient Descent.
	* Stochastic GD could avoid falling into local minima, because its direction of ascent varies more than the Standard GD

* Perceptron rule
* y_hat = Sigma_1_through_n(w_i*x_i) >= theta, we subtract theta from both sides
* y_hat = Sigma_i(w_i*x_i) >= 0, with the first i = 0, x0 = 1, notice the >= 0, if the left side is >= 0 y_hat is 1, otherwise y_hat is 0
* w_i = w_i + delta_w_i
* delta_w_i = alpha(y - y_hat)*x_i
* y - label, y_hat - output, alpha - learning rate
* if the output is correct - no weight change
* if the label(y) is 1, but the output(y_hat) 0, this means that the current weight is not large enough, so we have to bump it up, hence delta_w in direction of x_i
* if the label(y) is 0, but the output(y_hat) 1, this means that the current weight is too large, so we have to take it down, , hence delta_w in direction opposite of x_i

* Gradient Descent - more robust to non-linear separability
* No thresholding
* E(w) = 1/2 * sigma_m((y_m - (sigma_i(x_m_i * w_i)))^2 )
* some derivation later
* derivative_of_the_error_wrt_w_i = sigma_M(y - a)(-x_i)

* perceptron guarantee: delta_w_i = alpha(y - y_hat)*xi - finite convergence guaranteed when data is linearly separable
* gd: calculus        : delta_w_i = alpha(y - a)*x_i

* Perceptron vs Delta Rule
	* The Perceptron will use the thresholded output, the delat rule will use the untrhesholded output
	* If the data is linearly separable, the perceptron rule converges after a finite number of iterations
	* The delta rule does NOT require data be lineraly separable, but it only converges asymptotically towards the min error, possibly requiring unbound amount of time

* Perceptron's discontinuous threshold function makes it indifferentiable, hence not suitable for gradient descent

* So we turn the thresholding into a sigmoid function instead of sigma_i(w_i*x_i) >=0 , we do 1/(1+e^-(w_i*x_i))

* Now we can differentiate the thresholding function

* Derivative of sigmoid = sigmoid(w*x) * (1 - sigmoid(w*x)) - it approaches zero for larger positive numbers; it approaches zero for smaller negative numbers

* Backpropagation: computationally beneficial organization of the chain rule

* Momentum term in backprop - helps to get through local minima(though it may also get you through a global minimum :); helps to ge through flat regions; helps to get trough steep regions faster

* Backprop over multilayered networks is only guaranteed to converge toward some local minimum, because the Error surface may contain many local minima

* When in local minima with respect to one of the weights, you may not be in local minima with respect to the other weights. The more weights, the more escape routes away from this single dimension's local minima

* The more nodes, the more layers, the more cmplex the network, the more likely it is to overfit => regularization - keep the weights within reasonable range

* complexity is also related to the scale of the weights - you can overfit because of some weights being too large

* Restriction Bias - Perceptrons are linear, so they restrict us to planes; beyond that NNs are not very restrictive in terms of what function they can represent:
* Boolean - network of threshold like units
* Continuous  - as long as it is connected(no jumps) - we can represent it with a single hidden layer, as long as there are enough units in that layer. We can think of this as each hidden unit being in charge of one small patch of the continuos function being modeled. All the patches get stiteched together at the output
* Arbitrary - just add one more hidden layer, which would account for the jumps, that couldn't be handle at the continuous layer. more or less you take multiple continuous networks as your input and you stitch them together

* Preference Bias:
* Prefer simpler networks
* Prefer smaller weights
* Start with random small weights
* Random - just try different things so you don't get stuck in the same local minima
* Small - so you avoid overfitting
* Prefer correct over incorrect(not sure that won't be a preference for any algo), but all things being equal, prefer the less complex network with smaller weights

* Given there isn't much restriction, we can come up with arbitrarily complicated networks - this may result in overfitting.
* To combat that, we use something of the likes of pre-prunning - you come up with sufficintly robust, but not too complicated NN architecture, e.g. hey I'm gonna have only two hiddne layers, the first will have 16 units, the second 8
* Or we can use CV to decide how many hiddne layers to use, how many nodes to put in each layer, or we can use it to decide when to stop training, because the weights are not too large

* Inductive bias - smooth interpolation between data points - given two +ve examples with no -ve examples between them, BACKPROP tends to label the points in between as positive as well

* Backpropagation is susceptible to overfitting the training examples at the cost of decreasing generalization accuracy over the unseen examples

* Given enough weight-tuning iterations, backprop will create complex decision surfaces that fit noise in the training data => low training error, high generalization error

* TO counter that  - weight decay(regularization) - keep weights small - bias against complex decision surfaces

* Plot validation error vs training error learning curves - the point of the lowest validation error is the number of iterations to use in future training

* Overfiiitng is most severe for small training sets - use k-fold cross-validation in this case 
	* break the data down to k equal parts
	* from 1 through k
		* index is your validation set, all other partitions your training set
		* train the network and record the number of iterations with the lowest cv error
	* find the mean number of iterations with lowest cv error
	* train the network for this mean number of iteration with ALL DATA



## KNN
* 1 downside of remembering is no generalization, you may not find a record that exactly matches the query point
* 1-NN will likely overfit, especially if the query point and the training point are identical - you may model the noise
* It is also possible that the query matches different records - same feature data, but different labels
* Many techniques construct function approximation only on the neighborhood of the query.
* Not generalizing over the entire instance space could be very beneficial when the target function is very complex
* Cost of classifying can be high - all computation take place at classification time.
* KNN and similar, usually considers all attributes. If the target concept depends only on some, but not all attributes, similar instances might be far away in eucledian space
* Locally Weighted Regression can be viewed as generalization of KNN
* KNN never forms explicit general hypothesis for the target function
* KNN classification - you count your neighbors and classify as the highest count for a class. Regression - you take the mean of the neighbors.
* KNN - you can also weight by distance. The closer the neighbor is, the higher the weight in the vote or in calculating the mean. One possible weight calculation - 1 / distance^2. If the distance is 0, assign the label of the 0 distance training instance as a class(or projection) for the query record. If more than one zero distance training records - majority(or mean)
* Weights allow considering all training records. If all training records are considered - tha algo is called Global, if only the k-nearest records are considered, the algo is called Local. Global methods would be slower, since the computation would be more expensive.
* Taking weighted average over k-nearest neoghbors, can smooth out the impact of isolated noisy records
* Inductive bias - similarity is assumed by closeness in eucledian space
* Nearest nighbor methods are susceptibe to the curse of dimensionality - the more attributes we add, the further away in eucledian distance two similar distances my become, if the attributes we add don't have much to do with similarity between the instances.
* To address the above problem, we could weight the attributes. You determine weights, either by domain knowledge, or this could be treated as a separate optimization problem where we minimize error for the weights in determining similarity between instances with known labels.
* Assigning different weights to attributes could be viewed as stretching(weight>1) or squashing(weight<1) the axes
* Since computation at query time is expensive, indexing training instances is very important. One method for indexing is kd-tree
* Regression - approximating a real-valued target function
* Residual - error in aproximating the target function
* Kernel function - the function that determines the weight of the training instance

* RBFs can be viewed as two layer network, where the first hidden layer neuron values are Gausians(Kernels) and the second layer are the weights for these Gausians in the nearest neighbor computation.
* RBF networks globally apoximate the target function to a linear combination of many local Kernel functions. The network can be viewed as a smooth linear approximation of many local aproximations to the target function.
* RBFs are EAGER learners - the network is learned prior to query time
* An advantage to RBF networks is that they can be trained more efficiently than backprop networks, because the the input layer and the output layer of rbfs are trained separately
* Case Base Reasoning (CBR) does not use Eucledian distance - instance representation is richer symbolic description and similarity measure is more elaborate

* Lazy methods may consider the query instance when generalizing beyond the training data
* Eager methods canNOT consider the query instance
* Because of the above two - the dinstinction between lazy and eager is related to the distinction between local and global approximations of the target function
* RBF's local approximations are not specifically targeted to the query point, because it is an eager algo => it is still a global and eager algo
* Eager algos are more restrictive because they must commit to a signle hypothesis that covers the entire instance space
* Even local approximation eager methods such as RBF, do not allow for customization to unseen query instances

* Preference bias - the thing that encompasses our belief about what makes a good hypothesis
* KNN preference bias
	* Near Points are similar
	* Smoothness - averaging
	* All Features matter equally

* Curse of dimensionality
* As the number of features or dimensions grows, the amount of data we need to generalize accurately grows exponentially
* If you want your neighborhood to be small, you'd have to fill it up with data x's between 1 and 10 - 10; x1 and x2 between 1 and 10 - 10^2, 3D - 10^3 - exponential growth
* You are better of giving it more data, than giving it more dimensions
* Free Lunch Theorem: An learning algo that you create, is going to have the property that if you average over all the possible instances, it is not doing any better than random
* A practical way of thinking about this is: If I don't know anything about the data that I need to learn over, it doesn't matter what I do, because there are all kinds of possible data sets
* However, if I have the domain knowledge, I can use that to choose the best learning algorithm


## Boosting

* Ensemble Learning general form:
	* Learn over a subset of the data - this generates a rule
	* Keep learning over different subsets of the data and generate rule for each
	* Combined the learned rules into one complex rule
* Example
	* Pick data uniformly at random - run a a Learner on it
	* Average the results from each learner
* Ensemble zero order polynomial, each learner gets a single data point - you combine all learners - they get you the mean - this just like N-NN
* Learning a bunch of third order polynomials on a buch of random subsets and averaging these third order polynomials does better than
* Learning a single third order polynomial on the entire set
* Because: It's sort of like online cross-validation. Not trying to fit all possible points by using a subset, it manages to find the important structure, rather than being misled by any of the individual points.
* You don't get trapped by a few noisy points, you sort of average out all the variance and the differences
* This is called Bagging or Bootstrap Aggragation

* Error: Pr_D( h(x) != c(x))
* Every time we see a bunhc of examples, we are gonna make the harder ones more important to get right
* You work with probability distros for the records - the ones we get wrong, we assign a higher probability to get drawn for the next round
* Weak Learner - no matter what the distro over your data is, it will do better than chance - Pr_D( h(x) != c(x)) <= 1/2 - epsilon - no matter the distribution, epsilon some really small number
* Another way of saying this - you always get some information from the Learner
* If there IS a distribution for which none of the hypothesys will do better than chance, this means There is NO WEAK learner.
* Having a WEAK learner is a pretty STRONG condition - you have to have many hypotheses that do well on many examples
* Boosting high level algo:
	* Given training examples (x_i, y_i) y_i is -1 or 1
	* For t = 1 to T
		* Construct D_t
		* Find WEAK classifier h_t(x) with small error -  epsilon_t = PR_D_t(h_t(x) != c(x)) < 1/2
	* Output final H(x)

* D_t - start with uniform, then
* D_t+1(i) = (D_t(i) *  e^(-alpha_i * y_i * h_t(x_i)) ) / Z_t, where alpha_t = ( (1/2) * (1 - epsilon_t)/epsilon_t)
* if y_i and h_t(x_i) disagree, this will drive D_t(x_i) up, if they agree, it will drive D_t(x_i) down
* instances that were incorrectly classified will have a greater probability of being drawn in the next round
* instances that were correctly classified will have a smaller probability of being drawn in the next round
* H_final(x) = sign( Sigma_t( alpha_t * h_t(x)))  - alpha_t - greater if you error is lower = > more accurate learners get higher weight

* Finding many rough rules of thumb, is easier than finding a single highly  accurate prediction rule
* When there IS a distribution for which none of the hypotheses does better than chance, there is NO WEAK learner for this hypothesys space(All the learners are weaker then weak - error >= 1/2)
* You can combine simple things to get comlicate things, e.g. you combine linear separators as a weighted average you get a non-linear separator - that's why boosting works
* General feature of ensemble methods: if you try to look at some particular hypothesys class, because you are doing weighted average over hypotheses drawn from that hypothesys class, the final combined hypothesys is at least as complicated as the original hypothesys class.
* Part of the reason that we are getting a non-linear final hypo is because at the end we pass the weighted average through a non-linear funtion - the SIGN
* we are able to get simple hypos, add them together in order to get a more complicated ones - the simple ones are local, sort of like localized linear regression in KNN
* You are able to be more expressive, even though you are using simple hypotheses, because you are combining them in some way
* Boosting is agnostic to the learner - you can plug in any type of learner(as long as it is a weak learner)
* Boosting does not alway overfit
* Boosting's testing learning curve may never go up, the error may continue decreasing

## SVM

* Find the line of least commitment - the line that gives as much space as possible from the boundaries
* you taking a new point, projecting onto a line and lokking at the value that comes out - y = w_transpose*x + b
* equation of the decision boundary = w_transpose * x + b = 0, since this IS the decision boundary, projection a point from it, onto it(itself) will be 0 - neither positive, nor negative
* 2/||w|| - the distance between two points on the margin lines connected with a line perpendicular to both
* SVM solution - maximize 2/||w|| while classifying everything correctly - y_i(w*x_i + b) >=1; y_i is either 1 or -1; this combines w*x + b >= 1 for the positive and wx + b <=-1 for the negative
* an easier problem to solve is (||w||^2)/2, because it is quadtratic optimization problem and these are well studied
* The above one somehow turns into w = SIGMA of alpha_i * y_i * x_i; when we recover the w's, we can get the b(follows from the line above the line above)
* most of the alphas end up being Zeros, so only few alphas matter - the ones that are closest to the margin lines - these are called the Support Vectors
* W(alpha) = Sigma_i(alpha_i) - 1/2 * Sigma_i_j( alpha_i*alpha_j * y_i*y_j * (x_trans_i*x_j) )  - meaning:
* alpha_i*alpha_j - figure out which points matter for your decision boundary
* y_i*y_j - how do they relate to one another in terms of their output labels with respect to...
* x_trans_i*x_j - how similar they are to one another
* Kernel Trick - we can substitute the (x_trans_i*x_j) with any measure of how two points are alike or how they are different - in the case of a origin centered circle (x_trans_i*x_j)^2
* You can't just use ANY transfromtaion, but in practice you can use ALMOST anything:
* for any function that you use, there is some transformation into a higher dimensional space that is equivalent, though it may trun out you need an infinite number of dimensions to represent it
* K - Kernel the W(alpha) can be written as
* W(alpha) = Sigma_i(alpha_i) - 1/2 * Sigma_i_j( alpha_i*alpha_j * y_i*y_j * K(x_i, x_j) )
* Kernel - way to measure similarity, in the Linear SVM, K returns (x_trans_i dot x_j) - the dot product would be large positive if they are similar, large negative if dissimilar
* Kernel - way to inject domain knowledge into the SVM
* K = (x_trans_i dot x_j) - Linear
* K = (x_trans_i dot x_j)^2 - Origin Centered Circle
* The above two can be generalized to:
* K = (x_trans_i dot x_j + c)^p - polynomial kernel
* K = e^-( (||x_i - x-j||^2) / (2*sigma^2) ) - radial basis function kernel
* K = tanh(alpha*x_i*x_j + theta)   - sigmoid like
* Kernels need to satisfy the Mercer condition
* We can think of SVMs as being EAGER LAZY learners or ONLY as LAZY as Necessary, because the classifier was dependent on just subset of the data - the support vectors
* Eager - you still train using all the data so you find your support vectors, but when classifying only the support vectors matter so Lazy in that sense
* As you add more and more weak learners, the boosting algo becomes more and more confident in its answers, it effectively creates a larger and larger margin
* Larger margin minimizes overfitting
* If Boosting uses ANN with many layers and neurons, then boosting may overfit.
* It is because ANN is prone to overfitting - if ANN fits the data perfectly, all the training examples will have the same weight for the next iteration
* SO the next iteration will just produce the same ANN. In the end you may end up with an enseble of identical overfitting ANNs
* weighted linear combination of the same thing, is the thing itself. The weights would also be the same since hypothetically the ANN fit the data perfectly
* If you have an underlying weak learner that overfits, it is difficult for boosting to overcome that
* Boosting can also overfit in the case of pink noise(uniform noise)
* SVMs are Linear Learning Machines that
	** Use a dual representation
	** Operate in a Kernel induced feature space - f(x)=∑αiyi φ(xi),φ(x) +b  is a linear function in the feature space implicitly defined by K
* Bad Kernel would have a mostly diagonal Kernel matrix - all points orthogonal to each other - no structure, no clusters
* In SVMs we fight overfitting by trying to find the classifier with the max margin
* As we add more and more weak learners, the confidence of the classification grows: H_final(x) = Sign(Sigma_t(alpha_t * h_t(x)) / Sigma_t(alpha_t))
* As we add more and more weak learners, the error may stay the same, but the confidence of prediction grows
* i.e. if we measure confidence of how close H_final is to -1 or 1, the negative predictions will be getting closer to -1, the positive to +1
* i.e the margin between the positives and negatives will grow
* Large margins tend to minimize overfitting
* Boosting will overfit if the underlying weak learners overfit, e.g. NN may fit the data perfectly, then all the records will get the same weights in the distro for teh next round,
* Then you end up sampling the same or very similar subset of the data, then you end up fitting the same or very similar NN
* In the end, the weighted sum of the same thing is that thing and you overfit
* Bostting also tends to overfit in the case of 'Pink noise' - pink noise is uniform noise

## Comp Learning Theory

* Sample Compexity - the fewer samples an algo it needs in order to generalize effectively, the better it is at learning
* Defining Inductive Learning:
	* Learning From Examples
	* Probability of successfull training - 1 - delta
	* Number of examples to train on - m
	* Complexity of hypo class - comlexity of H
	* Accuracy to which target concept is approximated - epsilon
	* Manner in which training examples presented - batch(here are all training samples)/online(here's one example - give me a label, there's another one...)
	* Manner in which training examples selected
		* Learner asks teacher - here is X, teacher responds with c(X)
		* Teacher chooses Xs and their corresponding c(X) and gives them to the Learner
		* Fixed Distribution - x, c(x) chosen from D by nature
		* Evil worst distribution

	* The teacher could tell the learner to ask the only question that matters - e.g. IS the person Charles Isbell? - learner turns around and asks, teacher responds with an 'Yes'
	* Teacher has the advantage of knowing the answer
	* If the Learner initiates to exchange(hey teacher, what's the label for x), the learner needs a question that elminates as many hypotheses as possible. similar to DT, find the question that splits the labels best - log(n)
	* Back to Teacher providing questions - there usually is a constraint, you can't just provide the ultimate question
	* k-bit input strings, if attribute can take on 3 possible values - 3^k possible hypos
	* the teacher can give some positive examples, which would help the learner detrmine what's irrelevant - bit indices that differ are irrelevant
	* the the teacher can provide some negative examples - which help the learner finalize what is relevant - try flipping each bit fro the bits you currently consider relevant - if the label doesn't change - the attribute is relevant
	* Learner with constraint queries - in the worst case, you'd have to enumerate all possible hypos before finding the only one correct one -  for a k-bit string there are 2^k possible ones. The learner is linear in the |H|, but exponential in K
	* Learner with mistake bounds
	* The learner asks the question about X, but it also guesses the answer - the teacher informs the learner if the guess was correct
	* The learner would have some bound on mistakes it could make until it figures out the correct hypo
	* If the learner guesses wrong - it better learn a lot from that
	* if it guess right - it doesn't have to learn too much from it
	* even if the learner needs to explore 2^n hypos - it will never make more then k+1 mistakes - because it will learn if a certain bit matters from each mistake
	* mistake bound - how many misclassifications can a learner make over na infinite run


* Consistent Learner - learner consistent with the data - learned to produce the training labels
* Version Space - all the hypotheses consistent with the training data(all the correct hypotheses with respect to the training data)
* True error vs Training error
	* Training error is the percentage misclassified from the training set
	* True error is the probability that the classifier will incorrectly classify data drawn from the true distribution
	* It's OK to misclassify examples you will almost never see - since they are so rare, they will not impact the true error much
	* On the other hand you want to correctly classify common examples, so you have a low true error
* Version Space is epsilon-exhausted iff all the hypotheses in the Version space have error <= epsilon
* k - number of hypos with error > epsilon
* Hausler Theorem:   Probability there's at least one hypo in the space with error > epsilon <= k(1-epsilon)^m - upper bounded by |H|(1-epsilon)^m
* Hausler Theorem: how much data do we need to knock out the hypos that have error > epsilon - |H|*(1 - epsilon)^m <= |H|*e^(-epsilon*m)
* Hausler Theorem: m >= 1/epsilon * (ln(|H|) + ln(1/delta)) - upper bound on training samples that the Version Space is NOT epsilon exhausted - 
* If you know the size of your Hypothesis Space and you know what your target epsilon and delta are, from the above line, you'd know how many samples you need so you can be PAC
* If the target concept is not in H - agnostic learner - needs to find the hypo that is closest to the target concept - 
* m >= 1/(2*epsilon^2) * (ln(|H|) + ln(1/delta))

## VC Dimensions

* Size of Hypothesys Space - infinite in many cases - ANN, Linear Regression, SVM, DT with continuous attributes
* Syntactic-write(infinite) - sometimes it is posssible to convert it to  Semantic - meaningfully different - eliminate lots of hypothesis based on the observed data, e.g. data between 1 and 10, if you have to threshold it, do not consider hypos where the threshold param is less than 0 or greater than 10
* Syntactic - all the possible hypotheses one can write
* Semantic - all the possible hyotheses that are meaningfully different from one another, e.g from previous example theta < 11, theta < 12, theta < 13, are NOT meaningfully different since they will all produce the same answer
* Power of Hypothesis space - what is the largest set of Inputs that the hypothesis class can label in all possible ways (shatter)? - the numeric answer of this is called VC Dimension
* VC Dimensions relay to the amount of data needed to learn in hypothesis class
* As long as VC Dimensions are finite, even if the hypo class is infinite, we will be able to say, how much data we need to be able to learn
* Examples how many ways can you shatter inputs to a function which returns interval membership:
* 1 - 2 ways = the number of all possible ways
* 2 - 4 ways = --||--
* 3 - all possible ways = 2^3, but you can't shatter three points into a T F T arrangement, so you can't shatter an input set of 3 to 8 possible outcomes => VC Dimensions = 2
* It's easier to get a lower bound, less hypothesis - show that there EXISTS some set of points that can shatter the hypothesys class
* For upper bound you have to prove that NO example exists that can shatter the hypo class
* The above two statements are translated to:
* Can Shatter: There EXISTS a set a points of certain size, such that for all possible labelings for the points in that set, there exists a hypo that works for this labeling
* CanNOT Shatter: For All Possible arrangement of points, there exists a labeling, which no existing hypo can make work
* VC dimension is often the number of parameters - threshold in 1D - 1 (theta); interval in 1D - 2 ([a, b]) separator line in 2D - 3 (w1x1 + w2x2 +b);
* 3 dimensions - 4; d-dimensional hyperplane - d+1 - d the weights for each dimensions + theta(the threshold)
* m <= (1/epsilon)(8*VC(H)*log2(13/epsilon) + 4*log2(2/epsilon))
* The VC dimension concept it doesn't require hypothesys class be infinite, it's just that an infinite hypothesys class requires a VC dimension
* What is the VC dimension of a finite hypothesys class: 
* if d = VC(H) => there are 2^d distinct hypothesys => 2^d <= |H| => d = log2(|H|)
* A finite hypo class, or a finite VC dimension give us finite bounds, hence make things PAC learnable
* H is PAC learnable iff the VC dimension is finite

## Bayesian Learning

* Learn the most probable hypo given data and some domain knowledge argmax_h Pr(h | Data)
* Bayes Rule - Pr(h | Data) = ( Pr(Data | h) * Pr(h) ) / Pr(Data) derived from Pr(h, D) = Pr(h | D) * Pr(D) = Pr(D | h) * Pr(h) => Pr(h | D) = ( Pr(D | h) * Pr(h) ) / Pr(D)
* Pr(D) - prior on the data - we typically ignore it - Sigma_i(P(D| h_i)) - normalization factor
* Pr(D | h) - the likelihood we will see the data, given that the hypo h is true = the probability we will see the training labels for the specific hypothesis
* Pr(h) - prior on the hypothesys - capture our prior belief that this hypo is likely or unlikely compared to other hypos; This is our domain knowledge
* Bayesian Learning:
	* For each h in H
		* calculate Pr(h|D) ~ Pr(D|h) * Pr(h)
	* h_map = argmax_h Pr(h|D) map - max aposteriori - largest posterior given all your priors
	* h_ml = argmax_h Pr(D|h) (we dropped the Pr(h)) ml = maximum likelihood - we use this if we assume all h are equally likely(come from an uniform distribution), so Pr(h) will always be the same in the Pr(D|h) * Pr(h) calculation => Pr(h|D) ~ Pr(D|h)
	* h_ml is h_map if the prior (P(h)) is uniform
* The above formulation is nice in math terms, but it is not computationally feasible for large hypothesys spaces
* in a noise free world - Pr(h|D) = 1/|VS| - https://classroom.udacity.com/courses/ud262/lessons/454308909/concepts/4733385550923
* Noise free world, the true concept is in the Hypo space, all hypos are equally likely:
	* Pr(h) = 1 / |H|
	* Pr(D|h) = 1 if h gets all labels correctly, 0 otherwise
	* Pr(D) = Sigma_h_i_over_hypo_space(Pr(D| h_i)*Pr(h_i)) - Pr(D| h_i) will be 1 if h_i is in the Version Space(VS), 0 otherwise => Pr(D) = Sigma_h_overVS(1 * 1/||H||) = ||VS||/||H|| =>
	* Pr(h | D) = (1 * (1 / ||H||) ) / (||VS|| / ||H||) = 1 / ||VS|| if h belongs to the VS == just pick something from the consistent set of hypos
* h_ml = argmax_h Pr(h | D) = argmax_h Pr(D | h) = argmax_h Product_i(label_i | h) = argmax_h Product_i gausian formula for label_i
* The log of a product is the same as the sum of the logs, and the log of e to something is just that thing hence:
* argmax_h Product_i gausian formula for label_i = argmax_h Sum_i(-1/2* (label_i - h(x_i))^2) / sigma^2), we just took a log of the product to make things easier
* argmax_h Sum_i(-1/2* (label_i - h(x_i))^2) / sigma^2) = -argmax_h Sum_i( label_i - h(x_i))^2 )
* -argMAX_h Sum_i( label_i - h(x_i))^2 ) = argMIN_h Sum_i( label_i - h(x_i))^2 ) - sum of squared errors giving us the ML hypo is based on Bayesian Learning
* minimizing the sum of squared errors between the labels and the predicted labels gives us the ML hypothesis
* assumptions for ML to work - uncorrelated, independently drawn, gaussian noise with mean zero
* when you are minimizing the SSE, you are assuming the data that you have is corrupted by gaussian noise, if that's not the case, you are doing the wrong thing 

* The maximum apriori hypo is the one that minimizes error and minimizes the size of your hypo:
* h_map = argmax_h Pr(D|h)*Pr(h) = argmax_h ( lg(Pr(D|h) + lg(Pr(h))) ) = argmin_h( -lg(Pr(D|h)) - lg(Pr(h)))
* the above is the same as argmin_h(length(D|h) + length(h)) - The maximum apriori hypo is the one that minimizes error and minimizes the size of your hypo
* In other words - you want the simplest hypo that minimizes your error
* The name of all this is Minimum Description Error
* there is often a trade-off between the two terms - if I get more complected(lengthy) hypo, I can drive the error down
* I could sacrifice accuracy for a simpler(less lengthy) hypo
* The best hypo is the one that minimizes the error, without paying to much price for overcomplicating the hypo
* an NN example would be big weights - we gonna need more bits(more lenght) do describe larger weights, the larger the weight, the more we overfit
* in NNs The complexity is not in the number of parameters directly, but in how many bits you need to represent the value of parameters - 
* I can have 10 params and they are all binary => I need 10 bits; if this 10 params are arbitrary large numbers instead, I'd need arbitrary many bits to represent them
* Bayes Optiml Classifier -  on average you canNOT do any better than doing a weighted vote of all the hypotheses according to the hypothesis given the data
* Bayesian Learning gives the ability to talk optimality and gold standards

## Bayesian Inference
* Topological Sort Graph must by a directed asyclic graph
* The fewer the parents the more compact the distribution
* Marginalization: P(x) = Sigma_y(P(x, y))
* Conditional Prob: P(x, y) = P(y|x)*P(x) = P(x|y)*P(y)  if x and y are independent of one another = P(x, y) = P(x) * P(y)
* Bayes Rule: P(y|x) = ( P(x|y)*P(y) ) / P(x)

* Naive Bayes:  P(V|a1, a2,.....,a_n) = ( Product_i(P(a_i | V)) * P(V) ) / Z ; Z is normalization over P(a_i | V)
* MAP(Max aposteriori) CLass = argmax_v P(V) * Product_i(P(a_i | V))

## Randomized Optimization
* X belongs to the Reals - take derivative of the function and set it to Zero, the x solutions would be the optimal values
* Optimization Approaches:
	* Generate and Tests - feasible only for small input spaces; it also helps for some crazy functions - it may be easier to just plaster it with possible values, then try to reason about it
	* Calculus - can do, but assumes the function is differentiable and we can solve for its derivative equals 0
	* Newton's method - gradient descent(or ascent): guess a position, calculate the derivitave at the position, next guess is in the direction of ascent(or descent)
	* Newton's methid may NOT work if there are multiple optima
* If big input space, no derivative(or hard to calculate derivative), complex function, multiple local optima...then other options:
* Hill Climbing, pick a point find neighbors, move in the direction of the best neighbor until there aren't better than you neighbors
* This doesn't solve the mutliple local optima set-up, for that:
* Randomized Hill Climbing Restarts - you try multiple times with random different starting points with the hope that from least one you'd be able to Hill Climb to the global optimum
* It's cheap - The is just lenearly more expensive than the single Hill Climb
* Basin of attraction - sometimes the optimum can be reached from a really small percentage of the input space, so we'd have to make sure we cover as much of the input space as feasibly possible
* RHC may not do better than evaluating the whlose space in the worst case, but may not do any worse(assuming we keep track the evaluated values, so we don't re-evaluate them)
* RHC also depends a lot on the size of the attraction basin - if it is big, RHC works well, if it is very small, it is less of a win
* Another way to combat mutltiple local optima:
* Simulated Annealing(based on methalurgy - when hot, lots of energy molecules just jump around like crazy, as it cools off things jump arund less, until they eventually settle)
* Annealing Algo:
	* pick random x in X
	* for a finite set of iterations:
		* SAMPLE a neighbor x_t
		* if fitness(x_t) >= fitness(x) - jump to x_t
		* else P(x, x_t, T) = e^(fitness(x_t) - fitness(x) / T) - probability of jumping to the sampled neighbor
		* decrease temperature T 

	* At high temperatures, you always jump, since P has a large value, as temperaure decreases, it becomes less likely to jump to less fit neighbors
	* Decrease T slowly so we give it a chance to go towards where the high value points are
	* The algo has a remarkable property - the higher the fitness of a point is, the more likely it is for simulated aneealing to converge to it:
	* Pr(ending at x) = ( e^(fitness(x)/T) ) / Z_t(normalization) - Boltzmann Distribution

* Genetic Algorithms
* Population of individuals(hypothesys)
* Local Search - mutation (tweak the individual(hypothesis))
* CrossOver - combine two individuals(hypos)
* The last two will get you to the next generation
* Population like paralel random restarts
* GA Skeleton:
	* P_o - initial population of  of size_k
	* Repeat until convergence:
		* compute fitness of all individuals in P_t(Population at time t, current Generation)
		* select most fit individuals (top half(truncate), weighted probability(roulette wheel))
		* pair up the most fit individuals, replacing least fit individuals via crossover/mutation

* One point cross-over:
	* choose one of 9 positions, then flip flop:
	* child 1 gets the bits to the left of the random position of parent 1 at the same positions and the bits to the right from parent 2 at the same positions
	* child 2 gets the bits to the left of the random position of parent 2 at the same positions and the bits to the right from parent 1 at the same positions
* Inductive bias of one point x-over - locality of bits matter; subspace to optimize -  assumes there are pieces that are right and if we can reuse these pieces, we can get even better
* If locality doesn't matter(it hudlling together doesn't matter), another way to cross over:
* Flip the bits at random positions - every bit comes from one of the parents
* If there are sub-pieces that are correct that may be preserved in the offspring
* this is called Uniform cross-over - this is the way we get our genes from our parents

* Backprop moves slowly from one hypo to a new very similar hypo. This makes backprop prone to falling in a local optimum. In contrast, GA can move abruptly since an offspring can be very different from its parent. Also since GA maintains populations, there are multiple hypos at the same time, covering multiple points of the search space, Hence GA is much more immune to falling into a local optima.
* Schema Theorem: More fit schemas will tend to grow in influence, especially schemas containing a small number of defined bits(containing a large number of '*') and especially when these defined bits are near one another
* Lamarkian Evolution - the experiences of an individual directly affects the genes passed onto that individual's offspring <=> there is a direct relation between an individual evolution and species evolution. Most disagree, but the concept has been applied successfully in GA
* Baldwin effect - similar to Lamark's idea - though it is more like survival of the fittest: If a new predator appears, only the species who LEARN how to avoid it survive and pass their genes on.
* An application of the Baldwin effect would be a NN with some weights allowed to adjust themselves

* Mimic
* Generate samples from P_Theta_t(x)
* Set Theta_t+1 = n-th percentile
* Retain the only the samples whose fitness is better that Theta_t+1
* Estimate P_Theta_t+1(x)
* Repeat
* This should work if:
	* We can estimate P_Theta_t+1(x) - given a finite set of data, can we estimate the probability distribution
	* P_Theta_t is close to P_Theta_t+epsilon

* Dependency trees:
* allow you to capture some relationships between the features(Maximum spanning tree in Information between parent and children):
* p(x) = Product_over_i(p(x_i | parent(x_i))
* while not having to pay exponantial cost for the estimation(the true dependency would require for every feature to calculate exponentially costly prob - p(x) = P(x1| x2...x_n)*P(x2| x3...x_n)*...*P(x_n)

* Mimic does well with structure - while RHC, GA can sometimes get confused by two different values that are optima, Mimic does well
* Though it usually doesn't happen, Mimic can get stuck in local optima
* Mimic gets randomized restarts for free, via its probabilistic nature
* For all these nice properties, we pay a price in time complexity:
* Mimic usually takes far fewer iterations, but the per iteration cost if far greater - prims algo, draw from distro, remove unfit Xs, estimate probabilities, construct dependency tree, etc
* Mimic is worth using when the Fitness function is very expensive - since there are far fewer iterations, you'd have to compute the fitness function for every X, far fewer times
* Examples of expensive fitness functions: function the performs a detail simulation of how a rocket ship performs

## Information Theory
* If a sequence is predictable, i.e. has LESS uncertainty, then it has LESS information - Shannon described this as Entropy:
* If you had to predict the next symbol in a sequence - what is the miminum number of yes/no questions you would expect to ask.
* 10 coin flips of a fair coin - you'd have to ask 10 questions to figure out the sequence - Information(Entropy) is 1
* 10 coin flips of a coin that always turns up heads - you won't need to ask any questions - Information(Entropy) is 0
* Variable length encoding - symbols that occur more often we encode with less bits, symbols that occur less often, we encode with more bits - this way we save on the amount of bits needed to be transmitted. this also explains why morse code symbols are encoded with variable length
* Example(tree below) A - 50% (0), B -12.5%(110), C - 12.5%(111), D - 25%(10)

       /\
   A  0  1
      	 /\
    D   0  1
           /\
        B 0  1 C

 * Expected questions per bit .5 * 1 + .125 * 3 + .125 * 3 + .25*2 = 1.75
 * Since we had to ask less questions with this language, this language has less information compared to a language where A,B,C,D have uniform probability
 * Number of bits per symbol:
 	* SIGMA_symbols(P(symbol)*number_of_bits_to_encode_the_symbol)
 * This is also called Entropy
 * The number of bits to encode the particular symbol can also be represented as log(1/P(symbol))    
 * e.g symbol A - P = 1/2 - log(1/(1/2)) = 1; symbol D - P = 1/4 - log(1/(1/4)) = 2; symbol B,C - P = 1/8 - log(1/(1/8)) = 3
 * Entropy = Sigma( P(symbol) * log(1/P(symbol))) <==> -Sigma( P(symbol) * log(P(symbol)))
 * Joint Entropy - randomness contained in two variables together:
 	* H(x, y) = -Sigma( P(x, y) * log( P(x, y) ))
 * Conditional Entropy - randomness of one variable given another variable:
 	* H(y | x) = -Sigma( P(x, y) * log( P(y | x)))
 * if x and y are independent, P(y | x ) = P(y), so is H(y | x) = H(y), H(x, y) = H(x) + H(y)
 * although conditional entropy can tell us when two variables are conditionally independent, it is not an adequate measure of dependence:
 * H(y | x) - this may be small if x tells us a great deal about y, or it may be small because H(y) is very small
 * so we use Mutual Information:
 * I(x, y) = H(y) - H(x|y)
 * Mutual information is a measure of the reduction of randomness of a variable, given knowledge of some other variable
 * Kullback-Leibler(KL) Divergence
 * Mutual Information is a case of KL Divergence
 * Measures the similarity(distance D) between two distributions:
 	* p, q - distros to measure the distance between:
 	* D(p || q) = Integral( P(x) * log( P(x)/P(q)) ) - always non-negative, Zero when p = q
 * When is KL Divergence used - usually in supervised learning, we try to model the data after some well known distribution

 ## Parametric vs Non-parametric
 * Parametric - fixed number of params, e.g fit a line through a points y = theta_0 + theta_1 * b
 * Non-parametric - potential for infinite number of parameters - KNN, DT
 	* As data grows, so do the number of parameters describing the data, e.g. the same straight line we fit through Linear Reg, could be a polynomial with many degrees if we use KNN
 * SVM's could be considered eager lazy learners - if we use a Non-Linear kernel, we end up emulating KNN(for wharever the Kernel defines as a neighbor) at training time
 * It may be more fitting to say SVM's are eager non-parameteric learners, since they could end up having infinite number of parameters just like KNN(they could literally use KNN as a Kernel), but since they do it at training, rather than query time, they are eager




 # Markov Decision Processes
 * States
 * Actions - A, A(s)
 * Model - Transition Model - T(s, a, s_prime) produces Pr(s_prime| s, a) - it is stationary, once you have it, it doesn't change
 * Reward - R(s), R(s, a), R(s, a, s_prime)
 * Policy - pie(s) -> a : given I'm in state S, what is the most optimal action to take

 * Markovian property - only the present matters
 * Temporal Credit Assignment Problem - over multiple time steps, (s, a, r), I was in a state s, took an action a, got a reward r - figure out what the actual R is after many iterations
 * if you have a finite Horizon, you lose stationarity - depending on the time step you are in, you may take a different action fro the same state, because you may be running out of time
 * pie(s, t) -> a
 * Utility of sequences
 * Stationary preference - if one sequence of states has a greater utility than another sequence of states, if we just prepend the same state to both sequences, the first one will still have greater utility than the second