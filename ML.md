# Machine Learning Notes

## Neural Nets

* Standard Gradient Descent vs Stochastic Gradient Descent
	* In Standard the error is summed up over all training records, then the weights are updated
	* In Stochastic the error is calculated for the individual training instance and the weights are update immediately - for each training instance
	* Because it uses true gradient, standard gradient descent is often used with larger learning rate(step size) per weight udate than Stochastic Gradient Descent.
	* Stochastic GD could avoid falling into local minima, because its direction of ascent varies more than the Standard GD

* Perceptron vs Delta Rule
	* If the data is linearly separable, the perceptron rule converges after a finite number of iterations
	* The delta rule does NOT requre data be lineraly separable, but it only converges asymptotically towards the min error, possibly requiring unbound amount of time

* Perceptron's discontinuous threshold function makes it undifferentiable, hence not suitable for gradient descent

* Derivative of sigmoid = sigmoid(w*x) * (1 - sigmoid(w*x)) - it approaches zero for larger positive numbers; it approaches zero for smaller negative numbers

* Momentum term in backprop - helps to get through local minima(though it may also get you through a global minimum :); helps to ge through flat regions; helps to get trough steep regions faster

* Backprop over multilayered networks is only guaranteed to converge toward some local minimum, because the Error surface may contain many local minima

* When in local minima with respect to one of the weights, you may not be in local minima with respect to the other weights. The more weights, the more escape routes away from this single dimension's local minima

* Inductive bias - smooth interpolcation between data points - given two +ve examples with no -ve examples between them, BACKPROP tends to label the points in between as positive as well

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

* Many techniques construct function approximation only on the neighborhood of the query.
* Not generalizing over the entire instance space could be very beneficial when the target function is very complex
* Cost of classifying can be high - all computation take place at classification time.
* KNN and similar, usually consider all attributes. If the target concept depends only on some, but not all attributes, similar instances might be far away in eucledian space
* Locally Weighted Regression can be viewed as generalization of KNN
* KNN never forms explicit general hypothesis for the target function
* KNN classification - you count your neighbors and classify as the highest count for a class. Regression - you take the mean of the neighbors.
* KNN - you can also weight by distance. The closer the nighbor is, the higher the weight in the vote or in calculating the mean. One possible weight calculation - 1 / distance^2. If the distance is 0, assign the label of the 0 distance training instance as a class(or projection) for the query record. If more than one zero distance training records - majority(or mean)
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

## Boosting

* Finding many rough rules of thumb, is easier than finding a single highly accurate prediction rule
* When there IS a distribution for which none of the hypotheses does better than chance, there is NO WEAK learner for this hypothesys space(All the learners are weaker then weak - error >= 1/2)
* You can combine simple things to get comlicate things, e.g. you combine linear separators as a weighted average you get a non-linear separator - that's why boosting works
* General feature of ensemble methods: if you try to look at some particular hypothesys class, because you are doing weighted average over hypotheses drawn from that hypothesys class, the final combined hypothesys is at least as complicated as the original hypothesys class.
* Part of the reason that we are getting a non-lenear final hypo is because at the end we pass the weighted average through a non-linear funtion - the SIGN

## SVM

* Find the line of least commitment - the line that gives as much space as possible from the boundaries
* you taking a new point, projecting onto a line and lokking at the value that comes out - y = w_transpose*x + b
* equation of the decision boundary = w_transpose * x + b = 0, since this IS the decision boundary, projection a point from it, onto it(itself) will be 0 - neither positive, nor negative
* 2/||w|| - the distance between two points on the margin lines connected with a line perpendicular to both
* SVM solution - maximize 2/||w|| while classifying everything correctly - y_i(w*x_i + b) >=1; y_i is either 1 or -1; this combines w*x + b >= 1 for the positive and wx + b <=-1 for the negative
* an easier problem to solve is (||w||^2)/2, because it is quadtratic optimization problem and these are well studied
* The above one somehow turns into w = SIGMA of alpha_i * y_i * x_i; when we recover the w's, we can get the b(follows from the line above the line above)
* most of the alphas end up being Zeros, so only few alphas matter - the ones that are closest to the margin lines - these are called the Support Vectors
* Kernel - way to measure similarity, in the Linear SVM, it is (x_i dot x_j) - the dot product would be large positive if they are similar, large negative if dissimilar
* Kernel - way to inject domain knowledge into the SVM
* Kernels need to satisfy the Mercer condition
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

## Comp Learning Theory

* Consistent Learner - learner consistent with the data - learned to produce the the training labels
* Version Space - all the hypotheses consistent with the training data(all the correct hypotheses with respect to the training data)
* True error vs Training error
	* Training error is the percentage misclassified from the training set
	* True error is the probability that the classifier will incorrectly classify data drawn from the true distribution
	* It's OK to misclassify examples you will almost never see - since they are so rare, they will not impact the true error much
	* On the other hand you want to correctly classify common examples, so you have a low true error
* Version Space is epsilon-exhausted iff all the hypotheses in the Version space have error <= epsilon
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
* VC dimension is often the number of parameters - threshold in 1D - 1 (theta); interval in 1D - 2 ([a, b]) separator line in 2D - 3 (w1x1 + w2x2 +b);
* 3 dimensions - 4; d-dimensional hyperplane - d+1 - d the weights for each dimensions + theta(the threshold)
* __m <= (1/epsilon)(8*VC(H)*log2(13/epsilon) + 4*log2(2/epsilon))__ 
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
