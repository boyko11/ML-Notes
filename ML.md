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