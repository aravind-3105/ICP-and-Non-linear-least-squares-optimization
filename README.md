# ICP + Non-linear least squares optimization

## Non Linear Least Squares Optimization

### 1.1 Gradient Descent
Implement the gradient descent algorithm using numpy and what you have learned from class to solve for the parameters of a gaussian distribution.

To understand the task in more detail and look at a worked through example, checkout the subsequent section. You have to implement the same using just numpy functions. You can refer to [Shubodh's notes](https://www.notion.so/saishubodh/From-linear-algebra-to-non-linear-weighted-least-squares-optimization-13cf17d318be4d45bb8577c4d3ea4a02) on the same to get a better grasp of the concept before implementing it.
* Experiment with the number of iterations.
* Experiment with the learning rate.
* Experiment with the tolerance.

Display your results using matplotlib by plotting graphs for 
* The cost function value vs the number of iterations
* The Ground Truth data values and the predicted data values.

Your plots are expected to contain information similar to the plot below:

<!-- <figure> -->
<img src='./helpers/sample_plt.png' alt=drawing width=500 height=600>

<!-- <figcaption align='center'><b>A sample plot, you can use your own plotting template</b></figcaption>
</figure> -->
<!-- head over to [this page](https://saishubodh.notion.site/Non-Linear-Least-Squares-Solved-example-Computing-Jacobian-for-a-Gaussian-Gradient-Descent-7fd11ebfee034f8ca89cc78c8f1d24d9) -->

## Worked out Example using Gradient Descent

A Gaussian distribution parametrized by $a,m,s$ is given by:

$$ y(x;a,m,s)=a \exp \left(\frac{-(x-m)^{2}}{2 s^{2}}\right) \tag{1}$$

### Jacobian of Gaussian

$$\mathbf{J}_y=\left[\frac{\partial y}{\partial a} \quad \frac{\partial y}{\partial m} \quad \frac{\partial y}{\partial s}\right] \\
= \left[ \exp \left(\frac{-(x-m)^{2}}{2 s^{2}}\right); \frac{a (x-m)}{s^2} \exp\left(\frac{-(x-m)^{2}}{2 s^{2}}\right);  \frac{a (x-m)^2}{s^3}\exp \left(\frac{-(x-m)^{2}}{2 s^{2}}\right)\right]$$

## Problem at hand

> Given a set of observations $y_{obs}$ and $x_{obs}$ we want to find the optimum parameters $a,m,s$ which best fit our observations given an initial estimate.

Our observations would generally be erroneous and given to us, but for the sake of knowing how good our model is performing, let us generate the observations ourselves by assuming the actual "actual" parameter values as $a_{gt}=10; m_{gt} =0; s_{gt} =20$ ($gt$ stands for ground truth). We will try to estimate these values based on our observations and let us see how close we get to "actual" parameters. Note that in reality we obviously don't have these parameters as that is exactly what we want to estimate in the first place. So let us consider the following setup, we have:

- Number of observations, $num\_obs = 50$
- Our 50 set of observations would be
    - $x_{obs} = np.linspace(-25,25, num\_obs)$
    - $y_{obs} = y(x_{obs};a_{gt},m_{gt},s_{gt})$  from $(1)$

Reference:

→[linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

- Say we are given initial estimate as:

    $$a_0=10; \quad m_0=13; \quad s_0=19.12$$

### Residual and error to be minimized

Okay, now we have set of observations and an initial estimate of parameters. We would now want to minimize an error that would give us optimum parameters.

The $residual$ would be given by

$$ r(a,m,s) = \left[ a \exp \left(\frac{-(x_{obs}-m)^{2}}{2 s^{2}}\right) - y_{obs}\ \right]$$

where we'd want to minimize $\|r\|^2$. Note that $r$ is a non-linear function in $(a,m,s)$.

Also, note that since $y$ (and $x$) are observations in the above equation, after simplification, we get $\mathbf{J}_r = \mathbf{J}_y$ [above](https://www.notion.so/c9e6f71b67a44bb8b366df2fccfc12d0) (since $y_{obs}$ is a constant).

Let us apply Gradient Descent method for minimization here. From [Table I](https://www.notion.so/From-linear-algebra-to-non-linear-weighted-least-squares-optimization-13cf17d318be4d45bb8577c4d3ea4a02),  

$$\Delta \mathbf{k} = - \alpha \mathbf{J_F} = -\alpha \mathbf{J}_r^{\top} {r}(\mathbf{k})$$

Note that $\mathbf{J_F}$ is the Jacobian of "non-linear least squares" function $\mathbf{F}$ while $\mathbf{J}_r$ is the Jacobian of the residual. 

where $\mathbf{k}$ is $[a,m,s]^T$. 

- Some hyperparameters:
    - Learning rate, $lr = 0.01$
    - Maximum number of iterations, $num\_iter=200$
    - Tolerance, $tol = 1e-15$

## Solution for 1 iteration

To see how each step looks like, let us solve for 1 iteration and for simpler calculations, assume we have 3 observations, 

$$x_{obs}= \left[ -25, 0, 25 \right]^T, y_{obs} = \left[  4.5783, 10, 4.5783 \right]^T. $$

With our initial estimate as $\mathbf{k_0} = [a_0=10, \quad m_0=13, \quad s_0=19.12]^T$, the residual would be 

$$ r(a_0,m_0,s_0) = \left[ a_0 \exp \left(\frac{-(x_{obs}-m_0)^{2}}{2 s_0^{2}}\right) - y_{obs}\ \right]$$

Therefore, $r=[-3.19068466, -2.0637411 , 3.63398058]^T$.

#### Gradient Computation

Gradient, $\mathbf{J_F}$=

$$\mathbf{J_r}^{\top} \mathbf{r}(\mathbf{k})$$

We have calculated residual already [above](https://www.notion.so/c9e6f71b67a44bb8b366df2fccfc12d0), let us calculate the Jacobian $\mathbf{J_r}$.

$$\mathbf{J}_r
= \left[ \exp \left(\frac{-(x-m)^{2}}{2 s^{2}}\right); \frac{a (x-m)}{s^2} \exp\left(\frac{-(x-m)^{2}}{2 s^{2}}\right);  \frac{a (x-m)^2}{s^3}\exp \left(\frac{-(x-m)^{2}}{2 s^{2}}\right)\right]$$

$$\implies \mathbf{J_r} = \left[ \begin{array}{rrr}0.1387649 & 0.79362589, & 0.82123142 \\-0.14424057 & -0.28221715  & 0.26956967 \\0.28667059 & 0.19188405, & 0.16918599\end{array}\right]$$

So ,

$$\mathbf{J_F} = \mathbf{J_r}^{\top} \mathbf{r}(\mathbf{k})$$

$$\mathbf{r}(\mathbf{k}) =  \left[ \begin{array}{r}-3.19068466 \\ -2.0637411 \\ 3.63398058 \end{array} \right]$$

$$ \begin{aligned} \implies \mathbf{J_F} = \left[ \begin{array}{r} 0.89667553 \\ -1.25248392 \\-2.56179392\end{array} \right] \end{aligned}$$

#### Update step

$$
\Delta \mathbf{k} = - \alpha \mathbf{J_F} \\
\mathbf{k}^{t+1} = \mathbf{k}^t + \Delta \mathbf{k}
$$

Here, $\alpha$ our learning rate is 0.01.

$$
\Delta \mathbf{k} = - \alpha\times\left[ \begin{array}{r} 
0.89667553 \\ -1.25248392 \\-2.56179392
\end{array} \right] = \left[ \begin{array}{r}
-0.00896676 \\ 0.01252484 \\0.02561794
\end{array}\right]
$$

$$
\mathbf{k}^{1} = \mathbf{k}^{0} + \Delta \mathbf{k} \\ \left[\begin{array}{r} 10 \\ 13 \\ 19.12 \end{array}\right] + \left[\begin{array}{c} 9.99103324 \\ 13.01252484 \\ 19.14561794 \end{array} \right]
$$

With just one iteration with very few observations, we can see that we have gotten *slightly* more closer to our GT parameter  $a_{gt}=10; m_{gt} =0; s_{gt} =20$. Our initial estimate was $[a_0=10, \quad m_0=13, \quad s_0=19.12]$. However, the above might not be noticeable enough: Hence you need to code it for more iterations and convince yourself as follows:

![](images/Img1.png)
![](images/DistCurvesQ1.png)

### 1.2: Another Non-Linear function
Now that you've got the hang of computing the jacobian matrix for a non-linear function via the aid of an example, try to compute the jacobian of a secondary gaussian function by carrying out steps similar to what has been shown above. The function is plotted below:
<img src='./helpers/non_linear.png' alt=drawing width=500 height=600><br>
Using the computed jacobian, optimise for the four parameters using gradient descent, where the parameters to be estimated are: 

$p_1$ = 2,  $p_2$ = 8,  $p_3$ = 4,  $p_4$ = 8. 

Do this for $x_{obs} = np.linspace(-20,30, num\_obs)$, where $num\_obs$ is 50.

![](images/NonLinearFunctionDist.png)
![](images/NonLinearFunctionErrorPlot.png)


#### Questions
  1. How does the choice of initial estimate and learning rate affect convergence? Observations and analysis from repeated runs with modified hyperparameters will suffice.
  2. Do you notice any difference between the three optimizers? Why do you think that is? (If you are unable to see a clear trend, what would you expect in general based on what you know about them)

#### Answers

1.
  ![Observations](assets/AllGD.jpg)
  ![a](assets/ChangeInit.jpg)

  
  <b>Effect of choice of initial estimate:</b>The solution can be obtained if the choice of initial estimates are roughly close to the actual solution without a major difference i.e. (10,13,19.12) fits well but (1,25,19.12) does not when the learning rates and number of observations are kept constant. This has been shown in the above graphs The iterations needed would need to be increased when initial estimate is far away espcially when the learing rate remains constant. <br>
  <b>Effect of choice of learning rate:</b>Though it is expected that a high learning rate can make the convergence faster, but this can lead to overshooting thereby, not reaching the final ground truth value. If it is too low, the number of iterations taken for convergence would increase as it takes more iterations to actually reach the minima. Different value combinations of iterations and learning rate have been demonstrated above.
  
2. 
    In utilising different different optimisiers, unlike the gradient descent whixh merely moves towards the 'minima' at a fixed learning rate, the other two optimisers are relativel more sophisticated in their approach. The Gauss-newton is able to obtain a closed form soluiton of the locally-linear approximation for next iteration. The LM algorithms takes the best of both worlds and actual changes the learning factor with a damping factor so as to increase or decrease learning rate to actually achieve the minima. It follows the gradients when the error is more and follows the curvature when error is less.<br>
    It it more robust and can handle mor distance initial estimates better but is relatively slower are there is rejection in update too if it error is increasing in a particular step.<br>
    In terms of initial estimates, only LM can handle a much larger range of in initial estimates as it will follow the gradients initially to reach closer while for GN and GD, the initial estimates much be close to the actual ground truth.<br>
    LM and GN would converge faster than GD as GD only follows a constant learning rate wgile the other two exploit other factors of curvature too in critical cases.

___

# Different Optimizers


Replace gradient descent with Gauss-Newton and Levenberg Marquardt algorithms and repeat question 1.1. 

To quickly recap, Gauss-Newton and Levenberg Marquardt are alternate update rules to the standard gradient descent. Gauss Newton updates work as:

$$\delta x = -(J^TJ)^{-1}J^Tf(x)$$

Levenberg Marquardt lies somewhere between Gauss Newton and Gradient Descent algorithms by blending the two formulations. As a result, when at a steep cliff, LM takes small steps to avoid overshooting, and when at a gentle slope, LM takes bigger steps:


$$\delta x = -(J^TJ + \lambda I)^{-1}J^Tf(x)$$

___

## Gauss-Newton Algorithm
![](images/GNerrorPlot.png)
![](images/GNdist.png)

___

## Levenberg Marquardt Algorithm
![](images/LMerrorPlot.png)
![](images/LMdist.png)

___

# 2. Iterative Closest Point

In this subsection, we will code the Iterative Closest Point algorithm to find the alignment between two point clouds without known correspondences. The point cloud that you will be using is the same as the one that you used in Assignment 1.

## 2.1: Procrustes alignment

1. Write a function that takes two point clouds as input wherein the corresponding points between the two point clouds are located at the same index and returns the transformation matrix between them.
2. Use the bunny point cloud and perform the procrustes alignment between the two bunnies. Compute the absolute alignment error after aligning the two bunnies.
3. Make sure your code is modular as we will use this function in the next sub-part.
4. Prove mathematically why the Procrustes alignment gives the best aligning transform between point clouds with known correspondences.


1.3833830427744288e-09 is the absolute alignment error

![b](assets/BeforeProc.png)
![c](assets/AfterProc.png)

![](./proof_page-0001.jpg)
![](./proof_page-0002.jpg)
![](./proof_page-0003.jpg)
![](./proof_page-0004.jpg)

# ICP alignment

1. Write a function that takes two point clouds as input without known correspondences and perform the iterative closest point algorithm.
2. Perform the ICP alignment between the two bunnies and plot their individual coordinate frames as done in class.
3. Does ICP always give the correct alignment? Why or Why not?
4. What are other variants of ICP and why are they helpful (you can look at point to plane ICP)?


## Results: 
<b>1. </b>
### Sample-1
![b](assets/BeforeICP.png)
![c](assets/AfterICP.png)


<b>2.</b>

### Sample-2
![b](assets/SmallTransBefore.png)
![c](assets/SmallTransAfter.png)

<b>3.</b>

No ICP does not always give correct alignment. This is because of the nearest neighbour step in ICP which doesn't always give correct correspondences and the existence of noise between the two pointclouds.

When doing orthogonal procrustes, if all the correct correspondences are known and there exists no noise, then we are guaranteed to get a perfect alignment in a single step. 

However in the ICP algorithm, we have no way of finding the perfect correspondences. Therefore we use a nearest neighbour approach where we take each point in the second set of points (lets say we want to align this with the first set of points) and find the nearest neighbour to it in the first set of points. We set the correspondence of each point in this second set of points as the nearest neighbour found in the first set. This can lead to many points in the second set to have the same correspondence in the first set. 

The above reason can cause the ICP algorithm to get stuck at a particular orientation that does not allow for perfect alignment because of incorrect correspondences. 

This is why ICP cannot get perfect alignment every time. This problem is seen more when there is a huge misalignment between the two sets of points since nearest neighbour will never be able to find the correct correspondences.

<b>4.</b> 

Point-to-Plane ICP and Generalized ICP are two other variants of ICP. They consider the fact that the object we are looking at has surfaces and that the points lie somewhere on those surfaces. Point-to-Point (the version we have implemented) ICP minimizes distance between sets of points and does not consider surfaces.

Point-to-Plane changes the cost function over which we operate. ICP minimizes squared distance between points. in Point-to-Plane ICP we project the point to point error vector onto the normal of the surface. this make the proof based on svd impossible and hence a least squares approach is taken.

Generalized icp is a generalised version that combines Point-to-Point, Point-to-Plane and Plane-to-Plane metrics for ICP.

In practice we find that these variants converge in lesser iterations. This can be because they try to align the surfaces rather that just trying to align individual points. 

Therefore these variants are useful since they converge in lesser iterations and take into account the fact that points lie on a surface.