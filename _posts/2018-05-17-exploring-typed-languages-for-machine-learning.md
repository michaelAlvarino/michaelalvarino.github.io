---
excerpt: TODO
---

### Why Not Python?
One of the most frustrating problems I run into when creating and prototyping machine learning algorithms is when my algorithm enters an edge-case during a loop which was developed earlier, then breaks because some data structure has changed. For example, replacing a python `tuple` with a `dict` could require new accessor methods. This is a simple change, but very annoying to catch at runtime if your algorithm also does some heavy data preprocessing as many protoyped algorithms will. Additionally I have had times where I fixed an algorithm then let it train for several hours only to realize that my function to save the resulting parameters throws an error due to my fix. Both of these issues can be fixed by using statically typed languages.

### Which Typed Language?
In this post I will document my experience using two typed languages, c++ and Go.

### Why c++?
c++ speaks for itself. It is an extremely performant and low-level language which provides enough abstraction for developers to be productive. Another reason I chose c++ was curiosity. I have never truly dealt with memory management or built a substantial c++ application. As we will see, I still can not say I have done either of those things.
The machine learning community around c++ is very well established, there are many linear algebra libraries and statistical packages with every statistical distribution your heart could desire.

### why Go?
Go is, by now, fairly well known and has had some statistical and linear algebra libraries built to support my endeavors. More importantly however, Go fixed several of my biggest issues and pain points in developing c++, while maintaining a significant performance improvement over python.

### The c++ Task
As a Machine Learning engineer c++ and many of the problems it introduces were very new to me. First and foremost I was worried about having to learn about working with pointers and memory management. Also, I was worried about using outside libraries, building them, and packaging my final product. As we will see, some of these concerns ended up more relevant than others.
The model I chose to implement was the extremely simple and well-known beta-bernoulli model. In this model we have a set of data points

$$X, n \in (1, ... , N)$$

where each

$$x \in {0, 1}$$

$$x \sim Bernoulli(\Theta) = \Theta^x(1-\Theta)^{1-x}$$

$$\Theta \sim Beta(\alpha, \beta) = \frac{\Theta^{\alpha-1}(1-\Theta)^{\beta-1}}{B(\alpha, \beta)}$$

Where $$B(\alpha, \beta)$$ is a set of gamma functions we will be able to ignore for the MAP solution. In this case the prior is conjugate to the likelihood and their product looks like the following.

$$P(\Theta | D) = P(D| \Theta) P(\Theta) = \prod\Theta^x(1-\Theta)^{1 - x} P(\Theta)$$

$$= \Theta^{\alpha + \sum x - 1}(1-\Theta)^{\beta + N - 1 - \sum x}$$

Looking at this closely one could recognize that all we need to get from our data are two things.

1. The number of data points $$N$$
1. The number of successes $$\sum x$$

So effectively, I needed a c++ program that would read in a file, count the number of lines, and count the number of successes. With the result I could calculate the parameters to my $$Beta$$ distribution, whose sample would be $$\Theta$$, the parameter to my Bernoulli random variable.

My first issue was in reading in a file. Using c++ you can open a file, to get a pointer, read from that pointer line by line, then cast those values to integers and move on. Alternately, you could simply feed a file in using `cat` and the `stdin`. Which of these is best practice or better? I have no idea. 

The next problem was in the $$Beta$$ distribution. I was not surprised that I would need an external library, but the way that library was installed and used seemed unmaintainable. To use and install this library I would use the Ubuntu package manager `sudo apt-get install ...`, which means that if any other project I was working on needed a different version, then building this version would break! A solution to this could be Docker, but then I had trouble understanding how to mount volumes in Docker to access my training data. This was much too complicated a solution, and I found the build process for my c++ project extremely fragile.
I looked into several library managers for c++ but all seemed to either have limited library support, very steep learning curves, or be system-based rather than project based.

What I wanted was a language as simple as c, in which I could be as productive as c++, but with a very simple library management tool.

### The Go Task
The Go task I chose was very similar, in fact it was an abstraction upon the previous model to be able to predict more categories than a 0/1 random variable. Since our random variable is now $$x \in 0, 1, 2, ..., N$$ we know it is sampled from a categorical distribution. Further, we need  a prior over our category probabilities which is a valid distribution. It so happens that the Dirichlet distribution is the conjugate prior to the categorical distribution. This model is especially well described in Kevin Murphy's [Machine Learning](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/ref=sr_1_2?ie=UTF8&qid=1526585875&sr=8-2&keywords=machine+learning+a+probabilistic+perspective) textbook in chapter 3.4.

$$likelihood = P(D | \Theta) = \prod \theta^{N_k}$$

$$prior = P(\Theta | \alpha) = \frac{1}{B(\alpha)} \prod \theta^{\alpha_k -1}$$

Where $$N_k$$ is the number of times an event of category $$k$$ has been seen, aka a count.

$$posterior = P(\Theta | D) = P(D | \Theta) P(\Theta) = \prod \theta^{N_k} \theta^{\alpha_k - 1}$$
$$ = \prod \theta^{N_k + \alpha - 1}$$

This is another simple model, in this case we need only $$k$$ counts to determine the parameters of our Dirichlet distribution whose sample will be the parameters of our categorical distribution. Really we can use $$k-1$$ counts.

This finally brings us to Go. Initially I had read some [chat threads](https://www.reddit.com/r/golang/comments/79ggpf/machine_learning_with_go/) about using golang for machine learning and it seems like it's mostly used for deployment in production. This solves my first issue because it indicates Go is a good language to create reliable machine learning applications in and was the initial reason for my venture into typed languages for machine learning.

Reading the [Golang tour](https://tour.golang.org/welcome/1) and reading the [go command docs](https://golang.org/cmd/go/) I quickly found out that go is extremely opinionated in project structure and source code organization. In go projects all source code for all projects are kept in the users `go` workspace. To use an external library programmers must use the command line utility `go get github.com/username/project/...` to retrieve source code from a repository (repos other than github are also supported obviously). This command downloads the original source code from the project to your workspace in order to be imported to your source code, then compiled with `go build`. Further, `go install` will build your project and place the executable in your `$GOPATH/bin` directory. 

This was all in stark contrast and extremely opinionated when compared to c++. In c++ libraries are system dependent and installed into `/usr/includes`, or `/usr/local/includes` directories, or sometimes just header files copied into project source code. The resulting binary can also be placed in any of `/usr/bin`, `/usr/local/bin`, or any other location on your `PATH` environment variable.

Of course the executable location for go is also `PATH` variable dependent, however go is new and controlled enough that there is a single standard for the language.

One downside to go is my personal productivity level in it. Currently I am not experienced enough in go to really think I will be productive in a language with no support for classes or overloaded methods or some of the other conveniences of c/c++. However, it seems that this was all part of the design of go and helped produce a very reliable language and tools that work as-promised!

### In Conclusion

c/c++ are extremely powerful and performant languages in the hands of the right developer. Unfortunately I found that the variety of best practices, system dependence, and breadth of knowledge necessary for simple tasks too big of a barrier to entry for the language. Go provides a similar language which is much more opinionated, has significant performance improvements over python, and a good community around Machine Learning applications.

Going forward with side projects and small machine learning tasks, I intend to use Python to prototype algorithms and go to improve performance where possible. One interesting task would be improving the performance of my [Efficient Thompson Sampling for Probabilistic Matrix Factorization](https://github.com/michaelAlvarino/ParticleThompsonSamplingMAB). Before attempting that I want to continue on a simple progression through some machine learning algorithms and common bayesian models like some different naive bayes classifiers.
