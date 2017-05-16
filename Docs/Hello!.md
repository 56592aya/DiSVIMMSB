The network generation is as follows:

$for\,\,a\in\mathcal{N}:$

$\,\,\,\,\,\,\,\theta_{a}\sim Dir(\alpha_{[K]})$

$for\,\,(a,b)\in\mathcal{N}\times\mathcal{N}:$

$\,\,\,\,\,\,\,z_{a\rightarrow b}\sim Mult(\theta_{a})$

$\,\,\,\,\,\,\,z_{a\leftarrow b}\sim Mult(\theta_{b})$

$\,\,\,\,\,\,\,y(a,b)\sim Bern(z_{a\rightarrow b}^{T}B\,z_{a\leftarrow b})$

The NIPS paper by Airoldi et al 2008 states that:

So this means that the order of indexes indicates the order of potential link, and the direction of the arrow indicates the potential behavior upon initiation versus reception(I am still waiting to hear from Airoldi et al to make sure, no luck yet!). Consider the scenario of how an opion leader mayy interact with a follower versus follower with an opinion leader(or expert or novice relationship). Although the possibility of link in one direction should be very much higher that the other way around in these scenarios if the group memberships differ.

We begin by writing down the ELBO:

$$\begin{aligned}
\mathcal{L} & = & \underset{a}{\sum}\underset{b\in sink(a)}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & + & \underset{a}{\sum}\underset{b\in sink(a)}{\sum}\underset{k}{\sum\Big(1-}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big)log\,\epsilon\\
 & + & \underset{a}{\sum}\underset{b\notin sink(a)}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & + & \underset{a}{\sum}\underset{b\notin sink(a)}{\sum}\underset{k}{\sum}\Big(1-\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big)\Big(log\,(1-\epsilon)\Big)\\
 & + & \underset{a}{\sum}\underset{b}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & + & \underset{a}{\sum}\underset{b}{\sum}\underset{k}{\sum}\phi_{a\leftarrow b,k}\Big(\Psi(\gamma_{b,k})-\Psi(\sum_{h}\gamma_{b,h})\Big)\\
 & + & \sum_{a}log\,\Gamma(\sum_{k}\alpha_{k})-\sum_{a}\sum_{k}log\,\Gamma(\alpha_{k})+\sum_{a}\sum_{k}(\alpha_{k}-1)\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & + & \sum_{k}log\,\Gamma(\eta_{0}+\eta_{1})-\sum_{k}log\,\Gamma(\eta_{0})-\sum_{k}log\,\Gamma(\eta_{1})\\
 & + & \sum_{k}(\eta_{0}-1)\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)+\sum_{k}(\eta_{1}-1)\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & - & \underset{a}{\sum}\underset{b}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}log\,\phi_{a\rightarrow b,k}-\underset{a}{\sum}\underset{b}{\sum}\underset{k}{\sum}\phi_{a\leftarrow b,k}log\,\phi_{a\leftarrow b,k}\\
 & - & \sum_{a}log\,\Gamma(\sum_{k}\gamma_{a,k})+\sum_{a}\sum_{k}log\,\Gamma(\gamma_{a,k})-\sum_{a}\sum_{k}(\gamma_{a,k}-1)\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & - & \sum_{k}log\,\Gamma(\tau_{k0}+\tau_{k1})+\sum_{k}log\,\Gamma(\tau_{k0})+\sum log\,\Gamma(\tau_{k1})\\
 & - & \sum_{k}(\tau_{k0}-1)\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)-\sum_{k}(\tau_{k1}-1)\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})\Big)\end{aligned}$$

This can be further simplified dividing expressions between links and non links as follows:

$$\begin{aligned}
\mathcal{L} & = & \underset{a}{\sum}\underset{b\in sink(a)}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & + & \underset{a}{\sum}\underset{b\in sink(a)}{\sum}\underset{k}{\sum\Big(1-}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big)log\,\epsilon\\
 & + & \underset{a}{\sum}\underset{b\notin sink(a)}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & + & \underset{a}{\sum}\underset{b\notin sink(a)}{\sum}\underset{k}{\sum}\Big(1-\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big)\Big(log\,(1-\epsilon)\Big)\\
 & + & \underset{a}{\sum}\underset{b\in sink(a)}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & + & \underset{a}{\sum}\underset{b\notin sink(a)}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & + & \underset{a}{\sum}\underset{b\in source(a)}{\sum}\underset{k}{\sum}\phi_{b\leftarrow a,k}\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & + & \underset{a}{\sum}\underset{b\notin source(a)}{\sum}\underset{k}{\sum}\phi_{b\leftarrow a,k}\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & + & \sum_{a}log\,\Gamma(\sum_{k}\alpha_{k})-\sum_{a}\sum_{k}log\,\Gamma(\alpha_{k})+\sum_{a}\sum_{k}(\alpha_{k}-1)\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & + & \sum_{k}log\,\Gamma(\eta_{0}+\eta_{1})-\sum_{k}log\,\Gamma(\eta_{0})-\sum_{k}log\,\Gamma(\eta_{1})\\
 & + & \sum_{k}(\eta_{0}-1)\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)+\sum_{k}(\eta_{1}-1)\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & - & \underset{a}{\sum}\underset{b\in sink(a)}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}log\,\phi_{a\rightarrow b,k}-\underset{a}{\sum}\underset{b\in source(a)}{\sum}\underset{k}{\sum}\phi_{b\leftarrow a,k}log\,\phi_{b\leftarrow a,k}\\
 & - & \underset{a}{\sum}\underset{b\notin sink(a)}{\sum}\underset{k}{\sum}\phi_{a\rightarrow b,k}log\,\phi_{a\rightarrow b,k}-\underset{a}{\sum}\underset{b\notin source(a)}{\sum}\underset{k}{\sum}\phi_{b\leftarrow a,k}log\,\phi_{b\leftarrow a,k}\\
 & - & \sum_{a}log\,\Gamma(\sum_{k}\gamma_{a,k})+\sum_{a}\sum_{k}log\,\Gamma(\gamma_{a,k})-\sum_{a}\sum_{k}(\gamma_{a,k}-1)\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & - & \sum_{k}log\,\Gamma(\tau_{k0}+\tau_{k1})+\sum_{k}log\,\Gamma(\tau_{k0})+\sum log\,\Gamma(\tau_{k1})\\
 & - & \sum_{k}(\tau_{k0}-1)\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)-\sum_{k}(\tau_{k1}-1)\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})\Big)\end{aligned}$$

Next we want to find the variational parameters that maximize the variational lower bound:

$$\begin{aligned}
\mathcal{L}\Big[\underset{a\rightarrow b}{\phi_{a\rightarrow b,k}}\Big] & = & \phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & - & \phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}log\,\epsilon\\
 & + & \phi_{a\rightarrow b,k}\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & - & \phi_{a\rightarrow b,k}log\,\phi_{a\rightarrow b,k}\\
 & = & \phi_{a\rightarrow b,k}\Bigg(\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)-\phi_{a\leftarrow b,k}log\,\epsilon\\
 &  & +\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)-log\,\phi_{a\rightarrow b,k}\Bigg)\end{aligned}$$

Hence maximizing $\mathcal{L}\Big[\underset{a\rightarrow b}{\phi_{a\rightarrow b,k}}\Big]$ with respect to $\underset{a\rightarrow b}{\phi_{a\rightarrow b,k}}$:

$$\begin{aligned}
\dfrac{\partial\mathcal{L}\Big[\underset{a\rightarrow b}{\phi_{a\rightarrow b,k}}\Big]}{\partial\underset{a\rightarrow b}{\phi_{a\rightarrow b,k}}}\\
=0 & \Longrightarrow & \Bigg(\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)-\phi_{a\leftarrow b,k}log\,\epsilon\\
 &  & +\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)-log\,\phi_{a\rightarrow b,k}\Bigg)-1=0\\
 & \Longrightarrow\\
\underset{a\rightarrow b}{\phi_{a\rightarrow b,k}} & \propto & exp\Bigg(\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})-log\,\epsilon\Big)+\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\Bigg)\\
 & \propto & \boxed{\epsilon^{-\phi_{a\leftarrow b,k}}\times exp\Bigg(\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)+\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\Bigg)}\end{aligned}$$

Similarly for $\underset{a\rightarrow b}{\phi_{a\leftarrow b,k}}$ we have:

$$\begin{aligned}
\mathcal{L}\Big[\underset{a\rightarrow b}{\phi_{a\leftarrow b,k}}\Big] & = & \phi_{a\leftarrow b,k}\Bigg(\phi_{a\rightarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)-\phi_{a\rightarrow b,k}log\,\epsilon\\
 &  & +\Psi(\gamma_{b,k})-\Psi(\sum_{h}\gamma_{b,h})-log\,\phi_{a\leftarrow b,k}\Big)\Bigg)\\
\dfrac{\partial\mathcal{L}\Big[\underset{a\rightarrow b}{\phi_{a\leftarrow b,k}}\Big]}{\partial\underset{a\rightarrow b}{\phi_{a\leftarrow b,k}}}\\
=0 & \Longrightarrow & \Bigg(\phi_{a\rightarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)-\phi_{a\rightarrow b,k}log\,\epsilon\\
 &  & +\Psi(\gamma_{b,k})-\Psi(\sum_{h}\gamma_{b,h})-log\,\phi_{a\leftarrow b,k}\Big)\Bigg)-1=0\\
\underset{a\rightarrow b}{\phi_{a\leftarrow b,k}} & \propto & exp\Bigg(\phi_{a\rightarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})-log\,\epsilon\Big)+\Big(\Psi(\gamma_{b,k})-\Psi(\sum_{h}\gamma_{b,h})\Big)\Bigg)\\
 & \propto & \boxed{\epsilon^{-\phi_{a\rightarrow b,k}}\times exp\Bigg(\phi_{a\rightarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)+\Big(\Psi(\gamma_{b,k})-\Psi(\sum_{h}\gamma_{b,h})\Big)\Bigg)}\end{aligned}$$

For the case of the nonlinks we do not use the averaging over the links. The assumption might produce extra bias for the directed case more than that of the undirected graph. Instead we update the $\phi$’s for nonlinks and reduce the cost of computation by only sampling a portion of them(here $2\times\#links$).

$$\begin{aligned}
\underset{a\nrightarrow b}{\phi_{a\rightarrow b,k}} & \propto & exp\Bigg(\underset{a\nrightarrow b}{\phi_{a\leftarrow b,k}}\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})-log\,(1-\epsilon)\Big)+\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\Bigg)\\
\underset{a\nrightarrow b}{\phi_{a\leftarrow b,k}} & \propto & exp\Bigg(\underset{a\nrightarrow b}{\phi_{a\rightarrow b,k}}\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})-log\,(1-\epsilon)\Big)+\Big(\Psi(\gamma_{b,k})-\Psi(\sum_{h}\gamma_{b,h})\Big)\Bigg)\end{aligned}$$

Turning into the global parameters for the full data:

$$\begin{aligned}
\mathcal{L}\Big[\gamma_{a,k}\Big] & = & \Big(\sum_{b\in sink(a)}\phi_{a\rightarrow b,k}+\sum_{b\notin sink(a)}\phi_{a\rightarrow b,k}\Big)\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & + & \Big(\sum_{b\in source(a)}\phi_{b\leftarrow a,k}+\sum_{b\notin source(a)}\phi_{b\leftarrow a,k}\Big)\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)\\
 & + & (\alpha_{k}-\gamma_{a,k})\Big(\Psi(\gamma_{a,k})-\Psi(\sum_{h}\gamma_{a,h})\Big)+log\,\Gamma(\gamma_{a,k})-log\,\Gamma(\sum_{h}\gamma_{a,h})\\
\dfrac{\partial\mathcal{L}\Big[\gamma_{a,k}\Big]}{\partial\gamma_{a,k}} & = & 0\\
 & \Longrightarrow & \Big(\Psi'(\gamma_{a,k})-\Psi'(\sum_{h}\gamma_{a,h})\Big)\Bigg(\sum_{b\in sink(a)}\phi_{a\rightarrow b,k}+\sum_{b\notin sink(a)}\phi_{a\rightarrow b,k}\\
 &  & +\sum_{b\in source(a)}\phi_{b\leftarrow a,k}+\sum_{b\notin source(a)}\phi_{b\leftarrow a,k}+\alpha_{k}-\gamma_{a,k}\Bigg)=0\\
\gamma_{a,k} & = & \alpha_{k}+\sum_{b\in sink(a)}\phi_{a\rightarrow b,k}+\sum_{b\notin sink(a)}\phi_{a\rightarrow b,k}+\sum_{b\in source(a)}\phi_{b\leftarrow a,k}+\sum_{b\notin source(a)}\phi_{b\leftarrow a,k}\end{aligned}$$

Although with sampling of only portion of the nonlinks this will turn into:

$$\begin{aligned}
 &  & \alpha_{k}+\sum_{b\in sink(a)}\phi_{a\rightarrow b,k}+\tfrac{\#nonsinks_{train}(a)}{\#nonsinks_{minibatch}(a)}\times\sum_{b\notin sink(a)}\phi_{a\rightarrow b,k}+\sum_{b\in source(a)}\phi_{b\leftarrow a,k}+\tfrac{\#nonsources_{train}(a)}{\#nonsources_{minibatch}(a)}\times\sum_{b\notin source(a)}\phi_{b\leftarrow a,k}\\\end{aligned}$$

For $\tau_{k0}$ and $\tau_{k1}$:

$$\begin{aligned}
\mathcal{L}\Big[\tau_{k}\Big] & = & \sum_{a}\sum_{b\in sink(a)}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & + & \sum_{a}\sum_{b\notin sink(a)}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & + & (\eta_{0}-\tau_{k0})\Big(\Psi(\tau_{k0})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & + & (\eta_{1}-\tau_{k1})\Big(\Psi(\tau_{k1})-\Psi(\tau_{k0}+\tau_{k1})\Big)\\
 & - & log\,\Gamma(\tau_{k0}+\tau_{k1})+log\,\Gamma(\tau_{k0})+log\,\Gamma(\tau_{k1})\\
\dfrac{\partial\mathcal{L}\Big[\tau_{k}\Big]}{\partial\tau_{k}} & = & \begin{cases}
\dfrac{\partial\mathcal{L}\Big[\tau_{k0}\Big]}{\partial\tau_{k0}} & =0\\
\dfrac{\partial\mathcal{L}\Big[\tau_{k1}\Big]}{\partial\tau_{k1}} & =0
\end{cases}\\
 & \Longrightarrow & \begin{cases}
\Big(\Psi'(\tau_{k0})-\Psi'(\tau_{k0}+\tau_{k1})\Big)\Bigg(\sum_{a}\sum_{b\in sink(a)}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}+\eta_{0}-\tau_{k0}\Bigg) & =0\\
\Big(\Psi'(\tau_{k1})-\Psi'(\tau_{k0}+\tau_{k1})\Big)\Bigg(\sum_{a}\sum_{b\notin sink(a)}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}+\eta_{1}-\tau_{k1}\Bigg) & =0
\end{cases}\\
 & \Longrightarrow\\
\tau_{k0} & = & \boxed{\eta_{0}+\sum_{a}\sum_{b\in sink(a)}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}}\\
\tau_{k1} & = & \boxed{\eta_{1}+\sum_{a}\sum_{b\notin sink(a)}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}}\end{aligned}$$

Since we do not sample all the links and nonlinks at each iteration we have to reweigh these quantities as follows:

$$\begin{aligned}
 &  & \eta_{0}+\tfrac{\#train\,links}{\#minibatch\,link}\sum_{a}\sum_{b\in sink(a)}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\\
 &  & \eta_{1}+\tfrac{\#train\,nonlinks}{\#minibatch\,nonmarkdownlinks}\sum_{a}\sum_{b\notin sink(a)}\phi_{a\rightarrow b,k}\phi_{a\leftarrow b,k}\end{aligned}$$

More on sampling:

At each iteration, we sample a minibatch of nodes(for now only one), and for each node in the minibatch we acuire all its training links and additionaly sample its training nonlinks uniformly at random for twice the size of its links.
