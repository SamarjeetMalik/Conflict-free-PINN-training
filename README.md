
<h4 align="center">Conflict free Physics Informed NN training methods</h4>



​This is a generic method for optimization problems involving **multiple loss terms** (e.g., Multi-task Learning, Continuous Learning, and Physics Informed Neural Networks). It prevents the optimization from getting stuck into a local minimum of a specific loss term due to the conflict between losses. On the contrary, it leads the optimization to the **shared minimum of all losses** by providing a **conflict-free update direction.**


​The ConFIG(Conflict Free Inverse Gradient) method obtains the conflict-free direction by calculating the inverse of the loss-specific gradients matrix:

```math
\boldsymbol{g}_{ConFIG}=\left(\sum_{i=1}^{m} \boldsymbol{g}_{i}^\top\boldsymbol{g}_{u}\right)\boldsymbol{g}_u,
```

```math
\boldsymbol{g}_u = \mathcal{U}\left[
[\mathcal{U}(\boldsymbol{g}_1),\mathcal{U}(\boldsymbol{g}_2),\cdots, \mathcal{U}(\boldsymbol{g}_m)]^{-\top} \mathbf{1}_m\right].
```

Then the dot product between $\boldsymbol{g}_{ConFIG}$ and each loss-specific gradient is always positive and equal, i.e., $`\boldsymbol{g}_{i}^{\top}\boldsymbol{g}_{ConFIG}=\boldsymbol{g}_{j}^{\top}\boldsymbol{g}_{ConFIG}> 0  \quad \forall i,j \in [1,m]`$​.

## Paper Info

<h4 align="center">ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks</h4>
<h6 align="center"><img src="./docs/assets/TUM.svg" width="16"> <a href="https://qiauil.github.io/">Qiang Liu</a>, <img src="./docs/assets/PKU.svg" width="14"> <a href="https://rachelcmy.github.io/">Mengyu Chu</a>, and <img src="./docs/assets/TUM.svg" width="16"> <a href="https://ge.in.tum.de/about/n-thuerey/">Nils Thuerey</a></h6>

<h6 align="center">
    <img src="./docs/assets/TUM.svg" width="16"> Technical University of Munich
    <img src="./docs/assets/PKU.svg" width="14"> Peking University
</h6>

***Abstract:*** The loss functions of many learning problems contain multiple additive terms that can disagree and yield conflicting update directions. For Physics-Informed Neural Networks (PINNs), loss terms on initial/boundary conditions and physics equations are particularly interesting as they are well-established as highly difficult tasks. To improve learning the challenging multi-objective task posed by PINNs, we propose the ConFIG method, which provides conflict-free updates by ensuring a positive dot product between the final update and each loss-specific gradient. It also maintains consistent optimization rates for all loss terms and dynamically adjusts gradient magnitudes based on conflict levels. We additionally leverage momentum to accelerate optimizations by alternating the back-propagation of different loss terms. The proposed method is evaluated across a range of challenging PINN scenarios, consistently showing superior performance and runtime compared to baseline methods. We also test the proposed method in a classic multi-task benchmark, where the ConFIG method likewise exhibits a highly promising performance. 

## Installation

* Install through `pip`: `pip install conflictfree`
* Install from repository online: `pip install git+https://github.com/tum-pbs/ConFIG`
* Install from repository offline: Download the repository and run `pip install .` or `install.sh` in terminal.
* Install from released wheel: Download the wheel and run `pip install conflictfree-x.x.x-py3-none-any.whl` in terminal.

## Usage

For a muti-loss optimization, you can simply use ConFIG method as follows:

Without `ConFIG`:

```python
optimizer=torch.Adam(network.parameters(),lr=1e-3)
for input_i in dataset:
    losses=[]
    optimizer.zero_grad()
    for loss_fn in loss_fns:
        losses.append(loss_fn(network,input_i))
    torch.cat(losses).sum().backward()
    optimizer.step()
```

With `ConFIG`:

```python
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector
optimizer=torch.Adam(network.parameters(),lr=1e-3)
for input_i in dataset:
    grads=[]
    for loss_fn in loss_fns:
    	optimizer.zero_grad()
    	loss_i=loss_fn(input_i)
        loss_i.backward()
        grads.append(get_gradient_vector(network)) #get loss-specfic gradient
    g_config=ConFIG_update(grads) # calculate the conflict-free direction
    apply_gradient_vector(network) # set the condlict-free direction to the network
    optimizer.step()
```

More details and examples can be found in our [doc page](https://tum-pbs.github.io/ConFIG/).

To reproduce the result in our paper, please check the [experiments](https://github.com/tum-pbs/ConFIG/tree/main/experiments) folder.

## Additional Info
This project is part of the physics-based deep learning topic in [**Physics-based Simulation group**](https://ge.in.tum.de/) at TUM.
