# Deep Eyes: Progressive Visual Analytics for Designing Deep Neural Networks

作者：Nicola Pezzotti, Thomas Hollt, Jan van Gemert, Boudewijn P.F. Lelieveldt, Elmar Eisemann, Anna Vilanova

## Abstract

DNN 在很多方面都已经胜过了人类，准确度已经超过了人类。比如说在pattern recognition problem. 

<img src="https://user-images.githubusercontent.com/68700549/115153268-04e06580-a043-11eb-9f61-726c7dc21c36.png" alt="Human-versus-machine-based-face-recognition-performance-in-the-FRGC-database-26-shows" style="zoom: 67%;" />

就比如说看这张图，人工智能已经胜过了人类。

我们来对比一下传统的分类器(features are handcrafted)，现在的neural network，会从data中学习features，从而得到参数，这个计算量是相当大的。现在的neural networks，一层一层的神经网络结构，我们需要自己去design。对于一个好的neural network来说，有一些参数很重要，比如说，这个神经网络有多少层啊，有多少个filters，他们是怎么连接啊，等等，这都会影响到我们的network的performance。我们现在设计神经网络是有一个基本的guidelines，设计这个神经网络是一个trial-and-error process. 但是这个process会花很长时间，几天，几周，几个月都有可能。这篇论文中，作者呢，提出了DeepEyes, a progressive visual analytics system.这个是干嘛用的呢？在training的过程中，可以看这个design of neural networks。他们提出了一个新的visualizations，做层的识别，这个层呢，是可以去学习stable set of patterns，所以，这个是可以用来做一些细节分析的。这个system呢，有助于发现一些问题，比如说superfluous filters or layers, and information that is not being captured by the network. 他们呢，描述了这个system的有效性，他们会分析几个案例，他们展示了一个trained network是怎么被compressed reshaped, 适用于其他Problem的。



## 1. Introduction

最近几年，DNN非常的火，在很多方面也有很好的表现，比如说image and speech recognition。 DNN主要就是相互连接的layers。我们来看看DNN的结构。

<img src="https://user-images.githubusercontent.com/68700549/115154296-2132d100-a048-11eb-920f-84d7992fc527.png" alt="Figure_01 5c7712513151e" style="zoom:50%;" />

每一层中的每个neural，都是一个filter。在每一层，都会提取出一些pattern出来。比如说，把一张图放进这里，第一层，往往提取出来的是colors and edges，然后这个特征会在后面不断的积累，最终这个neural network就能够进行识别其他更为复杂的patterns，比如说grids or stripes. 在每一层中，放进成百上千个filters，就可以进行复杂的图像识别等功能。但是，这个training需要一个好的GPU，因为运算量太大。

但是DNN在目前来说都是black box，就是不知道里面发生了什么。现在也有很多research在尝试解释这个black box，可视化这个black box，让大家尽可能地可以理解这里面发生了什么。现在，ML，可视化领域，他们都投放了很多的精力去明白how a trained network behaves. 比如说，把每层所学的pattern给show出来。但是现在，little effort on the creation of tool，就是怎么设计这个神经网络。比如说，我现在遇到什么问题，然后需设计神经网络，但是没有工具能帮助我们设计这个神经网络。即使现在有一些基本的guideline，但是这个过程是一个试错的过程，花费的时间精力会很大。比如说，一些专家，可以改变每一层部分的filter，但是这个改变的效果却不知道是怎么样的，需要结果出来才知道会发生什么，这个过程就需要很长时间。所以呢，一个DNN可视化分析工具呢，就变得非常的必要。最近有一个模式，叫做progressive visual analytics，旨在提升复杂的ML 算法的交互。这个交互是干嘛的呢？就是为用户提供实时结果，就是算法evolves， the training of the network in this setting. 然而，DNN复杂很多，因为因为数据量太大了

所以呢，作者就提出DeepEyes，这个就可以在training过程中可以design这个network。经过一番讨论，发现DeepEyes提供比较少的feedback on how a DNN can be improved by the designers. 为了解决这个limitation，所以就做了几个调整

* Identification of stable layers, 就是稳定层的识别。这个可以分析更多的细节
* Identification of degenerated filters，就是识别一些不要的filters，对整体结果没啥贡献的，可以删掉。
* Identification of pattern undetected, 就是有些个pattern没有被检测到，需要添加一些filters或者是layers
* Identification of oversized layers, 就是有些layers有一些是没有用到的filters（但是我不知道什么时候会没有用到这个filter。这个要跟第二个区分开来，第二个是用了，但是没啥贡献，这个是直接没有用到，可以在layer中被reduced的）
* Identification of unnecessary layers or the need of additional layers,就是不必要的layers，或者是需要更多的layers

这个work的主要东西就是DeepEyes framework。DeeEyes在把训练到a single, progressive visual analytics fraework的过程中首次整合一些机制来解决所有现在的tasks来分析DNNs。我们来看看本文献中堆DeepEyes的贡献

* 是一个新的，数据驱使的一个分析model，基于我们的输入空间的部分区域的样本，这个就可以做progressive analysis of the DNN during training
* Perplexity Histograms，是一个新的方法来做这个identification of stable layers of the DNN
* 就是改造一些已经存在的方法使得可以进行细节分析：activation heatmap, Input map, and filter map

在下一章节，介绍了一些基础知识，让我们这些读者更好地明白这篇paper。Section 4会正式展示DeepEyes，会描述他们的方法，可视化设计等。紧接着，他们会介绍一些例子，比如说DNN for the classification of handwritten digits. Implementation details 会在section 6 里展示。 

## 2. Deep Learning Primer

DANN 可以用于image classification。这个的最终目标就是预测一张图片的分类。Training set是一个高纬度的数据$x\in R^n$ ,还有predict vector $y\in {0,1}^d$ ,where d is the total number of labels. 

![WeChat Screenshot_20210418144215](https://user-images.githubusercontent.com/68700549/115156845-53e2c680-a054-11eb-9877-5c30b5a7eeb4.png)

我们来看看作者给的这个DNN例子，这个五层。第0层是data，第1第2是CNN, 第三第四是FC，我们先假设DNN有$L$ 层layers，每层layer $l\in L$ 都包含着很多的neurons that computes the filter functions $f_i^l \in F^l$ ，或者简单来说，就是filters 。

<img src="https://user-images.githubusercontent.com/68700549/115157363-d8cedf80-a056-11eb-958c-614bf8b47625.png" alt="WeChat Screenshot_20210418150032" style="zoom: 80%;" /> 

我们再看这个b-e的图，一个pixel就是一个dimension，我们再看回再上面的一幅图，我们看layer 1，就是CNN的部分，我们知道这个过程不是取所有的dimensionality，而是取部分，subset $R^{k^l} \in R^n$ , the receptive fields where $k^l$ is the size for layer $l$. 这里，我们理解一下什么是receptive field。就是经过一个kernel后，会出现一个值，那个值就是receptive field。 在卷积神经网络中,**感受野(Receptive Field)**是指特征图上的某个点能看到的输入图像的区域,即特征图上的点是由输入图像中感受野大小区域的计算得到的。在CNN中，我们为什么要用这么多的filters呢？就是因为要用来detect不同的形状。就比如说我们看上图的d,e, $f^1_4$ 擅长检测耳朵一样的形状，$f^1_2$ 擅长检测圆形一样的形状，我们就看receptive field值的大小

给定一个receptive field，那么就会有1对1对应的filters and neurons, 看上图b-e就可以理解，这个receptive field就会不断的堆叠起来，每一个filter可能就擅长检测某一种特征。这些filter functions提取到的信息会比single neuron更多。同理，对于其他的layers，也是这么操作，也是相同的意义。越deeper layers, receptive fields就会越大。这个filter function的设计就是为了detect更加复杂的pattern。

![WeChat Screenshot_20210418211613](https://user-images.githubusercontent.com/68700549/115169213-6298a000-a08b-11eb-8199-cdf05f5cb856.png)

正如上图所示，把很多的filter 累积起来，就可以detect比较复杂的pattern。然后这个堆叠起来的filters就是neuronal receptive field. 就是一个3D的啊。

![WeChat Screenshot_20210418211613](https://user-images.githubusercontent.com/68700549/115169529-40535200-a08c-11eb-8139-e3ea7e83d3fd.png)

对于FC层，也是一样的操作，只不过是变成神经元的那种全连接，1-to-1 correspondence. 在FC，一个neuron就是一个filter。

这个section，就是对DNN有一个简单的介绍。In modern architectures many different layers are used to define the filter functions such as max-pooling and normalization layers. 如果还想要有更深的了解，建议去读LeCun的Deep Learing 论文， 

链接在此：https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf

DNN的过程中，我们可以看到每一层，input data啊，receptive field啊，都是interpretable的，但是abstract weights and relationships between neurons are not. 

![WeChat Screenshot_20210418224512](https://user-images.githubusercontent.com/68700549/115174744-d04ac900-a097-11eb-99f7-de8d973bf908.png)



## 3. Related work

现存的可视化DNN的方法中有三种：weight-centric, dataset-centric, and filter-centric 

* Weight-centric: 

  这个technique旨在视觉化不同层之间filters的关系通过可视化那个learnable parameters, weights. 一个很直接的可视化方式就是node-link diagrams，有点像FC那种可视化，只不过呢，这里就把weight可视化成edges，也就是line thinkness。但是这个方法不适合最近比较新的network，因为FC filter太多了，parameters太多了。于是，Liu et al.就提出，可以把biclustering-based edge buding approach. 也就是说把neurons 整合在一块如果这些neurons都是share相同的label的话。如果edges有相同或者都是非常大的weght的话，也放在一块。然而，DNN是只有在最后一层才正式分类，所以，在之前的layers就做这种bunding就不太合适。

* dataset-centric:

  就是提供一个整体的view来看看input data在network是怎么process的，而不是一步一步地进行分析。

* filter-centric:

  就是找出那个filter detect的pattern是什么。一个简单的方法就是找出highest activation of receptive fields. 



## 4. Deep Eyes

这里呢，就会介绍DeepEyes，combine novel data- and filter-centric visualization techniques. 我们会从overview (4.1)开始，然后介绍一些细节(4.2-4.5)，这个training example呢，我们用的是MNIST，contains 60K training, 10K validation. 用的是SGD，还有MNIST-network in caffe. 这个network有两层CNN layers，分别有20个和50个filters。然后再连接FC，再分别有500个和10个filters。我们使用这个network仅仅只是用来证明我们的实现过程，和为了证明可重新生产性，我们使用caffe提供的architecture和parameters来做这个实验，尽管这个architecture并不是有非常好的classification performance。

### 4.1 Overview

<img src="https://user-images.githubusercontent.com/68700549/115265677-754fbb00-a105-11eb-81db-87316e496bb0.png" alt="WeChat Screenshot_20210419115018" style="zoom:50%;" />

这张图呢，就是这个system的overview。 在训练集中取出一部分来做这个filter activations，这部分就是mini-batches. 这个loss function的作用就是计算我的预测值跟真实值的差距。然后再根据loss function，和backpropagation来计算and update这个weight。DeepEyes是建立在理解receptive fields  $\pi (\delta _r^l)(x)$, （我们人类是可以可视化这个receptive fields的，也是可以明白这个的）和 the activation of filter functions $f_i^l (\pi (\delta _r^l)(x))$ 的关系是理解这个网络可以detect到什么pattern的东西的关键的基础上。

每一个mini-batches都要训练这个网络，我们在每一层中都要去取receptive fields的实例和与之对应的filter activation。对于每一个input，我们都会取至少50%的实例。我们用这个information可以用来create a continuously-updated dashboard, 这个dashboard可以告诉我们这层DNN layer detect到了什么。我们在做training的过程中，一般我们就会画loss and accuracy over time plot. 这里呢，作者做了一个补充，就是perplexity histograms，这个迭代作用就是identify a layer detect stable set of patterns. 至于detailed analysis，作者就用了三个tightly-connected visualization来做这个分析。这三幅图又分别的作用

![WeChat Screenshot_20210419132507](https://user-images.githubusercontent.com/68700549/115277791-b0a4b680-a112-11eb-96cb-7f23a6004250.png)

这个activation heatmap的作用就是identify degenerated filters. 这个input map的作用就是展示某一层filter activation显示出来的receptive fields，就是找这个关系，通过这个我们可以查看哪些个是没有detect的。Filter map的作用就是展示filter activate之间的相似性。然后Input and filter map结合起来看，就是可以identfy oversized and unnecessary layers。

### 4.2 Perplexity histogram as layer overview

就是说我们标准的做法就是使用loss-accuracy over time plot。但是嘞，这种方法仅仅只是给出了大体上的一个趋势，但是对于每一层的变化时没有一个是没有任何信息的。给出一个神经网络，我们是很有必要去分析里面的细节的，就比如说，stable set of patterns。作者的solution就是说认为每一个filter的作用就是用来identify一个特定的pattern。这就是相当于把每一个filter当做是一个classifier，然后我们去分析它的performance over time。就是说，我们想要分析这个classifier‘s ability对于检测这个pattern的稳定性，是增加的，还是减弱的，还是怎样的。如果是stable，就说明这个layer学的跟之前是一样的，如果是decrease，说明layer给network的信息是减少的，increase就表示增加

![WeChat Screenshot_20210419135419](https://user-images.githubusercontent.com/68700549/115281235-c5834900-a116-11eb-88b0-c3162c5f0500.png)

我们在mini-batch的每一个input中，我们随机地挑选出一些receptive fields还有他们对应的filter activation。就是上图的a,b所示。就是把一整列的receptive field都给提取出来。把这些activation都用L1-norm的方法给计算成probability vecotr $p \in R^{|F^l|}$ ，$|F^l|$ 表示这个layer有多少个filters。然后我们再根据这个probability来计算perplexity value。这个perplexity代表的是什么意思呢，这个概念是来自information theory的，表示这个layer对这个pattern detect地有多好。我们看到上图 标有②的位置，我们看到所有的都是相等的，那么perplexity value就会等于the number of filters $|F^l|$. 然后对于e，就是把每一个receptive field对应算出来的perplexity value $[1,|F^l|]$ for every layer $l$. 然后这个perplexity value就要累加起来。我们对每一个receptive field都这么做，从而可以看到整个layer的变化。我们看下面那张图，green bar表示增加，red bar表示减少。如果我们看到图中的perplexity的值较小，说明这个layer detect pattern的能力增加。但是这个histogram，并不是说一个bin表示一个receptive field的column，作者而是用了一个整体性概述，作者表明，这并不会影响我们去看这个layer是否稳定。其实，我们就是要通过这个perplexity histogram来检测这个layer是否稳定。

<img src="https://user-images.githubusercontent.com/68700549/115285126-81df0e00-a11b-11eb-9a51-4adba2bdded7.png" alt="WeChat Screenshot_20210419142814" style="zoom: 80%;" />    

我们来看看上图，这是MNIST-network 的Conv1 and Conv2. 我们在图中是可以看到一些峰值的，这些峰值表明一些patches 没有被detected到。 在10 iteration之后，在the first layer中，我们可以看到是往low perplexity中转变。然后我们再对比下conv1 and conv2的after 10 iteration的图，我们可以看到conv1的整体perplexity values是有一种下降的趋势的，然而在conv2中，after 10 Iteration，perplexity value整体有一种上升的趋势。这就表明第二层layer对于这种perplexity value整体下降这种趋势的变化会在第二层中detect这个pattern变得不那么具体。这个Histogram是每iteration都会有的图，我们就是要看这个layer是否稳定。我们再看after 80 iteration，第一层的layer还是unstable，但是对比下second layer，after 80 iteration，它的value是有下降的趋势的，表明，第二层会变得越来越详细。在300 iteration之后，基本趋于稳定了，可以进行细节研究了。

### 4.3 Activation Heatmap

等上面的histogram之后，我们知道layer已经趋于稳定了，我们就可以做细节分析，第一步可以做的就是detailed analysis。detailed analysis就是做activation heatmap. 我们把每一个filter都放在一个cell里，比如说conv1有20个filters，那么就会有20个cells。Activation heatmap的主要作用就是快速定位出degenerated filters. 我们旨在找出dead filters ，比如说：filters that are not activating to all instances, and filters that are activating to all instances. 这两种情况，说明这个filter就是废的，因为没有提供任何有价值的信息。我们可以看heatmap visualization，我们用两种方式来做这个heatmap，一种是maximum-activaiton, 另一种是frequency-Activation。

<img src="https://user-images.githubusercontent.com/68700549/115295412-b9ec4e00-a127-11eb-9bcc-93f429dd4dae.png" alt="WeChat Screenshot_20210419155540" style="zoom: 67%;" />

那我们怎么选这个maximum value呢？我们会随机地挑出一些receptive fields，在每一个filter都要挑出receptive field，然后选取最大值。我们计算activation的最大值$\mu _i^l$ , 每一个filter $f_i^l$ 都要计算，公式就是$\mu _i^l = max(f_i^l (\pi (\delta _t^l)(x)))$，选出来之后就可以进行visualize了。我们也用相似的方法来检测high Activation on every input，我们需要统计这个high activation 的frequency，然后在画这个frequency heatmap。 那我们怎么去定义这个high activation呢？就是上面我们有计算出$\mu _i^l$, 我们我们用$\beta * \mu _i^l$,  where $\beta=0.5$. 这里呢，就用两种不同的颜色来表示max- and fre- activation. 对于这个heatmap来说，我们要从整体来看，作者提供了一个option，就是让这个值是sorted，从小到大进行排列的，而且一定是以当前的值，比如说，我这是进行到第一个iteration，那么，这些值，就是来自于第一个iteration的值。为什么要做这个sorted呢？就是因为在训练的过程中，parameters是会变的，也就是weights是会改变的，这会导致一个filter的最大值是会改变的，这可能会导致misleading。比如说，一个filter，可能在之前是active，但是后来就“die”了。为了描述这个heatmap保持信息的可靠性，我们还得用另外一个值来衡量，就是这个activation的值高于max-activation的$\theta$倍，这里$\theta=0.8$. 上面这幅图就是after 100 iteration的，我们可以看到前10个的activation，都是没有超过max-activation的0.8，我们计算一下有多少个cells。low activation 表示这个filter不怎么provide information。然后我们可以再看看fre的图，这个是从大到小进行排列的，我们可以看到橙色框起来的部分，这表示high activation。两张图结合起来看，可以得出一个结论就是有些这个layer就是oversized，可以remove一些不必要的filter，这可以让training 更快，让network更小。对于最后一层的FC，4096 ($64*64$) filters，DeepEyes 采取的方法是用一个$5*5$的rectangle，然后制造一个$320*320$ 的image，来可视化这个heatmap。

### 4.4 Input map

在DeepEyes中，Input map是非常重要的。可以解决好多分析问题，之前提到的degenerated filter, pattern not detected, oversize, and unneecessary problems can be solved. Input map的产生是等stable layer出现之后。下面这张图就是MNIST-Network 的conv1, 对这一层的细节分析。作者把receptive fields视觉化成一个个的点，然后这个颜色就是看它原先的input是什么label。如果点在scatterplot中靠的很近的话就说明它们之间有相似的activation，也说明这些instance是similar input for current layer. 因为我们的neuronal receptive field是一个三维的数据，我们通过降维像PCA啊，达到二维的一个数据，然后还保留着neighborhood relationship。我们看这个input map，我们把所有的点都放在一块，做一个linked view，就是每一个点对应一个image patches。然后我们看这个scatterplot，如果颜色是很混合的，说明这一层在这iteration之前都不是非常有意义

![WeChat Screenshot_20210420101900](https://user-images.githubusercontent.com/68700549/115411934-e785d580-a1c1-11eb-9b1d-8ccab203c267.png)

我们再看下面这张图，这个是显示了4个filter activations，与此同时，把这四个也做一个input map,但是做的半透明，我们可以看到这些个半透明的，就是这四个filters的input map。然后再input map的上边再同样做一遍，只不过颜色变为黑白（展示intensity），这些黑圈，表示high activation。我们可以看到左下角那个，那些就是“Dead” filters，这样，我们就可以观察到哪些个filters是比较重要的。我们也可以在Activation heatmap中得到验证。通过这个方法，就可以找到一些degenerated filters. 我们看到这些个“dead” filters，这也表明这一层包含了一些不怎么需要的filters

![WeChat Screenshot_20210420104322](https://user-images.githubusercontent.com/68700549/115415865-400aa200-a1c5-11eb-9316-9dcf85e16223.png)

我们再看Max activation 这张图， 我们可以看到这个外围有一些不怎么明显的点，这些点被检测到是对应原图的背景，这就表明input map有一些点并没有产生strong activation

![WeChat Screenshot_20210420105831](https://user-images.githubusercontent.com/68700549/115418274-5e719d00-a1c7-11eb-8802-727331720b52.png)

这个input map是一种dataset-centric technique，有两方面得到提升。首先，就是去到receptive fields的sample，这会允许dataset-centric的视觉化，可以应用于CNN。其次的提升就是作者是使用neuronal receptive field，然后降维得到的结果, 而不是仅仅只是用the activation of filters under analysis。这就允许我们去分析a layer的输入与输出之间的关系，这在之前是做不到的。但是嘞，这两个features会给我们提供新的insights，同时也会增加计算量，因为我们要进行降维啊，在训练的过程中要sample到receptive filed然后再放进input map里啊。作者也用了几个降维的方法，比如说tSNE啊，是比较常用的。但是嘞，tSNE对于很大的数据量会比较慢，所以，最终，作者是采用HSNE (Hierarchical Stochastic Neighbor Embeddings), 这个方法就很快，几秒就搞定

### 4.5 Filter map

这个filter map的作用就是告诉我们filters之间的相似度。同样，我们把filters也visualize成scatterplot。如果许多filters 以相同的方式activate，这就表明这个layer包含了过多的filters。这里呢，我们会找到filters and label y之间的关系，因此，这个filter的颜色就是看这个是哪个training label activate这个filter最多次，然后这个circle size就表明这个filter跟那个label之间的关系强度，我们用大小，颜色来作为区分是因为比较简单操作，当然啦，你也可以使用其他方法，比如说，明亮度，饱和度啊等。如果出现一个簇有较大且颜色相近的话，就表明在这个阶段是可以进行分类的。据作者所了解，目前呢，就只有Rauber等人做出了类似的东西，他们使用pearson correlation between filter activations and the filters，这个方法的有点是可以cover所有的input但是不能用来分析CNN layers，是一个很大的缺点。为了克服这个缺点，作者采用的方法是计算相似度in a progressive way, 作者采用的是receptive field而不是所有的input。两个filters之间的相似度是用Jaccard Similarity来计算的。我们来看看这个公式，Jaccard similarity用$\phi _{i,j}$ and for two filters $i,j$ on layer $l$ , we can compute it as

$\phi _{i,j} = \frac{\Sigma_{r,x}min(f_i^l(\pi(\delta _r^l)(x)),f_j^l (\pi (\delta _r^l)(x)))}{\Sigma _{r,x}max(f_i^l(\pi (\delta _r^l)(x)),f_j^l(\pi (\delta _r^l)(x)))}$

Jaccard similarity的计算方式就是在两个filter之中，然后把一个一个receptive field的值进行对比，然后分子是取两者的最小值，分母是取两者的最大值，分子，分母都要分别求和，把两者之和进行相除，得到$\phi _{i,j}$,. 如果值越大，越接近于1，则说明两个filter是非常相似的，否则，不相似。

![WeChat Screenshot_20210420121246](https://user-images.githubusercontent.com/68700549/115429785-bd3c1400-a1d1-11eb-80c1-90e074a27bf9.png)

我们看上图的filter map，我们可以看到有两个filters在左边的filter activations那里有对应，可以看到，这两个filters是非常相像的。我们也要track是哪个label跟filter是最相关的，所以，对于每一个filter，都会计算vector $t_i^l \in R^d$ ,这个d表示的是一共有多少个labels，这个就把属于相同label的receptive field of instances给加起来，公式就是 $t_i^l (argmax(y))=\Sigma _{r,x}f_i^l (\pi (\delta _r^l)(x))$ . 每一个x都会对应一个label。因为我们是取最大值，也就是出现最多的那个label，然后在Filter map里面画出来的颜色就是这个label所对应的。然后这个点的size就看这个filter跟这个label的关联强度，这个关联强度的计算方式就是计算perplexity of the probabilities,也是用L1-norm计算出来的。这个size的大小跟perplexity value成反比，如果值越小，说明关联性越大。我们再看回上图中的Filter Map，我们看到一共有20个点，但是再对边下之前的input map，我们可以看到这些filters并没有把所有的label都给specify，说明这个layer并没有很好地完成classify。

### 4.6 From insights to network design

这里呢，就说一下是怎么从DeepEyes得到insights的。作者依旧用MNIST-Network来进行分析，我们得依旧先进行Iteration，知道看这个perplexity histogram稳定之后，作者就可以进行细节分析了。先分析Conv1, 然后Conv2, 紧接着是FC1，最后是FC2。

<img src="https://user-images.githubusercontent.com/68700549/115443692-8ec63500-a1e1-11eb-8a35-4bc36ae49b7c.png" alt="WeChat Screenshot_20210420140556" style="zoom:67%;" />

我们先来看看Conv1,我们可以看到input中，颜色的分类并不是非常地清晰，因为几乎杂合在一块了，就是非常乱。然后我们可以再看看Filter Map，还有选出来的几个filter，可以看到有几个filters并不是非常地active，我们看Activation Heatmap也可以得知。这就可以找到一些“Dead” filters，也可以得知，这个layer是oversized的，因为overly-redundant to non-existent patterns are learned by the filters. 

<img src="https://user-images.githubusercontent.com/68700549/115444292-5ecb6180-a1e2-11eb-9c4e-12dcd121c6a8.png" alt="WeChat Screenshot_20210420141139" style="zoom:67%;" />

接下来我们再看看Conv2。我们可以看到Input已经开始cluster了。比起Conv1,input map的receptive field会比Conv1多，因为Conv2有50个filters。我们再看Max activation，边缘区域不大active，表示background patches。

<img src="https://user-images.githubusercontent.com/68700549/115444797-0a74b180-a1e3-11eb-83cd-4f3e05ea6e01.png" alt="WeChat Screenshot_20210420141634" style="zoom:67%;" />

接下来我们来看看FC1，很明显，已经cluster了。我们看Max activation，可以看到基本已经覆盖了，表明每一个data input都有activate至少一个filter，因此，我们可以推出，每一个data input都是被identify到了的，在我们正式下结论之前，我们再看看FIlter Map，我们可以看到红色的点并不是非常多，为什么呢？我们可以倒回这一层的Input map中，可以看到红色跟绿色的点很近，表明这两个pattern很像。这说明FC1还不能完美地进行分类，所以还需要一层FC。我们可以再看看FIlter Activation，有一些也并不是非常黑色，表明，有些filters是可以去掉的。

<img src="https://user-images.githubusercontent.com/68700549/115445816-5aa04380-a1e4-11eb-947e-589a4a509947.png" alt="WeChat Screenshot_20210420142600" style="zoom:67%;" />

我们再看FC2，可以看到基本分类很不错了，基本上可以正确划分了，当我们去看FIlter activation那里的时候，可以看到有两个还是很强的active，表示还是会有一些network confused，那两个分别是digit-0 and digit-6. 根据DeepEyes对着四层的分析，作者把第一层的layer由原来的20个filters减到10个filters，然后把FC1由原来的500个减到100个，最后train出来的结果跟原来的结果都差不多 after 2000 Iterations。说明这个是有效的。

## 5. Test cases

这里呢，提供其他一些例子来帮助理解DeepEyes。最近几年呢，有很多的network architectures被提出来，其中AlexNet是其中一个比较出名的，AlexNet也很容易被改造，适用于其他问题中。AlexNet包含了5个卷积层，分别有96-256-384-384-256个filters，再接着3层的FCs，分别有4096-4096-1000个filters。下面这张图就是AlexNet

<img src="https://user-images.githubusercontent.com/68700549/115447180-2c236800-a1e6-11eb-95c4-f75e95e684bb.png" alt="AlexNet-1" style="zoom:67%;" />

这个这么巨大的network 具有超过16M的parameters。所以，对这个network进行减负还是很有必要的。这也说明了作者提出的方法在很多地方都是可以用的。在第一个例子中，作者先展示了DeepEyes是怎么帮助我们更好地理解DNN的fine-tune。第二个例子中，作者展示了通过DeepEyes来制作一个好的network architecture for medical imaging。

### 5.1 Fine tuning of  a deep neural network

训练一个大的DNN，需要很大的数据量，计算量和时间。为了解决这个困难，一个有效的方法就是找到一个已经训练好的network，然后进行fine-tune来解决不同的问题。FIne-tune的基本原理就是说像一些比较低级的filters，比如说detect color，edges，这些的，都是可以reused的。但是在多大程度上可以reuse呢？这我们是很不清楚的。在这例子中，作者为我们展示了DeepEyes是怎么帮助我们找到那些可以重复使用的filters，哪些又是要retrain的。作者使用AlexNet（provided in caffe）。AlexNet的原来作用是用于image-classification，现在，作者要通过fine-tune AlexNet来做image-style recognition. 在这个例子中，作者把最后一层的1000个filters （原来detect 1000个objects）变为20个filters (用来detect20种不同的styles)，比如说“romantic”，“vintage”，“geometric composition”

<img src="https://user-images.githubusercontent.com/68700549/115453423-a0153e80-a1ed-11eb-8d97-6c9e93fd59c4.png" alt="WeChat Screenshot_20210420153220" style="zoom:67%;" />

这个network还是需要100个iterations，用K40 GPU来训练还是需要超过7小时的，最终得到的accuracy is 34.5%。

首先可以确定的就是这个color and edges detectors是有效的，就比如说看下面这张图，可以看到蓝色和竖竖的边是可以detect到的。

<img src="https://user-images.githubusercontent.com/68700549/115476339-a7e5da80-a20f-11eb-9821-7ed30863c544.png" alt="WeChat Screenshot_20210420193542" style="zoom:80%;" />

第五个卷积层，我们可以看到有很多的input patches是没有被activated的，这就表明这一层是有很多问题的。我们可以看下面这张图，了解到“geometric composition”这种图是比较难被detected到的，因为没什么黑点在那里。因为原来的training network就没有太多的filters可以detect到这种类型的图，就导致很难区分这种图。我们也可以再看看这个“face” filter，这就表明face是可以被detect到的，但是face会跟各种不同的style结合在一起，说明这个filter擅长detect face而不擅长做style。更多的研究要去看transfer learning。DeepEyes 只是一个辅助工具，可以进行使用。

<img src="https://user-images.githubusercontent.com/68700549/115476473-f09d9380-a20f-11eb-8e4a-c96d0c8a5b70.png" alt="WeChat Screenshot_20210420193749" style="zoom:67%;" />

### 5.2 Mitotic figures detection

在medical imaging领域，作者在这领域也提出也可使用。在这领域，DNN来做recognition problems。DeepEyes的作用就是帮助我们去理解其中network的行为。The number of nuclei separations in tumor tissue is a measurement for tumor aggressiveness. 通过radiotherapy treatment，会有一种histological images，会有专门的pathologist为你分析。Nuclei separation, 也叫mitotic figures。下面这张图就展示的是mitotic figures, label as Negative.

![WeChat Screenshot_20210420202043](https://user-images.githubusercontent.com/68700549/115479214-f4ccaf80-a215-11eb-9421-e483c2387631.png)

算这个number of mitotic figures的用意是要知道radiation 的用量帮助治疗tumor。然而，这是个重复而且是无聊的工作，完全可以用DNN来替代。因此呢，作者找到Veta et al. 这些人提出的DNN trained on AMIDA dataset来检测这个mitotic figures的数量。这个DNN有4个卷积层，分别有8-16-16-32个filters，两个FCs 分别有100-2个filters。

<img src="https://user-images.githubusercontent.com/68700549/115481403-8e965b80-a21a-11eb-8a47-d8fcf8ad409d.png" alt="WeChat Screenshot_20210420205359" style="zoom:50%;" />

我们可以看到基本就分红蓝两种颜色，分别detect深红和浅红两种颜色，这两个features也是非常重要的。

<img src="https://user-images.githubusercontent.com/68700549/115482238-319ba500-a21c-11eb-80de-8c60fa02c7dc.png" alt="WeChat Screenshot_20210420210543" style="zoom: 50%;" />

上面那张图，就是第一层FC，可以看到有比较明显的分开，说明到这一层为止classification都做的很好。因此，可以判断说第二层的FC是不大需要的。因此，作者就把第二层直接给去掉了，直接把第一层和predict 层连接了起来，结果accuracy可以达到95.9%，跟之前的network architecture的结果差不多，但是去掉之后，速度会更快。

<img src="https://user-images.githubusercontent.com/68700549/115491796-a11a9000-a22e-11eb-9635-06928c962683.png" alt="WeChat Screenshot_20210420231732" style="zoom:67%;" />

## 6 Implementation



## 7 Conclusions

