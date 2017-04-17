

# MachineLearningCampProposal
This is a proposal project to attend Machine Learning Camp

I am going to reimplement and extend the paper "Hierarchical Question-Image Co-Attention for Visual Question Answering", appeared NIPS 2016. I got interested in this paper as well as attention based models after reimplementing the paper "Show, attend and tell: a neural image caption generation with visual attention" by Xu et al [5].
The details of the proposal implementation and extension will be illustrated in section 2 and section 3. 

## 2. Hierarchical Question-Image Co-Attention for Visual Question Answering

Recently, deep convolution neural networks (CNNs) and recurrent neural networks (RNNs) have shown promising results in many computer vision and natural language processing tasks such as image captioning, visual question answering, and machine translation. Visual Question Answering (VQA) has emerged as a challenge multi-discipline research problem because it requires both textual and visual information. Many research groups [2][3][4] have utilized CNNs net called VGG[6] and ResNet[7] to combine with a variant of RNNs called LSTM to tackle VQA. These networks learn to map the visual and textural features to a joint semantic space to correctly answer visual questions about an image. Recently, visual attention architectures [3][4] have shown a great potential to tackle many VQA datasets including COCO, VQA. The paper which I am going to reimplement and extend lies in attention based approach. 

The key contributions are listed as below:
### Co-Attention:
Previous works focus on visual attention, the model has a co-attention mechanism to attend both visual and textural information. In this paper, two co-attention mechanisms were proposed including parallel co-attention and alternating co-attention. 

The parallel co-attention computes the similarity between image and question features at all pairs of image regions and question location simultaneously. Given an image feature map $V = {{v_1},\dots,{v_N}}$ and question representation $Q = {{q_1}, \dots, {v_N}}$, the affinity matrix $\mathcal{C}$ is calculated as follows:
  
\begin{equation}
\mathcal{C} = tanh(Q^T\W V)
\end{equation}
 

Then, using the above affinity matrix to compute the attention weights with respect to visual features and textural features using softmax function
  H^v =tanh(W_v V + (W_q) Q C)  
  
  $$\alpha^v = softmax(\mathcal{U_v}H^v)$$
  
  $H^q =tanh(W_q Q + (W_v) Q C^T)$  
  
  $\alpha^q = softmax(\mathcal{U_q}H^q)$
  
  Then, attention vectors for visual and textural information are calculated as weighted sum of image features and textural features.
  
  $hat{v} = \sum_{n=1}^N v_n \alpha_n^v$
  
  $hat{q} = \sum_{q=1}^T q_t \alpha_t^q$
  
Similarly, alternating co-attention estimates the attention vectors as parallel mechanism but in a sequential order.

### Question Hierarchy:
The paper build a hierarchical architecture that co-attends image and question at three levels: word level, phrase level and question level. They utlized embedding matrix, 1-D convolutional neural networks and LSTM to represent questions in word level, phrase level and question level, respectively. 

### Encoding for predicting answers
The paper utilized multi-layer perceptrons (MLP) with inputs from above attented visual and textural features to recursively encode features to predict answer. 
     $h^w = tanh (W_w (hat{q^w} + hat{v^w}))$
     
     $h^p = tanh (W_p [(hat{q^p} + hat{v^p}), h^w])$
     
     $h^s = tanh (W_s [(hat{q^s} + hat{v^s}), h^p])$
     
     $p = softmax (W_h h^s)$
     
p is the probability of the final answer

## 3. Extension
### Archicture Extension

The above paper have shown a high accuracy rate in VQA tasks multiple-choice and open-ended questions. However, in both types of questions, the answer is still single word or just classification task. 

My main propose is to replace the above architectures in the last layer of encoding for predicting answers with an two-layer LSTM to have a sentence answer. The two-layer LSTM will be fed with the above attention features vectors at different levels (word level, phrase level) as extra inputs. The outputs should be a sentence to answer open-ended questions. For example, a question raised by users "What is this scene?", the answer should be "It is a diner party" (sentence) instead of "party" or "people" or "dog" (single word). 

$i_t = f_i(x_{t-1}, h_{t-1}, hat{v}, hat{q}) $

$f_t = f_f(x_{t-1}, h_{t-1}, hat{v}, hat{q}) $

$o_t = f_o(x_{t-1}, h_{t-1}, hat{v}, hat{q}) $

$g_t = f_g(x_{t-1}, h_{t-1}, hat{v}, hat{q}) $

$c_t = f_t \bigodot c_{t-1} + i_t \bigodot g_t $

$h_t = o_t \bigodot tanh{c_t} $

where f_i, f_f, f_o, f_g are non-linear functions of the inputs.

### New Dataset




## Reference

1. Jiasen Lu, Jianwei Yang, Dhruv Batra, Devi Parikh. Hierarchical Question-Image Co-Attention for Visual Question Answering. Neural Inforamtion Processing Systems (NIPS) 2016. 
2. Hyeonwoo Noh, Paul Hongsuck Seo, Bohyung Han. Image Question Answering using Convolutional Neural Network with Dynamic Parameter Prediction. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, US, 2016
3. Z. Yang, X. He, J. Gao, L. Deng and A. Smola. "Stacked Attention Networks for Image Question Answering," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 21-29.
4. Kevin J. Shih, Saurabh Singh, Derek Hoiem. Where To Look: Focus Regions for Visual Question Answering. CVPR, 2016
5. K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. Zemel, Y. Bengio. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention". ICML 2015
6. VGG
7. Resnet
8. COCO
9. VQA

