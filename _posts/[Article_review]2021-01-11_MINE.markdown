---
layout: post
title:  "Mutual information neural estimation"
subtitle:   "MINE"
categories: Article_review
tags: Representation_learning

comments: true
---

안녕하세요! 이번에 정리할 논문은 [Mutual information neural estimation](http://proceedings.mlr.press/v80/belghazi18a/belghazi18a.pdf) 제목의 논문입니다. 2018년에 ICML에서 발표되었습니다. 
본격적으로 소개하기 전에 간략히 논문을 설명하면 본 논문은 신경망 기반의 상호정보량 추정 모델 Mutual information neural estimator (__MINE__)을 제안했고 MINE으로 GAN의 mode collapse을 해결했고 information bottleneck에 적용하여 좋은 성능을 나타냈습니다. 

Introduction
---
상호정보량은 두 확률 변수 X, Z의 의존도를 수치화한 값으로 선형 관계 뿐만 아니라 비선형적인 통계적 관계도 포착하여 true dependence의 지표로 활용됩니다. 

![그림1](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig1.png)

그럼에도 불구하고 상호정보량은 역사적으로 계산하기 어려웠습니다. 이산 확률 변수의 경우에만 정확히 계산할 수 있고 연속 확률 변수는 확률분포가 알려진 경우에만 계산할 수 있습니다. 
하지만 우리는 실제 데이터에서 확률 변수의 샘플들을 다루기 때문에 정확한 확률 분포를 알 수 없습니다. 
비모수 추정 방법론들이 있지만 샘플 사이즈나 차원 크기에 scalable하지 않기 때문에 일반적으로 활용할 수 없습니다. 

본 논문은 MINE이라는 신경망 기반의 상호정보량 추정 모델을 제안합니다. MINE은 KL divergence의 dual representation을 활용하여 이 dual representation의 하한 (lower bound)를 최대화합니다. 
MINE은 scalable, flexible, trainable by back-propagation, theoretically robust하고 GAN의 mode collapse 문제를 해결하고 information bottleneck 방법에 적용하여 우월한 성능을 보였다고 합니다. 

Related work
---
### Dual representations of the KL-divergence

MINE의 key technique은 dual representation of the KL-divergence, 한글로 KL-divergence의 쌍대 표현이다. 그 중 더 tight한 추정치를 낳는 Donsker-Varadhan representation (DV)을 활용하여 설명한다. 

#### The Donsker-Varadhan representation

![그림2](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig2.png)

theorem 1을 해석하면  ![formular](https://render.githubusercontent.com/render/math?math=\mathbb{P})와 ![formular](https://render.githubusercontent.com/render/math?math=\mathbb{Q})의 KL divergence는 모든 어떤 domain ![formular](https://render.githubusercontent.com/render/math?math=\Omega)에서 실수 공간 ![formular](https://render.githubusercontent.com/render/math?math=\mathbb{R})로 매핑하는 함수 T에 대해 
확률 변수 ![formular](https://render.githubusercontent.com/render/math?math=\mathbb{P})에 대한 T의 평균 - 확률 변수 ![formular](https://render.githubusercontent.com/render/math?math=\mathbb{Q})에 대한 ![formular](https://render.githubusercontent.com/render/math?math=e^T)의 평균에 log를 취한 값을 뺀 것의 supremum입니다. 
supremum은 least upper bound로 정의되는데 해석학에서 처음 배운 개념인데 간단히 maximum으로 이해하셔도 무방합니다. 
즉 ![formular](https://render.githubusercontent.com/render/math?math=D_{KL}(\mathbb{P}||\mathbb{Q}))은 우변의 표현의 supremum과 같기 때문에 특정 T를 제외한 다른 후보 함수들의 값은 
![formular](https://render.githubusercontent.com/render/math?math=D_{KL}(\mathbb{P}||\mathbb{Q}))보다 작습니다 (하한의 역할). 

![그림3](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig3.png)

비슷한 논리로 모든 T의 부분집합인 
![formular](https://render.githubusercontent.com/render/math?math=\mathit{F})에 대해 supremum을 취하면 
![formular](https://render.githubusercontent.com/render/math?math=D_{KL}(\mathbb{P}||\mathbb{Q}))보다 작을 것입니다. 

![그림4](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig4.png)

위 수식은 f-divergence 기반의 표현이고 우변의 - 이후의 항이 theorem 1의 식의 - 이후의 항보다 크기 때문에 loose한 하한을 형성합니다. 
 
The mutual information neural estimator
--- 
### 1. Method

Related work의 DV representation을 기반으로 MINE의 아이디어는 **함수 T를 신경망 ![formular](https://render.githubusercontent.com/render/math?math=T_\theta)으로 선택하자!** 라는 것입니다. 
상호정보량을 추정하기 위한 함수로 활용되는 신경망을 *statistics network*라고 부릅니다. 

![그림5](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig5.png)

I(X;Z)는 상호정보량 실제값이고 ![formular](https://render.githubusercontent.com/render/math?math=I_\theta (X,Z))는 
![formular](https://render.githubusercontent.com/render/math?math=T_\theta)로 추정한 상호정보량입니다. 위 수식의 기댓값은 결합/주변 확률분포에서 뽑은 샘플로 추정됩니다. 
![formular](https://render.githubusercontent.com/render/math?math=\mathbb{P}_{XZ})의 샘플은 X와 Z를 같이 sampling하여 추출하고 
![formular](https://render.githubusercontent.com/render/math?math=\mathbb{P}_X \times \mathbb{P}_Z)의 샘플은 X는 결합 확률 분포에서 sampling한 X를 그대로, Z는 독립적으로 sampling하여 주변 확률 분포의 곱의 샘플을 추출합니다. ![formular](https://render.githubusercontent.com/render/math?math=T_\theta)는 
![formular](https://render.githubusercontent.com/render/math?math=I_\theta (X;Z))를 gradient ascent에 의해 최대화하는 방향으로 학습됩니다. 

하지만 ![formular](https://render.githubusercontent.com/render/math?math=I_\theta (X;Z))의 추정치도 확률 변수를 바로 다루는 것이 아니고 실현된 샘플을 통해 상호정보량을 추정하게 됩니다. 본 논문은 n개의 샘플로 추정된 ![formular](https://render.githubusercontent.com/render/math?math=I_\theta (X;Z))의 추정치를 
![formular](https://render.githubusercontent.com/render/math?math=\hat{I(X;Z)}_n)으로 정의합니다. 즉 우리가 MINE 알고리즘을 실행하면 구할 수 있는 추정값이 
![formular](https://render.githubusercontent.com/render/math?math=\hat{I(X;Z)}_n)입니다. 

![그림6](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig6.png)

MINE의 알고리즘은 다음과 같습니다. 

![그림7](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig7.png)

- ~~Draw n samples from the Z marginal distribution 에서 n이 b의 오타이지 않을까 싶습니다...~~ 

### 2. Correcting the bias from the stochastic gradients

b개 샘플의 미니배치로 계산된 목적함수의 gradient는 full batch gradient의 biased estimate입니다. bias는 분모의 추정치에 exponential moving average (지수 이동 평균)를 취함으로 줄어든다고 합니다. 지수 이동 평균은 t step의 값은 t-1 step의 값 X (1-a)(t step의 추정치)로 재귀적으로 지수배로 이전 값을 고려하는 방법입니다. 

![그림8](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig8.png)

### 3. Theoretical properties

이 섹션은 MINE의 추정치가 과연 실제값에 수렴할까에 대한 근거를 제시합니다. 

#### Consistency
![그림9](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig9.png)

Definition 3.2는 극한의 개념으로 strong consistency를 정의합니다. 간단히 설명하면, 어떠한 정확도 ![formular](https://render.githubusercontent.com/render/math?math=\epsilon)을 가지고 와도 실제값 I 에 ![formular](https://render.githubusercontent.com/render/math?math=\epsilon)보다 가까운 추정치 
![formular](https://render.githubusercontent.com/render/math?math=\hat{I}_n)이 존재한다면 ![formular](https://render.githubusercontent.com/render/math?math=\hat{I}_n)은 
strongly consistent하다고 정의합니다. Consistency에 대한 질문은 approximation에 대한 문제과 estimation에 대한 문제로 나눠집니다. 

![그림10](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig10.png)
![그림11](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig11.png)

- Consistency: 두 확률 변수의 실제 상호정보량과 n개의 샘플을 통해 신경망으로 추정한 값이 근사한가? 
- Approximation: 두 확률 변수의 실제 상호정보량을 근사하게 추정할 수 있는 신경망이 존재하는가? 
- Estimation: 한 신경망이 주어졌을 때 신경망이 이론적으로 계산하는 상호정보량과 n개의 샘플을 통해 계산된 상호정보량 값이 근사한가? 

간단히 말하면 approximation은 __상호정보량을 신경망으로 근사할 수 있는가?__, 
estimation은 __이론적으로 신경망으로 근사할 수 있다면 실제 샘플을 통해 정확하게 추정이 가능한가?__ 에 대한 질문입니다. 
approximation은 신경망의 universal approximation theorem으로 설명이 가능합니다. 이론적으로 신경망은 모든 형태의 함수를 근사할 수 있기 때문에 정확한 상호정보량을 추정할 수 있습니다. estimation에 대해서도 lemma를 통해 증명하고 있습니다 (supplementary materials). 그래서 approximation과 estimation이 가능하기에 의미적으로/수식적으로 MINE은 strongly consistent하다고 증명할 수 있습니다. 

#### Sample complexity

이 섹션에서는 신경망이 특정 조건들을 만족할 때 원하는 정확도로 추정하기 위한 샘플 개수를 제시합니다. 만족해야 하는 가정들은 신경망이 M-bounded, L-Lipschitz continuous, bounded domain 해야 한다는 것들이 있습니다. 

![그림13](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig13.png)

Empirical comparisons
---
본 섹션에서 MINE과 기존 k nearest neighbors 기반의 상호정보량 추정 방법론을 비교하여 MINE이 기존 방법론보다 실제 상호정보량을 정확히 추정함을 보입니다. MINE은 deterministic한 함수로 표현된 세 변수의 상호정보량을 계산하여 MINE이 deterministic nonlinear transformation에 상관없이 일정한 추정치를 산출함을 보입니다. 

![그림14](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig14.png)
![그림15](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig15.png)

Applications
---
### Maximizing mutual information to improve GANs

MINE을 GAN에 적용해 GAN의 mode collapse 문제를 해결할 수 있음을 보여줍니다. mode collapse란 생성모델에서 실제 데이터는 여러 개의 mode를 가짐에도 불구하고 학습한 생성모델이 그 중 몇 개의 mode만 생성하는 현상을 말합니다. 

GAN은 생성 모델 중 하나로 기본적으로 생성자와 분류자, 두 신경망으로 구성됩니다. 분류자는 실제 데이터와 가짜 데이터를 0~1의 확률로 구분하도록 학습하고 생성자는 분류자가 구분하기 어렵도록 가짜 데이터를 실제 데이터처럼 생성하도록 학습합니다. 생성자와 분류자가 서로 경쟁하듯 학습되며 생성자는 실제 데이터 분포를 모사하게 됩니다. 두 신경망은 value function을 통해 학습됩니다. 

![그림16](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig16.png)

일반적으로 mode collapse를 줄이기 위해 우변의 두 번째 항인 negative entropy of generator's loss, 생성자 손실 항의 음의 엔트로피를 조절합니다. 본 논문은 이 항 대신 상호정보량 항을 사용합니다. 

MINE을 적용하기 위해 InfoGAN의 구조를 활용합니다. InfoGAN은 일반적인 GAN과 달리 샘플을 생성하기 위한 prior로 noise와 code variable을 결합해서 활용합니다. code variable은 데이터 분포 내에 잠재된 구조적 특징을 나타낸다고 가정합니다. 본 논문은 code variable c와 생성된 샘플의 상호정보량을 최대화하기 위해 c와 생성된 샘플의 상호정보량 항을 목적식에 추가했습니다. 

![그림17](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig17.png)

Spiral data와 25 Gaussians data를 생성하여 실험하였고, GAN+MINE이 GAN 단독 모델보다 mode collapse 현상을 더 줄였다는 것을 보였습니다. 
또한 MNIST 데이터의 숫자를 랜덤하게 3개씩 겹친 stacked MNIST 데이터 (1000개의 모드를 가짐)에서도 GAN+MINE이 1000개의 모드를 잘 찾았다는 것을 보였습니다. 

![그림18](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig18.png)
![그림20](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig20.png)

### Maximizing mutual information to improve inference in bi-directional adversarial models

Adversarial bi-directional model은 GAN에서 생성 뿐만 아니라 역으로 샘플에서 추론을 하도록 학습되는 모델입니다. 
p(x,z) = p(z|x)p(x)와 q(x,z) = q(x|z)p(z) 즉 ![formular](https://render.githubusercontent.com/render/math?math=x \rightarrow z) 방향의 추론과 
![formular](https://render.githubusercontent.com/render/math?math=z \rightarrow x) 방향의 생성이 동시에 잘 되도록, 
![formular](https://render.githubusercontent.com/render/math?math=p(x, z) \approx q(x,z))가 되도록 인코더와 디코더를 동시에 학습합니다. 하지만 실제로는 인코더-디코더 구조가 reconstruction을 잘 못한다고 합니다 (reconstruction은 generative model과 inference model에 다 중요한 성질). 그래서 z와 x 간의 상호정보량 항을 목적식에 추가하고 MINE을 적용했습니다. 그 결과 기존 방법론들과 competitive한 reconstruction 결과를 보였습니다. 

### Information bottleneck 

Information bottleneck (IB)란 output y에 대해 input X가 가지고 있는 정보를 추출하기 위한, 적절한 표현 Z를 추출하기 위한 정보 이론 기반의 방법론입니다. 최적의 Z는 Y를 예측하는데 불필요한 정보를 줄이고 X를 압축하여 얻은 잠재 표현입니다. 결국 IB는 ![formular](https://render.githubusercontent.com/render/math?math=X \rightarrow Z \rightarrow Y)의 Markovian 구조를 가진 인코더 (q(Z|X)를 추정)이고 이는 IB Lagrangian을 최소화함으로 학습됩니다. 이는 일반적인 cross entropy loss에 X와 Z의 상호정보량 항을 더한 것으로 Z는 Y를 예측하기 위한 정보를 담고 X에 불필요한 정보를 최소화하도록 학습하게 됩니다. 

![그림19](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig19.png)

여러 IB 모델들과 비교한 결과 MINE이 Y를 예측하는데 좋은 성능을 보였고 이는 Deep Variational Bottleneck 방법론과 comparable한 결과입니다. 

![그림21](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/mine_fig21.png)
