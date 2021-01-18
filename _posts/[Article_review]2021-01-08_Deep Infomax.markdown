---
layout: post
title: [ICLR 2019] Learning deep representations by mutual information estimation and maximization
subtitle: Deep Infomax, Mutual information maximization, MINE
gh-repo: hwseo95/hwseo95.github.io
gh-badge: 
tags:
comments: true
---

안녕하세요! 제가 처음으로 정리할 논문은 [Learning deep representations by mutual information estimation and maximization](https://arxiv.org/pdf/1808.06670.pdf) 제목의 논문입니다. 2019년에 ICLR에서 발표되었습니다. 이 논문을 공부하면서 PR12 논문읽기 모임의 발표 영상 자료와 저자들이 작성한 마이크로소프트 블로그를 많이 참고하였습니다. 링크는 아래와 같습니다.

[PR12 영상자료: https://www.youtube.com/watch?v=YNicvevmByo&t=929s](https://www.youtube.com/watch?v=YNicvevmByo&t=929s)  
[마이크로소프트 블로그: https://www.microsoft.com/en-us/research/blog/deep-infomax-learning-good-representations-through-mutual-information-maximization/](https://www.microsoft.com/en-us/research/blog/deep-infomax-learning-good-representations-through-mutual-information-maximization/)

## Introduction

딥러닝의 한 주요 목적은 인풋 데이터의 유용한 잠재 표현을 학습하는 것입니다. 
잠재 표현을 학습하는 방법에는 지도 학습 세팅에서 클래스 구분에 적합하도록 학습하거나 비지도 학습 세팅에서 인풋을 복원하기 위한 잠재 표현을 학습 (Autoencoder)하는 등 다양한 목적으로 학습할 수 있지만, 
본 논문은 인풋의 정보를 최대한 보존 (상호정보량을 최대화)하는 잠재 표현을 학습하려 합니다. 
인풋과 아웃풋 (잠재표현)의 상호정보량 (mutual information, MI)을 최대화하는 목적은 새로운 것이 아닙니다. 
1990년대부터 Infomax principle이 제안되었지만 딥러닝 구조에 바로 적용하기에는 한계가 있었습니다.
상호정보량은 고차원의 데이터, 연속 변수에 대해서 계산하기 힘들다고 알려져 있습니다. 
하지만, 2018년에 [MINE](http://proceedings.mlr.press/v80/belghazi18a/belghazi18a.pdf) 등 신경망 기반의 상호정보량 추정 모델이 개발됨에 따라 Infomax principle을 신경망에 적용할 수 있게 되었습니다.

본 논문은 인풋의 정보를 최대한 보존하는 잠재 표현을 학습하는 방법론인 Deep Infomax (DIM)를 제안합니다. 
추후에 설명하겠지만 인풋의 정보를 보존하는 것만으로는 표현이 한계점을 갖기 때문에 DIM은 추가로 1) 데이터의 지역적인 정보를 보존하고 2) 특정 통계적 정보를 갖도록 학습합니다.
본 논문은 이미지 데이터에 대해서 주로 설명하지만 모든 temporal data에 다 적용가능하다고 합니다.

## Related work

상호정보량은 비지도 학습에서 중요한 역할을 합니다. 
$$I(X, Y) = D_{KL}(P(X, Y)|P(X)P(Y)) = \integral_x \integral_y p(x, y)log\frac{p(x,y)}{p(x)p(y)}$$
상호정보량은 정보이론 관점에서 Y를 알게 됨으로 X에 대해 알게 된 정보의 양으로 해석할 수 있고 통계학 관점에서 X와 Y의 상호 의존성 (dependence)의 정도를 수치화한 값입니다. 상호정보량은 두 확률변수가 독립하다면 (independent) 0으로 최소, 두 확률 변수가 어떤 함수의 형태로 명시적으로 표현될 수 있다면 최대가 됩니다.

2018년에 제안된 MINE: Mutual information neural estimator는 연속 변수의 상호정보량을 추정하도록 학습을 하는데 DIM은 MINE을 프레임워크 안에서 활용합니다.
MINE에서는 KL divergence 기반의 상호정보량 표현으로 값을 추정하지만 본 논문에서는 Jensen-Shannon divergence (JSD) 기반의 표현으로 상호정보량을 추정하고 최대화하는데 이로서 Deep infomax가 다양한 상호정보량 추정 모델과 결합 가능하다고 주장합니다.

DIM과 비슷한 목적으로 제안된 Contrastive Predictive Coding (CPC)는 DIM처럼 상호정보량 개념을 활용한 접근법으로 잠재 표현을 학습합니다. 
두 방법론이 다른 점은 CPC는 각 local 변수를 sequential하게 처리하여 각 local 변수마다 summary feature (잠재 표현이라고 생각하면 됩니다)를 만들지만 DIM은 모든 local 변수들을 한꺼번에 모아 한 summary feature를 학습합니다.
local한 변수는 이미지에서는 픽셀 또는 패치, sequential data (문자열, time series)에서는 각 timestamp의 데이터를 지칭합니다. 
즉 CPC는 각 timestamp마다 잠재 표현을 학습하고, DIM은 각 timestamp (이미지 픽셀)을 집합하여 한 잠재 표현을 학습합니다.

## Deep Infomax

Deep Infomax는 잠재 표현을 학습하는 인코더, 인풋의 global 정보를 보존하기 위한 Discriminator (MINE), local 정보를 보존하기 위한 Discriminator (MINE), 잠재 표현에 특정 통계적 성질을 반영하기 위한 Discriminator로 총 1개의 인코더와 3개의 분별자로 구성됩니다.
이 구조를 통해 DIM의 인코더는 두 가지 목적을 따라 학습됩니다.

- Mutual information maximization:   
  인풋 데이터와 상호정보량이 최대인 아웃풋 (잠재 표현)을 만들도록, 목적에 따라 데이터 전체의 정보를 보존하거나 데이터의 부분적인 공간의 정보를 보존하도록

- Statistical constraints:   
  산출된 잠재 표현이 특정 통계적 성질(예. 독립성)을 반영하도록. 다시 말하면, 인코더로 매핑된 데이터가 특정 분포를 갖는 사전 확률 분포와 유사하도록

1. Mutual information estimation and maximization

![그림1](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/deepinfomax_fig1.png)

이미지 (다른 temporal data에도 적용 가능) $X$는 Figure 1처럼 여러 feature map을 통해 $ M \times M $ feature vector로 인코딩되고, 한 feature vector $Y$로 축약됩니다 $y=E_\psi(x)$. $y$는 인코더에서 $X$와의 상호정보량 $I(x, E_\psi(x))$가 최대가 되는 파라미터 $\psi$를 찾도록 학습하게 됩니다. 

![그림2](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/deepinfomax_fig2.png)

상호정보량 최대화를 위한 구조는 MINE이라는 신경망 기반의 상호정보량 추정 모델을 따릅니다. Figure 2에서 MINE 적용 원리를 알 수 있습니다. 
- discriminator (MINE)에게 feature vector $Y$와 한 이미지의 $M \times M$ feature map을 입력합니다. 
- discriminator는 같은 이미지에서 나온 $Y$와 feature map라고 판단하면 높은 상호정보량 (real score)을, 서로 다른 이미지의 $Y$와 feature map이면 낮은 상호정보량 (false score)을 추정합니다. (Figure 2에서 상호정보량 추정값을 score라고 부르는 것 같습니다.)
- discriminator와 인코더가 같이 학습이 되면서 discriminator가 같은/다른 이미지의 $Y$와 feature map인지를 잘 구분하도록 인코더가 feature vector $Y$를 학습합니다.
위의 구조로 $Y$는 $X$ 전체의 정보를 보존하도록 학습되고 논문에서 global DIM이라고 명시합니다.

논문 3.1 섹션의 후반 내용은 MINE에 대한 수식적인 이해가 필요한 내용입니다. 관심 있으신 분들은 읽어보심을 추천드립니다.  
MINE은 샘플이 결합 확률 분포 $\mathbb {J}$, $P(X, Y)$의 샘플인지 주변 확률 분포 $\mathbb{M}$, $P(X)$, $P(Y)$의 샘플인가를 구분하는 분류기를 학습하면서 상호정보량을 추정합니다. MINE은 Donsker-Varadhan representation (DV) 기반의 상호정보량의 하한(lower bound)을 최대화하여 $I(X, Y)$의 추정치를 찾아갑니다. 

$$I(X;Y)>= \hat{I}_\omega^(DV)(X;Y):=\mathbb{M}_\mathbb{J}[T_\omega(x, y)] - log\mathbb{E}_\mathbb{M}[e^{T_\omega(x,y)}]$$

여기서 $T_w$가 상호정보량을 추정하도록 학습되는 MINE의 신경망입니다.
그래서 DIM은 상호정보량을 추정 및 최대화를 동시에 진행하며 인코더를 학습합니다. 

$$(\hat{\omega}, \hat{\psi})_G = argmax_{\omega, \psi}\hat{I}_{\omega, \psi}(X;E_{\psi}(X))$$

여기서 $G$는 global을 의미하는데, 데이터 $X$ 전체와 잠재 표현 $E_\psi(X)$의 상호정보량을 최대화하는 목적이기 때문입니다.

DIM은 상호정보량을 최대화하는데 관심이 있지, 정확한 상호정보량을 계산하는 것에 관심이 없기 때문에 다른 표현 기반의 상호정보량 추정 목적식 (정확히 말하면 하한)을 활용할 수 있습니다. 본 논문은 Jensen-Shannon divergence (JSD) 기반, Noise-Contrastive Estimation (NCE) 기반의 목적식을 활용하여 상호정보량을 최대화하였습니다. 

2. Local mutual information maximization

![그림4](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/deepinfomax_fig4.png)

Global DIM은 한 데이터 (이미지) 전체와 잠재 표현과의 상호정보량을 최대화하지만 이는 task에 따라 부적합합니다. 예를 들어, 이미지에서 이미지와 관련 없는 배경 등 pixel-level noise에 대한 정보들을 인코딩하는 것은 유용하지 않을 수 있습니다. 위 그림은 고양이를 나타내는 이미지인데, 뒷 배경은 고양이와 관련없는 정보들입니다. 관련없는 정보들의 인코딩을 막고 이미지 전체에 공유된 정보를 학습하기 위해 Local DIM 구조를 제안합니다.

![그림3](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/deepinfomax_fig3.png)

Local DIM의 discriminator는 global DIM의 discriminator가 feature map 전체와 $Y$의 score를 계산하는 것과 달리 feature map의 각 local part (pixel 또는 patch)와 $Y$의 score를 모두 계산하여 평균을 취합니다. Local DIM은 global 정보인 $Y$와 local 정보인 각 pixel or patch의 상호정보량을 계산하여 이미지에 공유된 정보를 학습하고 불필요한 정보의 인코딩을 막습니다. 

$$(\hat{\omega}, \hat{\psi})_L = argmax_{\omega, \psi}\frac{1}{M^2}\sum_{i=1}^{M^2}\hat{I}_{\omega, \psi}(C_{\psi}^{(i)}(X);E_{\psi}(X)) $$

3. Matching representations to a prior distribution

잠재 표현이 원 데이터의 정보를 최대한 보존하는 것도 중요하지만 주어진 task에 따라 independent, compact or disentangled 등 통계적 제약을 만족해야 할 수 있습니다. DIM은 adversarial training 구조에서 잠재 표현 $Y$의 분포 $\mathbb{U}_{\psi, \mathbb{P}}$가 사전 확률 분포 $\mathbb{V}$를 따르도록 학습됩니다.

$$(\hat{\omega}, \hat{\psi})_P = argmin_{\psi}argmax_{\phi}\hat{\mathit{D}_{\phi}(\mathbb{V}||\mathbb{U}_{\psi, \mathbb{P}} = \mathbb{E}_{\mathbb{V}}[log\mathit{D}_\phi(y)] + \mathbb{E}_{\mathbb{P}}[log(1-\mathit{D}_\phi(\mathit{E}_{\psi}(x)))]$$

세 목적함수, global and local MI maximization and prior matching은 $\alpha$, $\beta$, $\gamma$ 정규화 파라미터가 곱해져 3가지 목적을 동시에 달성합니다.

![그림5](https://hwseo95.github.io/assets/img/Article_review/Representation_learning/deepinfomax_fig5.png)

## Experiments

본 논문은 크게 3가지 실험을 진행했습니다. 첫째, 분류 정확도 평가 둘째, proxy들을 활용하여 잠재 표현 평가 셋째, 좌표 정보 또는 가림 (coordinate information and occlusions)을 추가했을 때 분류 정확도 평가.  
CIFAR10, CIFAR100, Tiny ImageNet, STL-10, CelebA 등의 이미지 데이터셋을 활용했고 비교한 모델은 여러 unsupervised representation learning 방법론들과 CPC, global 정보만 반영하는 DIM(G) ($\alpha=1, \beta=0, \gamma=1$), local 정보만 반영하는 DIM(L) ($\alpha=0, \beta=1, \gamma=0.1$)입니다. representation learning 방법론의 분류 성능을 평가하기 위해 학습한 표현을 인풋으로 하는 간단한 분류기 (fc, convnet 등)을 마지막 단에 결합했습니다.

실험 결과를 간단히 설명하면 첫번째 실험에서 DIM(L)이 다른 unsupervised representation learning 방법론보다 월등히 높은 분류 성능을 보였고 이는 fully supervised learning의 분류 성능과도 comparable하다고 주장합니다. 두번째 실험에서 여러 proxy를 활용하여 linear separability, nonlinear separability, mutual information, dependency 등을 평가하였고, Local한 정보와 global한 정보를 동시에 반영하는 DIM(L+G)가 전반적으로 좋은 값을 갖더라고 주장합니다. 

세번째 실험에서는 인코더가 학습하는 과정에서 인풋 데이터에 랜덤하게 occulation을 부여 (이미지 어느 부분을 가리기)하거나 feature vector $Y$를 통해 이미지 픽셀의 위치를 맞추게 하는 테스크를 추가적으로 부여했을 때 분류 성능을 평가합니다. 그 결과 이런 테스크를 부여했을 때 분류 성능이 더 높았고 occluation을 통해 이미지 전체에 공유된 정보를 잘 학습할 수 있었고 인풋 데이터에 pretext task를 부여하여 유용한 표현을 학습하는 self-supervised learning에서도 잘 작동한다는 것을 보인 것 같습니다.







