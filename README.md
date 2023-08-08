# CapNet-Accelerator
Recently, variants of Convolutional Neural Networks (CNNs) have played an important role in classification tasks and are becoming popular in different domains. Capsule networks are introduced to overcome the two drawbacks of CNNs. These two drawbacks are the weaknesses of CNNs to classify pictures based on the spatial properties of their features and not being able to tolerate a picture’s rotation. Capsule networks perform their computations on vectors to mitigate these problems. Processing vectors needs tremendous computation power; therefore, hardware designs can be used to achieve higher performance in terms of processing time and even energy consumption.

In this project, we propose a hardware accelerator for capsule networks. Some approximations are employed in computations to make the design more efficient. We use our proposed design to classify MNIST and CIFAR-10 datasets. The accuracies of classification for MNIST and CIFAR-10 are 99.17% and 71.79%, respectively; These are, on average, 0.8% deviated from the baseline capsule model.
