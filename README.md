# Predicting Issues' Labels with RoBERTa

Many thanks to Red Hat's <a href="#"><img src="https://github.com/GiorgosKarantonis/images/blob/master/label_bot/red%20hat.png" width="21"></a> AICoE **[Thoth Team](https://github.com/thoth-station)** for giving me the opportunity to create this model while interning there! 
**Thoth is a group of *amazing* people that use Artificial Intelligence to create cool tools that make the lives of developers easier.**

## Introduction
**This multilabel classifier can predict any combination of *bug*, *question* and *enhancement*, using [RoBERTa](https://arxiv.org/abs/1907.11692), one of the best performing NLP models**. Additionally, the predefined set of labels can be easily extended to almost any set of labels thanks to the incorporated paraphrase detection component; you can simply specify their desired labels and use this component to map them to actual labels in the dataset. Finally, it is completely straightforward to define your own aliases for each one of the predefined labels by mapping the output probabilities to the labels of your choice. 

Utilizing the `app.py` endpoint, that uses the GitHub API, you can run this classifier to your personal or your organization's repos automating your operations. 

As far as I know this is the best performing implementation on this task. 🚀

<p align="center">
  <a href="#">
    <img src="https://github.com/GiorgosKarantonis/images/blob/master/label_bot/demo.gif" width="100%">
  </a>
</p>


## Performance
The original dataset is highly imbalanced which means that a classifier may look good just by learning to predict well the most probable label(s). Such an example is the, otherwise great, [Issue Label Bot](https://github.com/machine-learning-apps/Issue-Label-Bot) which *performs multiclass classification* and achieves a high overall accuracy due to the fact that it has a good accuracy on the two classes that dominate the dataset. If we assumed a uniform distribution of the labels (and also that the performance of the classifier wouldn't change) the overall accuracy would drop ~10%! 

My implementation manages to completely overcome this issue and **achieves equally high scores for all the distinct labels while extending the capabilities of the classifier to multilabel predictions**! More specifically, the average per label **accuracy is approximately 15% higher**, while the multilabel exact match accuracy is approximately 5% higher than the multiclass accuracy of the Issue Label Bot, **using only about 9% of the dataset** and without tweaking the decision threshold! 

You can have a look at the [`notebooks/stats.ipynb`](https://github.com/GiorgosKarantonis/Github-Issues-Classifier/blob/master/notebooks/stats.ipynb) notebook for a detailed performance report on various different metrics. 


## The Dataset
The original dataset consists of more than 3 million examples of GitHub issues and each one is described by its url, its repo, its title, its body and the assigned labels. Although the size of the dataset can be a big advantage to deep learning architectures there are two severe disadvantages. The first is that some labels, for example `bug` and its derivatives, dominate the dataset which can lead to unreliable classifiers. The second is the inconsistency of the labels, which means that labels that have the same meaning appear under different names in the dataset. You can check the [`notebooks/prepare_dataset.ipynb`](https://github.com/GiorgosKarantonis/Github-Issues-Classifier/blob/master/notebooks/prepare%20dataset.ipynb) and the [`notebooks/label_mapping.ipynb`](https://github.com/GiorgosKarantonis/Github-Issues-Classifier/blob/master/notebooks/label_mapping.ipynb) notebooks to get an idea how I deal with these issues and how the paraphrase detection component is incorporated to the classifier. 

Training, validation and testing, use a **uniform** sample of the original dataset containing a total of 90k examples for each one of the classes plus all the examples that correspond to combinations of classes. This is just a small fraction of the full dataset, but thanks to the nature of fine-tuning it's enough to achieve great performance. 

If you want to create your own version or see in detail how I created mine have a look at the [`notebooks/prepare_dataset.ipynb`](https://github.com/GiorgosKarantonis/Github-Issues-Classifier/blob/master/notebooks/prepare%20dataset.ipynb) notebook. 


## The Model(s)
**There are two models that can be used for predictions**; one is the RoBERTa base model fine-tuned on the issues' dataset for multilabel classification and the other one is the same with a, small, additional head. 

The second model passes to the network the title and the body of the issue separately and their respective scores, along with the score of their combination, are fed to the additional head. The idea behind this approach is that it allows to not only polarize the output scores but to also weight differently the titles and the bodies; for example the title of a specific issue may be a great indicator of its label while the body may be noisy and disorienting so if they are weighted the same useful information may be lost. Additionally, the way I have trained it provides an additional solution to the imbalance issue. 

Another advantage of this module can be seen when working with smaller datasets. For example, when using 20k examples per class and just by setting the threshold for each class at *0.5*, the model with the additional head manages to achieve slightly better per class accuracy, better per class recall, about 8% higher exact match accuracy and more uniform scores across the classes, compared to the model without the extra head. 

The outputs of both models are the independent probabilities of each class so that you can define the thresholds yourself depending on the metric that you want to optimize. Refer to the [`notebooks/stats.ipynb`](https://github.com/GiorgosKarantonis/Github-Issues-Classifier/blob/master/notebooks/stats.ipynb) for a more detailed report of the various scores. 

Finally, **my implementation also allows you to create and train your own heads just by defining a list of PyTorch layers**; see the [`notebooks/train_custom_head.ipynb`](https://github.com/GiorgosKarantonis/Github-Issues-Classifier/blob/master/notebooks/train_custom_head.ipynb) for more details and a more thorough explanation of the way this head is trained and when you should activate it.


## Using the classifier
If you want to use my pre-trained models for predictions, make sure to download them using the `fetch_models.sh` script, but obviously you can fine-tune the models on your own datasets as well even if you have a small train set; I was able to achieve good performance even with a total of 5,000 examples. 

Also, you can manage all the dependencies using the provided Pipfile. 

In order to make predictions simply do:

```python
bot = models.Bot()  # the additional head is deactivated by default; if you want to use it just pass use_head=True
scores = bot.predict(title=some_title, body=some_body)  # the inputs can be a single string, a list of strings, a pd.Series object or a pd.DataFrame object
```
Additionally, you can use the `app.py` endpoint to use the classifier's command line interface which also allows you to also deploy the classifier to your or your organization's repos. 

You may also have a look at the [`notebooks/predict.ipynb`](https://github.com/GiorgosKarantonis/Github-Issues-Classifier/blob/master/notebooks/predict.ipynb) file for a few examples on real issues. 


## Potential Improvements
If you are interested in improving the performance of the classifier, I would recommend considering the following: 

* **Train on more data.** I ran into RAM issues with the tokenizers but if you have more than 8Gb of RAM or you find a workaround I believe you can significantly boost the performance just by using more data. 

* **Use the RoBERTa large.** Instead of the base version.

* **Noise reduction in the bodies using summarization.** I experimented with this using [BART](https://arxiv.org/abs/1910.13461), but I ran into several bugs. If you plan on working on the summarization, I would advice you to use [T5](https://arxiv.org/abs/1910.10683)'s [tensorflow version](https://huggingface.co/transformers/model_doc/t5.html#tft5forconditionalgeneration). 

* **Fine-tune a language model first.** It's unfair to think of GitHub issues as regular text due to the fact that they are usually a hybrid between real text, code and logs etc. So fine-tuning a language model on this dataset and then fine-tuning this language model on classification could yield good results. In the `demos/prepare_dataset.ipynb` file you can find a sampling method that ensures that the examples used in the language model will be different than the ones used in the classification task. 

* **Use the paraphrase detection component to get better clusters than mine.** Despite all the workarounds used, you can further improve the performance by providing a cleaner and more consistent dataset. 

Also, if you are interested in improving the performance feel free to contact me! 🙂 
