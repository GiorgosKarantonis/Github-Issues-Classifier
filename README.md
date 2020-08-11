# Predicting Issues' Labels with RoBERTa

## Introduction
This multilabel classifier can predict any combination of *bug*, *question* and *enhancement*, using [RoBERTa](https://arxiv.org/abs/1907.11692), one of the best performing NLP models, fine-tuned on the multilabel classification task. Additionally, the predefined set of labels can be easily extended to almost any set of labels thanks to the incorporated paraphrase detection component; you can simply specify their desired labels and use this component to map them to actual labels in the dataset. **This implementation is not only capable of multilabel classification but also achieves much better scores compared to other available methods.**


## Performance
The original dataset is highly imbalanced which means that a classifier may look good just by learning to predict well the most probable label(s). Such an example is the, otherwise great, [Issue Label Bot](https://github.com/machine-learning-apps/Issue-Label-Bot) which *performs multiclass classification* and achieves a high overall accuracy due to the fact that it has a good accuracy on the two classes that dominate the dataset. If we assumed a uniform distribution of the labels (and also that the performance of the classifier wouldn't change) the overall accuracy would drop ~10%! 

My implementation manages to overcome this problem and **achieves equally high scores for all the distinct labels!** Please have a look at the `demos/stats.ipynb` file for a detailed performance report on various different metrics. 


## The Dataset
Training, validation and testing, use a **uniform** sample of the original dataset containing a total of 20k examples for each one of the classes plus all the examples that correspond to combinations of classes. This is just a small fraction, ~2%, of the full dataset, but thanks to the nature of fine-tuning it's enough to achieve quite good performance. 

If you want to create your own version or see in detail how I created mine you may have a look at the `demos/prepare_dataset.ipynb` file. 


## The Model(s)
There are two models that can be used for predictions; one is the RoBERTA base model fine-tuned on the issues' dataset for multilabel classification and the other one is the same with a, small, additional head. 

The second model passes to the network the title and the body of the issue separately and their respective scores are fed to the additional head. The idea behind this approach is that it allows to not only polarize the output scores but to also weight differently the titles and the bodies; for example the title of a specific issue may be a great indicator of its label while the body may be noisy and disorienting so if they are weighted the same useful information may be lost. In general, setting the threshold for each class at *0.5*, the model with the additional head manages to achieve slightly better per class accuracy, better per class recall, about 8% higher exact match accuracy and more uniform scores across the classes, while the model without the additional head, achieves higher precision in 2 out of the 3 classes. See the `demos/stats.ipynb` for a more detailed report of the various scores. 

The outputs of both models are the, independent, probabilities of each class so that you can define the thresholds yourself depending on the metric that you want to optimize. 

**Additionally, my implementation also allows you to create and train your own heads just by defining a list of PyTorch layers; see the `demos/SOME_FILE.ipynb` for more details.**


## Using the classifier
If you want to use my pre-trained models for predictions, make sure to download the pre-trained models using the `fetch_models.sh` script, but obviously you can fine-tune the models on your own datasets as well even if you have a small train set; I was able to achieve a good performance even with a total of 5,000 examples! 

Also, you can manage all the dependencies using the provided Pipfile. 

In order to make predictions simply do:

```python
bot = models.Bot()  # the additional head is activated by default. If you don't want to use it just pass use_head=False
scores = bot.predict(title=some_title, body=some_body)  # the inputs can be a single string, a list of strings, a pd.Series object or a pd.DataFrame object
```

You may also have a look at the `demos/predict.ipynb` file for a few examples on real issues. 


## Potential Improvements
If you are interested in improving the performance of the classifier, I would recommend considering the following: 

* **Noise reduction in the bodies using summarization.** I experimented with this using [BART](https://arxiv.org/abs/1910.13461), but I ran into several bugs. If you plan on working on the summarization, I would advice you to use [T5's](https://arxiv.org/abs/1910.10683) [tensorflow version](https://huggingface.co/transformers/model_doc/t5.html#tft5forconditionalgeneration). 😉

* **Fine-tune a language model first.** It's unfair to think of GitHub issues as regular text due to the fact that they are usually a hybrid between real text, code and logs etc. So fine-tuning a language model on this dataset and then fine-tuning this language model on classification could yield good results. In the `demos/prepare_dataset.ipynb` file you can find a sampling method that ensures that the examples used in the language model will be different than the ones used in the classification task. 

* **Use the paraphrase detection component to get better clusters than mine.**

Also, if you are interested in improving the performance feel free to contact me! 🙂 
