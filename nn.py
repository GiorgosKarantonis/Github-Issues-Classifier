tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer_dstl = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer_dstl_fast = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

model = TFBertModel.from_pretrained('bert-base-cased')
model_dstl = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


def get_embeddings(data, tokenizer=tokenizer_dstl, model=model_dstl):
    inputs = tokenizer( data, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='tf')

    outputs = model(inputs)
    logits = outputs[0]

    return logits


# to do: fix truncation issue
embeddings_title = get_embeddings(df['title'].values.tolist())  # n_examples x n_words (or threshold) x 768
embeddings_body = get_embeddings(df['body'].values.tolist())  # n_examples x n_words (or threshold) x 768


labels = np.transpose([df[c] for c in df.columns if c.startswith('label_')])


def define_classifier():
    inputs = tf.keras.Input(shape=[embeddings_body.shape[1], embeddings_body.shape[2]])
    x = inputs
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(embeddings_body.shape[2])(x)
    
    outputs = tf.keras.layers.Dense(labels.shape[1])(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


model = define_classifier()
model.compile(  optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'])


model.fit(  embeddings_body,
            labels, 
            batch_size=5, 
            epochs=20)

tf.keras.backend.clear_session()