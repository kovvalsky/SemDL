# adapted from GPT-2 script interactive_conditional_samples.py
import fire
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf

import model, sample, encoder

def run_on_prompts(model_name,prompt_list,nsamples=3):
    models_dir = os.path.expanduser(os.path.expandvars('../models'))
    batch_size = 1
    model_name=model_name
    seed=None
    batch_size=1
    length=1
    temperature=1
    top_k=0
    top_p=1
    assert nsamples % batch_size == 0
    
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    
    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)
        for p in prompt_list:
            raw_text = p
            context_tokens = enc.encode(raw_text)
            generated = 0
            print("INPUT: "+p)
            print("GPT-2 predicts:")
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)
