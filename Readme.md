# Work In Progress

#### This is TF2 re-implemetation of (TODO insert link) Temporal Fusion Transformer

### TODO references

mksquashfs data data.sqfs -all-root -action 'chmod(o+rX)@!perm(o+rX)

Run `./scripts/clone_original_implementation`, if you need to compare it with this repository.
Run `./scripts/download_<expriment name>_data`, to download raw data from experiment.

### Uploading dataset

```shell
gcloud auth login
cd gcs
terraform init
terraform plan
terraform apply
cd ..
gsutil cp -r data gs://tf2_tft/
```

```
/usr/local/lib/python3.10/dist-packages/temporal_fusion_transformer/modeling.py in tf__call(self, inputs, **kwargs)
     54                 (historical_features, historical_flags, _) = ag__.converted_call(ag__.ld(self).historical_variable_selection, (ag__.converted_call(ag__.ld(dict), (), dict(inputs=ag__.ld(historical_inputs), context=ag__.ld(static_context)['enrichment']), fscope),), None, fscope)
     55                 (future_features, future_flags, _) = ag__.converted_call(ag__.ld(self).future_variable_selection, (ag__.converted_call(ag__.ld(dict), (), dict(inputs=ag__.ld(future_inputs), context=ag__.ld(static_context)['enrichment']), fscope),), None, fscope)
---> 56                 (history_lstm, state_h, state_c) = ag__.converted_call(ag__.ld(self).historical_features_lstm, (ag__.ld(historical_features),), dict(initial_state=[ag__.ld(static_context)['state_h'], ag__.ld(static_context)['state_c']]), fscope)
     57                 future_lstm = ag__.converted_call(ag__.ld(self).future_features_lstm, (ag__.ld(future_features),), dict(initial_state=[ag__.ld(state_h), ag__.ld(state_c)]), fscope)
     58                 decoder_in = ag__.converted_call(ag__.ld(dict), (), dict(lstm_outputs=ag__.converted_call(ag__.ld(tf).concat, ([ag__.ld(history_lstm), ag__.ld(future_lstm)],), dict(axis=1), fscope), input_embeddings=ag__.converted_call(ag__.ld(tf).concat, ([ag__.ld(historical_features), ag__.ld(future_features)],), dict(axis=1), fscope), context_vector=ag__.ld(stati...

TypeError: in user code:

File "/usr/local/lib/python3.10/dist-packages/keras/engine/training.py", line 1284, in train_function  *
    return step_function(self, iterator)
File "/usr/local/lib/python3.10/dist-packages/keras/engine/training.py", line 1268, in step_function  **
    outputs = model.distribute_strategy.run(run_step, args=(data,))
File "/usr/local/lib/python3.10/dist-packages/keras/engine/training.py", line 1249, in run_step
    outputs = model.train_step(data)
File "/usr/local/lib/python3.10/dist-packages/keras/engine/training.py", line 1050, in train_step
    y_pred = self(x, training=True)
File "/usr/local/lib/python3.10/dist-packages/keras/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
File "/tmp/__autograph_generated_file1yp6i770.py", line 56, in tf__call
    (history_lstm, state_h, state_c) = ag__.converted_call(ag__.ld(self).historical_features_lstm, (ag__.ld(historical_features),), dict(initial_state=[ag__.ld(static_context)['state_h'], ag__.ld(static_context)['state_c']]), fscope)

TypeError: Exception encountered when calling layer 'temporal_fusion_transformer' (type TemporalFusionTransformer).

in user code:

    File "/usr/local/lib/python3.10/dist-packages/temporal_fusion_transformer/modeling.py", line 191, in call  *
        history_lstm, state_h, state_c = self.historical_features_lstm(
    File "/usr/local/lib/python3.10/dist-packages/keras/layers/rnn/base_rnn.py", line 626, in __call__  **
        return super().__call__(inputs, **kwargs)
    File "/usr/local/lib/python3.10/dist-packages/keras/utils/traceback_utils.py", line 70, in error_handler
        raise e.with_traceback(filtered_tb) from None

    TypeError: Exception encountered when calling layer 'historical_features_lstm' (type LSTM).
    
    Value passed to parameter 'input' has DataType bfloat16 not in list of allowed values: float16, float32, float64
    
    Call arguments received by layer 'historical_features_lstm' (type LSTM):
      • inputs=tf.Tensor(shape=(64, 168, 160), dtype=bfloat16)
      • mask=None
      • training=True
      • initial_state=['tf.Tensor(shape=(64, 160), dtype=bfloat16)', 'tf.Tensor(shape=(64, 160), dtype=bfloat16)']


Call arguments received by layer 'temporal_fusion_transformer' (type TemporalFusionTransformer):
  • inputs={'static': 'tf.Tensor(shape=(64, 1), dtype=int32)', 'known_real': 'tf.Tensor(shape=(64, 192, 3), dtype=float32)'}
  • kwargs={'training': 'True'}
```