from transformer_srl import dataset_readers, models, predictors

predictor = predictors.SrlTransformersPredictor.from_path("/home/lzj/lzj/plug4MSG/ckpts/srl/srl_bert_base_conll2012.tar.gz", "transformer_srl")
print(predictor.predict(
  sentence="Did Uriah honestly think he could beat the game in under three hours?"
))
