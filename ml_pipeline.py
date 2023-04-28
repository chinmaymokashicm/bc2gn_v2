from func_code.ml import Ml


ml_obj = Ml()


model = ml_obj.prepare_model()
df_eval_train, df_eval_test = ml_obj.generate_evaluation_results(model=model)
# print("Training eval results")
# print(df_eval_train)
# print("Test eval results")
# print(df_eval_test)